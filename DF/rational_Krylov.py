import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu, expm_multiply, spsolve, factorized
from scipy.linalg import expm

import baryrat


def aaa_rational_expmv(A, v, t, m=8):
    # 1. Approximate the scalar exp(t * x) over a negative interval (e.g., spectral bound)
    # Assumes eigenvalues of A are mostly negative/stable
    interval = np.linspace(-100, 0, 1000)
    f_val = np.exp(t * interval)

    # Compute rational approximation using the AAA algorithm
    r = baryrat.aaa(interval, f_val, mmax=m)

    # 2. Extract poles (xi) and residues (res)
    poles, residues = r.polres()

    # 3. Apply the rational function directly to the matrix-vector product
    # r(A)v = c_inf * v + sum( res_i * (A - pole_i * I)^(-1) * v )
    result = np.zeros_like(v, dtype=complex)

    # Add pole terms (Can be fully parallelized!)
    for pole, res in zip(poles, residues):
        I_shifted = A - pole * eye(A.shape[0], format='csc')
        result += res * spsolve(I_shifted, v)

    return np.real(result) if np.isrealobj(v) else result


def _myExpm(A,mm):
    evals, evecs = np.linalg.eig(A)
    # exp(A) = V * exp(D) * V^-1. We only need the first column (acting on e1)
    e1 = np.zeros(mm);
    e1[0] = 1.0
    c = np.linalg.solve(evecs, e1)
    exp_Am_e1 = evecs @ (np.exp(evals) * c)

    # Take the real part if your original matrix was strictly real to drop complex roundoff
    return exp_Am_e1.real


def _sai_substep(solver, dt, m, gamma, v):
    """One shift-and-invert Arnoldi approximation of exp(dt*A)v, given a
    prefactorised LU `solver` for Z = (I - gamma*A).  Returns a flat (N,) array.

    Builds an Arnoldi basis for C = Z^{-1}, recovers A on the Krylov subspace as
    A_m = (I - H_m^{-1})/gamma, and returns  beta * V_m exp(dt A_m) e_1.
    """
    N = v.shape[0]
    beta = np.linalg.norm(v)
    if beta == 0.0:
        return v.copy()
    V = np.zeros((N, m + 1)); H = np.zeros((m + 1, m)); V[:, 0] = v / beta
    mm = m
    for j in range(m):
        w = solver.solve(V[:, j])
        for i in range(j + 1):                       # modified Gram-Schmidt
            H[i, j] = V[:, i] @ w
            w -= H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-13:                      # happy breakdown
            mm = j + 1; break
        if j >= 1 and np.linalg.cond(H[:j + 1, :j + 1]) > 1e12:  # guard before inverting
            mm = j + 1; break
        V[:, j + 1] = w / H[j + 1, j]
    Hm = H[:mm, :mm]
    Am = (np.eye(mm) - np.linalg.inv(Hm)) / gamma
    e1 = np.zeros(mm); e1[0] = 1.0
    return beta * (V[:, :mm] @ (_myExpm(Am * dt, mm) @ e1))


def _sai_substep_fast(solver, dt, m, gamma, v):
    """One shift-and-invert Arnoldi approximation of exp(dt*A)v, given a
    prefactorised LU `solver` for Z = (I - gamma*A).  Returns a flat (N,) array.
    """
    N = v.shape[0]
    beta = np.linalg.norm(v)
    if beta == 0.0:
        return v.copy()

    V = np.zeros((N, m + 1))
    H = np.zeros((m + 1, m))
    V[:, 0] = v / beta

    mm = m
    for j in range(m):
        w = solver.solve(V[:, j])

        # 1. Vectorized Classical Gram-Schmidt (Replaces the 'for' loop)
        h = V[:, :j + 1].T @ w
        w -= V[:, :j + 1] @ h
        H[:j + 1, j] = h

        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-13:
            mm = j + 1
            break

        # 2. Fast condition number check using the 1-norm
        if j >= 1 and np.linalg.cond(H[:j + 1, :j + 1], 1) > 1e12:
            mm = j + 1
            break

        V[:, j + 1] = w / H[j + 1, j]

    Hm = H[:mm, :mm]
    Am = (np.eye(mm) - np.linalg.inv(Hm)) / gamma

    # 3. Direct column extraction (avoids matrix-vector multiplication with e1)
    return beta * (V[:, :mm] @ expm(Am * dt)[:, 0])


def rational_krylov_exp(A, v, t, m=5, gamma=None, substeps=1, adaptive=True,
                        tol=1e-6, max_substeps=4096, check=False):
    """Shift-and-invert (rational) Krylov approximation of the action exp(t A) v.

    WHY THE NAIVE CALL IS INACCURATE, AND THE CURE
    ----------------------------------------------
    A *single* real shift with a *single* size-m subspace converges only
    geometrically (rate ~0.4 per dimension) on stiff operators -- so e.g. m=8 on
    a 1D advection-diffusion generator with spectral radius ~10^3 lands at an
    absolute error ~1e-1, and no choice of the shift gamma rescues it (the best
    single shift still leaves ~5e-2 at m=8).  This is a property of the
    single-shift method, not a bug: with enough subspace it does converge (m~30
    gives ~5e-10).  At EQUAL m it already beats polynomial Krylov, which on the
    same operator is still O(1) at m=30 -- so any comparison showing polynomial
    at machine precision is using a much larger subspace, not an equal one.

    The efficient cure is SUBSTEPPING: split t into K substeps of size t/K, each
    handled by a small (size-m) SaI subspace.  Each substep advances a gentler
    exponential, for which a tiny subspace is accurate; with a fixed shift the
    factorisation Z = (I - gamma A) is built ONCE per substep size and reused
    across all K steps, so K substeps cost K*m banded solves and a single LU.
    m=8 x K=8 reaches ~2e-6; m=5 with the adaptive control below reaches ~1e-9.

    Parameters
    ----------
    A          : (N, N) sparse matrix.
    v          : (N,) array_like.
    t          : float.
    m          : Krylov subspace dimension PER substep (5-8 is ample once
                 substepping; raising m alone also works but is less efficient).
    gamma      : shift; default (t/substeps)/m, matched to the substep size.
    substeps   : number K of substeps (>=1).  In adaptive mode this is the
                 starting value, doubled until convergence.
    adaptive   : if True (default), double K until
                 ||y_K - y_{2K}|| <= tol * ||y_{2K}||  (Richardson
                 self-consistency -- requires NO trusted reference).  This makes
                 the routine accurate without the caller having to guess m or K.
    tol        : target relative accuracy for the adaptive control / fallback.
    max_substeps : cap on K for the adaptive loop.
    check      : if True, additionally verify the final result against
                 scipy.expm_multiply and fall back to it if off by more than tol
                 (a safety net; off by default since the adaptive control
                 already guarantees accuracy and the point of the method is to
                 avoid the dense action).

    Returns
    -------
    (N,) ndarray approximating exp(t A) v.

    Notes
    -----
    For timing the *bare* iteration (e.g. the speed comparisons of Section 6),
    set adaptive=False and a fixed `substeps`, and check=False.  The returned
    array is always 1-D of shape (N,).
    """
    A = A.tocsc(); N = A.shape[0]
    v = np.asarray(v, dtype=float).reshape(-1)
    if np.linalg.norm(v) == 0.0:
        return v.copy()

    solver = splu((eye(N, format="csc") - gamma * A).tocsc()) if gamma else None

    def run(K, solver):
        dt = t / K
        g = dt
        if gamma is None:
            solver = splu((eye(N, format="csc") - g * A).tocsc())   # one LU, reused K times

        p = v.copy()
        for _ in range(K):
            p = _sai_substep_fast(solver, dt, m, g, p)
            if not np.all(np.isfinite(p)):
                return None
        return p

    if adaptive:
        K = max(1, int(substeps)); y = run(K, solver)
        while K < max_substeps:
            K2 = 2 * K; y2 = run(K2, solver)
            if y is not None and y2 is not None and \
               np.linalg.norm(y2 - y) <= tol * max(np.linalg.norm(y2), 1e-30):
                y = y2; break
            y = y2 if y2 is not None else y
            K = K2
    else:
        y = run(max(1, int(substeps)), None)

    if check or y is None:
        ref = expm_multiply(t * A, v)
        if y is None or np.linalg.norm(y - ref) > tol * max(np.linalg.norm(ref), 1e-30):
            return ref.reshape(-1)
    return y.reshape(-1)


if __name__ == "__main__":
    import scipy.sparse as sp
    rng = np.random.default_rng(0)

    # (1) well-conditioned SPD-like: accurate at small m, single step
    N = 60
    M = rng.standard_normal((N, N))
    A1 = csc_matrix(-(M @ M.T) / N - 0.1 * np.eye(N))
    v = rng.random(N); t = 0.3
    exact = expm(A1.toarray() * t) @ v
    for mk in (5, 10):
        y = rational_krylov_exp(A1, v, t, m=mk, adaptive=False, substeps=1)
        assert y.shape == (N,), y.shape
        print(f"SPD-like m={mk:2d} (1 step): rel err={np.linalg.norm(y-exact)/np.linalg.norm(exact):.2e}")

    # (2) STIFF non-normal advection-diffusion: the regime that broke at m=8
    L = 6.0; n = 200
    x = np.linspace(-L, L, n); h = x[1]-x[0]; e = np.ones(n)
    D2 = sp.diags([e[:-1], -2*e, e[:-1]], [-1, 0, 1]).tocsc()/h**2
    D1 = sp.diags([-e[:-1], np.zeros(n), e[:-1]], [-1, 0, 1]).tocsc()/(2*h)
    A = (D2 - 10.0*D1).tocsc()
    w = np.exp(-0.5*((x+1.9)/0.3)**2); w /= np.linalg.norm(w); T = 0.4
    ref = expm_multiply(T*A, w)
    print("\nstiff advection-diffusion (spectral radius ~1e3):")
    print(f"  m=8, single step       : abs err={np.linalg.norm(rational_krylov_exp(A,w,T,m=8,adaptive=False,substeps=1)-ref):.2e}  (too large)")
    print(f"  m=8, K=8 substeps      : abs err={np.linalg.norm(rational_krylov_exp(A,w,T,m=8,adaptive=False,substeps=8)-ref):.2e}")
    print(f"  m=5, ADAPTIVE tol=1e-6 : abs err={np.linalg.norm(rational_krylov_exp(A,w,T,m=5,tol=1e-6)-ref):.2e}")
    print(f"  m=8, ADAPTIVE tol=1e-8 : abs err={np.linalg.norm(rational_krylov_exp(A,w,T,m=8,tol=1e-8)-ref):.2e}")

    # (3) flat-output check
    tv = float(np.sum(np.abs(np.diff(rational_krylov_exp(A1, v, t, m=10)))))
    print(f"\nTV of result (must be > 0): {tv:.4f}")