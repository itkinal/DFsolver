"""
1D Fokker-Planck experiments on the Ornstein-Uhlenbeck (OU) process, which has a
closed-form transition density (Gaussian) and stationary law -- an exact
reference for convergence, positivity, and conservation tests.

OU SDE:    dX = -kappa (X - m) dt + sigma dW
FPE:       p_t = kappa d/dx[(x-m) p] + (sigma^2/2) d^2/dx^2 p
         = -d/dx[mu(x) p] + d^2/dx^2[D p],   mu(x) = -kappa(x-m),  D = sigma^2/2.
Transition density from delta at x0:
   p(x,t) = N( mean(t), var(t) ),
   mean(t) = m + (x0 - m) e^{-kappa t},
   var(t)  = (sigma^2 / (2 kappa)) (1 - e^{-2 kappa t}).
Stationary law: N(m, sigma^2/(2 kappa)).

Discretisations (divergence form, absorbing BCs p_1=p_n=0), matching the paper:
  * DF  (sec:disc1d, ssec:em): advection by 2nd-order UPWIND F^B_2=(3,-4,1)/(2h)
        in the upwind direction (sign of mu(x)), diffusion centred. -L is an
        EM-matrix => e^{dt L} eventually positive (tau_0>0), Godunov ripples for
        small dt.
  * CTR (ssec:advdom): advection by 2nd-order CENTRED (1,0,-1)/(2h), diffusion
        centred. M-matrix iff Pe<2 (positive for all dt); for Pe>2 loses even
        the EM structure.
Both advance in time by the exact action e^{dt L} (rational-Krylov proxy via
scipy expm_multiply), so positivity differences are purely SPATIAL.
For reference we also include first-order UPWIND advection (positive but O(h)).
"""
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, identity
from scipy.sparse.linalg import expm_multiply, splu


def ou_grid(n, xlo, xhi):
    """n interior nodes; absorbing (Dirichlet 0) at the two endpoints x_1,x_n.
    We store the FULL grid of n points and keep p_1=p_n=0; the operator acts on
    interior nodes i=2..n-1 (0-based 1..n-2)."""
    x = np.linspace(xlo, xhi, n)
    h = x[1] - x[0]
    return x, h


# def mu_ou(x, kappa, m):
#     return -kappa * (x - m)        # drift; sign varies: mu>0 for x<m, mu<0 for x>m

def mu_ou(x, kappa, m):
    # Double-well V(x)=kappa(x^2-m^2)^2, drift mu=-V'=-4 kappa x (x^2-m^2).
    # build_L_1d discretises -d/dx[mu p], so mu must be -V' (confining toward the
    # wells at x=+-m).  The +sign would be anti-confining and leak mass to the
    # absorbing walls.  For the LINEAR Ornstein-Uhlenbeck tests use the line below.
    return -4*kappa*x*(x**2 - m**2)
    # return -kappa*(x - m)               # linear OU drift (use with ou_exact)


def build_L_1d(x, h, kappa, m, D, scheme, exam = 0):
    """Assemble L (so that p_t = L p) on interior nodes with absorbing BCs.
    scheme in {'DF','CTR','UW1'}:
      DF  : 2nd-order upwind advection (directional), centred diffusion
      CTR : 2nd-order centred advection, centred diffusion
      UW1 : 1st-order upwind advection, centred diffusion
    Divergence form: we discretise -d/dx[mu p] + d^2/dx^2[D p], coefficients
    (mu_i, D) multiply p at the stencil nodes (here D constant)."""
    n = len(x)
    mu = mu_ou(x, kappa, m)
    L = lil_matrix((n, n))
    for i in range(1, n - 1):                      # interior nodes (0-based)
        # --- diffusion: + d^2/dx^2[D p] = D (p_{i-1}-2p_i+p_{i+1})/h^2 (D const)
        L[i, i - 1] += D / h**2
        L[i, i]     += -2 * D / h**2
        L[i, i + 1] += D / h**2
        # --- advection: -d/dx[mu p].  Discretise d/dx[mu p] then negate.
        if scheme == 'CTR':
            # centred: (mu p)_x ~ ( (mu p)_{i+1} - (mu p)_{i-1} )/(2h)
            L[i, i + 1] += -mu[i + 1] / (2 * h)
            L[i, i - 1] += +mu[i - 1] / (2 * h)
        elif scheme == 'DF':
            # 2nd-order upwind in the upwind direction (sign of mu_i)
            if mu[i] > 0:   # backward F^B_2 = (3 f_i -4 f_{i-1}+ f_{i-2})/(2h)
                if i - 2 >= 0:
                    L[i, i]     += -3 * mu[i] / (2 * h)
                    L[i, i - 1] += +4 * mu[i - 1] / (2 * h)
                    L[i, i - 2] += -1 * mu[i - 2] / (2 * h)
                else:       # near-boundary fallback: 1st-order backward
                    L[i, i]     += -mu[i] / h
                    L[i, i - 1] += +mu[i - 1] / h
            else:           # mu<0: forward F^F_2 = (-3 f_i +4 f_{i+1} - f_{i+2})/(2h)
                if i + 2 <= n - 1:
                    L[i, i]     += +3 * mu[i] / (2 * h)
                    L[i, i + 1] += -4 * mu[i + 1] / (2 * h)
                    L[i, i + 2] += +1 * mu[i + 2] / (2 * h)
                else:       # near-boundary fallback: 1st-order forward
                    L[i, i]     += +mu[i] / h
                    L[i, i + 1] += -mu[i + 1] / h
        elif scheme == 'UW1':
            if mu[i] > 0:   # backward 1st order
                L[i, i]     += -mu[i] / h
                L[i, i - 1] += +mu[i - 1] / h
            else:           # forward 1st order
                L[i, i]     += +mu[i] / h
                L[i, i + 1] += -mu[i + 1] / h
        else:
            raise ValueError(scheme)
    # restrict to interior block (drop rows/cols 0 and n-1: absorbing)
    Lint = L[1:n - 1, 1:n - 1].tocsc() if exam == 0 else L.tocsc()
    return Lint


def gaussian_pdf(x, mean, var):
    return np.exp(-0.5 * (x - mean)**2 / var) / np.sqrt(2 * np.pi * var)


def ou_exact(x, t, x0, kappa, m, sigma):
    mean = m + (x0 - m) * np.exp(-kappa * t)
    var = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * t))
    return gaussian_pdf(x, mean, var)


def step_exp(p_int, L, dt):
    """exact action e^{dt L} p (rational-Krylov proxy)."""
    return expm_multiply(dt * L, p_int)


def l1(p, h):
    return h * np.sum(np.abs(p))


def l2(p, h):
    return np.sqrt(h * np.sum(p**2))


# ------------------------------------------------------------------ validation
if __name__ == "__main__":
    # OU parameters
    kappa, m, sigma = 1.0, 0.0, 1.0
    D = sigma**2 / 2
    # start from a narrow Gaussian (approx delta) so the exact transition law applies
    x0 = 1.0
    t0 = 0.15                       # initial "age" so IC is a resolved Gaussian
    xlo, xhi = -6.0, 6.0

    print("OU validation: peak Peclet at the grid edges, and exact-solution match")
    for n in (201, 401):
        x, h = ou_grid(n, xlo, xhi)
        mu = mu_ou(x, kappa, m)
        Pe = np.abs(mu) * h / D
        xi = x[1:-1]
        p = ou_exact(xi, t0, x0, kappa, m, sigma)
        # advance by T with DF, compare to exact
        T = 0.5
        dt = 0.5 * h
        L = build_L_1d(x, h, kappa, m, D, 'DF')
        nst = int(round(T / dt)); ddt = T / nst
        for _ in range(nst):
            p = step_exp(p, L, ddt)
        pe = ou_exact(xi, t0 + T, x0, kappa, m, sigma)
        print(f"  n={n:4d}  h={h:.4f}  max Pe={Pe.max():.2f}  "
              f"||DF-exact||_2={l2(p-pe,h):.3e}  mass={l1(p,h):.6f}")
