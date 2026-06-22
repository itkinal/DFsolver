# =============================================================================
#  dfrog_solver.py -- Diagonal Frog (DF) FORM (C) solver + BDF2 reference.
#
#  Regenerates, for the manuscript:
#    tab:dfrog2d-conv   Regime I (rho=0.8+0.1cos), joint Dt~h refinement
#    tab:dfrog2d-acc    Regime II (advection-dominated), positivity & mass
#    tab:conv           constant-coeff eq:exp-model (rho=0.8), spatial+temporal
#    fig:dfrog2d        3-panel figure  -> figs/dfrog2d_pos.png
#
#  FORM (C) central solve of (I - tau A_xy) V = b, tau = dt/2, A_xy one-sided:
#    T_x = P I - rho sqrt(tau) s1 F_x^fwd ,  T_y = Q I + sqrt(tau) s2 F_y^bwd ,
#    P = Q-scaled by beta (eq:PQbeta), iterate enters ONLY via -p^[k]:
#      T_y p* = alpha^+_2 b - p^[k] + alpha (tau A_xy b);   T_x p^{k+1} = p* .
#    alpha^+_2 = (PQ+1)I - Q rho sqrt(tau) s1 F_x^bwd + P sqrt(tau) s2 F_y^fwd  (Scheme B, reversed)
#    alpha     = PQ I    - Q rho sqrt(tau) s1 F_x^fwd + P sqrt(tau) s2 F_y^bwd  (= M+I, factor orientation)
#    The kept term alpha(tau A_xy b) is computed once/step (O(N)); it restores
#    second-order TIME.  Contraction q <= 4/PQ (mesh-robust, beta-robust).
# =============================================================================
import inspect
from typing import Callable, Any
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm
from scipy.sparse import diags, kron, identity, csc_matrix
from scipy.sparse.linalg import splu


class DFSolver:
    Nx:     int         = 120
    Ny:     int         = 120
    L:      int         = 1
    T:      float  = 1.0
    dt:     float  = 1.0

    thx:    Callable[[float], float] = lambda t: t
    thy:    Callable[[float], float] = lambda t: t
    D:      Callable[[float], float] = lambda t: t
    m:      np.ndarray = None

    expTI: Callable[[Any], Any] = staticmethod(expm)

    schemes: SimpleNamespace = None


    def __init__(self, p = None):
        if p is not None:
            for key, value in p.items():
                setattr(self, key, value)

        self.schemes = SimpleNamespace(
            CN = lambda p, prev=None: splu((p['IN'] - 0.5 * p['dt'] * p['Axy']).tocsc()).solve(
                (p['Pg'] + 0.5 * p['dt'] * (p['P']["Ffx"] @ p['Pg'] @ p['P']["Fby"].T)).reshape(-1)
            ).reshape(p['sh']),

            BE = lambda p, prev=None: splu((p['IN'] - p['dt'] * p['Axy']).tocsc()).solve(
                p['Pg'].reshape(-1)
            ).reshape(p['sh']),

            BDF2 = lambda p, prev=None: (
                splu((p['IN'] - p['dt'] * p['Axy']).tocsc()).solve(p['Pg'].reshape(-1)).reshape(p['sh'])
                if prev is None else
                    splu((p['IN'] - (2. / 3.) * p['dt'] * p['Axy']).tocsc()).solve(
                        ((4. / 3.) * p['Pg'] - (1. / 3.) * prev).reshape(-1)
                    ).reshape(p['sh'])
                )
            )

    def Lx1d(self, P, t):
        return self.thx(t) * P["Adv0x"] + 0.5 * self.D(t)[0, 0] * P["D2x"]

    def Ly1d(self, P, t):
        return self.thy(t) * P["Adv0y"] + 0.5 * self.D(t)[1, 1] * P["D2y"]

    def Lfull(self, P, t):
        D = self.D(t)
        fullL = self.thx(t)*P["KAx"] + 0.5*D[0, 0]*P["KDx"] + self.thy(t)*P["KAy"] + 0.5*D[1, 1]*P["KDy"] + D[0, 1]*P["Kxy"]
        return fullL.tocsc()

    # --------------------------- 1D stencils -------------------------------------
    @staticmethod
    def Ffwd(n, h):
        return diags([np.full(n, -3/(2*h)), np.full(n - 1, 4/(2*h)), np.full(n - 2, -1/(2*h))],
                     [0, 1, 2]).toarray()

    @staticmethod
    def Fbwd(n, h):
        return diags([np.full(n, 3/(2*h)), np.full(n - 1, -4/(2*h)), np.full(n - 2, 1/(2*h))],
                     [0, -1, -2]).toarray()

    @staticmethod
    def D2c(n, h):
        mat = diags([np.full(n - 1, 1/h**2), np.full(n, -2/h**2), np.full(n - 1, 1/h**2)], [-1, 0, 1])
        return mat.toarray()

    @classmethod
    def adv_base(cls,xs, h):
        n = len(xs)
        return np.where((xs <= 0.)[:, None], cls.Fbwd(n, h), cls.Ffwd(n, h)) @ np.diag(xs)  # 2nd-order upwind

    @classmethod
    def build(cls, Nx, Ny, L):
        hx = 2 * L / (Nx - 1)
        hy = 2 * L / (Ny - 1)
        xs = np.linspace(-L, L, Nx)
        ys = np.linspace(-L, L, Ny)
        Ffx = cls.Ffwd(Nx, hx)
        Fby = cls.Fbwd(Ny, hy)
        Fbx = cls.Fbwd(Nx, hx)
        Ffy = cls.Ffwd(Ny, hy)
        Ix = identity(Nx)
        Iy = identity(Ny)
        P = dict(
            hx=hx, hy=hy, xs=xs, ys=ys, Adv0x=cls.adv_base(xs, hx), D2x=cls.D2c(Nx, hx),
            Adv0y=cls.adv_base(ys, hy), D2y=cls.D2c(Ny, hy), Ffx=Ffx, Fby=Fby, Fbx=Fbx, Ffy=Ffy,
            Ffx_sp=csc_matrix(Ffx), Fby_sp=csc_matrix(Fby), INx=identity(Nx, format="csc"),
            INy=identity(Ny, format="csc"), Nx=Nx, Ny=Ny,
            KAx=kron(csc_matrix(cls.adv_base(xs, hx)), Iy, format="csc"),
            KDx=kron(csc_matrix(cls.D2c(Nx, hx)), Iy, format="csc"),
            KAy=kron(Ix, csc_matrix(cls.adv_base(ys, hy)), format="csc"),
            KDy=kron(Ix, csc_matrix(cls.D2c(Ny, hy)), format="csc"),
            Kxy=kron(csc_matrix(Ffx), csc_matrix(Fby), format="csc"), IN=identity(Nx * Ny, format="csc")
        )
        return P

    def _strang_central(self,P,p,dt,tm,central,prev):
        """
        One full Strang step with the cross factor advanced by an EXACT rational
        map  in {CN, BE, BDF2}, applied (with band-structured A_xy) by a
        direct 2D sparse solve.   carries the previous central INPUT, used by
        the two-step BDF2 map.  Returns (p_next, this_step_central_input)
        """

        Ex = self.expTI(0.5*dt*self.Lx1d(P,tm))
        Ey = self.expTI(0.5*dt*self.Ly1d(P,tm))
        Pg = Ex @ p
        D = self.D(tm)
        p = dict(
            P = P,
            Pg = Pg @ Ey.T,
            dt = dt,
            Axy = (D[0,1]*P["Kxy"]).tocsc(),
            IN = P["IN"],
            sh = Pg.shape
        )
        scheme_func = getattr(self.schemes, central, None)
        if scheme_func is None:
            raise ValueError(f'This scheme {central} is not implemented')

        out = scheme_func(p, prev=prev)
        newprev = Pg
        Pg = out @ Ey.T
        Pg = Ex @ Pg
        return Pg, newprev



    @staticmethod
    def solveC(b,dt,D,P,tol=1e-9,maxit=600):
        Dxy = D[0,1]
        s1 = np.sqrt(D[0,0])
        s2 = np.sqrt(D[1,1])
        rho = Dxy/(s1*s2)
        tau = 0.5*dt
        sq = np.sqrt(tau)
        s = tau*Dxy
        wbar = abs(rho)*s1 + s2
        beta = max(10*wbar,2*(wbar + np.sqrt(P["hx"]*P["hy"]/tau)))
        Pp = beta*sq/P["hx"]
        Qq = beta*sq/P["hy"]
        PQ = Pp*Qq
        lux = splu((Pp*P["INx"] - rho*sq*s1*P["Ffx_sp"]).tocsc())   # T_x
        luy = splu((Qq*P["INy"] + sq*s2*P["Fby_sp"]).tocsc())   # T_y
        Ffx = P["Ffx"]
        Fby = P["Fby"]
        Fbx = P["Fbx"]
        Ffy = P["Ffy"]

        def alpha(w):
            return PQ*w - Qq*rho*sq*s1*(Ffx@w) + Pp*sq*s2*(w@Fby.T)   # alpha = M+I (factor orientation)

        crb = s*(Ffx@b@Fby.T)                                                # tau A_xy b
        ab = ((PQ+1.)*b - Qq*rho*sq*s1*(Fbx@b) + Pp*sq*s2*(b@Ffy.T)) + alpha(crb)  # alpha^+_2 b + alpha(tau A_xy b)
        p = b.copy()
        for k in range(1,maxit+1):
            pn = lux.solve(luy.solve((ab-p).T).T)
            if np.max(np.abs(pn-p))<=tol*(1+np.max(np.abs(pn))):
                return pn,k,PQ
            p = pn

        return pn,maxit,PQ

    def _multExp(self, P, tm, dt, Pg,  flag = 'F', Ex = None, Ey = None):
        num_args = len(inspect.signature(self.expTI).parameters)
        if flag == 'F':
            if num_args == 1:
                if Ex is None and Ey is None:
                    Ex = self.expTI(0.5 * dt * self.Lx1d(P, tm))
                    Ey = self.expTI(0.5 * dt * self.Ly1d(P, tm))
                Pg = Ex @ Pg
                Pg = Pg @ Ey.T
            else:
                Pg = self.expTI(0.5 * dt * self.Lx1d(P, tm), Pg)
                Pg = self.expTI(0.5 * dt * self.Ly1d(P, tm), Pg.T).T
        else:
            if num_args == 1:
                if Ex is None and Ey is None:
                    Ex = self.expTI(0.5 * dt * self.Lx1d(P, tm))
                    Ey = self.expTI(0.5 * dt * self.Ly1d(P, tm))

                Pg = Ex @ Pg
                Pg = Pg @ Ey.T
            else:
                Pg = self.expTI(0.5 * dt * self.Ly1d(P, tm), Pg.T).T
                Pg = self.expTI(0.5 * dt * self.Lx1d(P, tm), Pg)

        return Pg, Ex, Ey

    def df_solve(self, P,p0,T,nsteps,track_min=False):
        Pg=p0.copy()
        dt=T/nsteps
        Ffx=P["Ffx"]
        Fby=P["Fby"]
        gmin=np.inf
        sw=0
        for n in range(nsteps):
            tm = (n+0.5)*dt
            D = self.D(tm)
            s = 0.5*dt*D[0,1]
            Pg, Ex, Ey = self._multExp(P, tm, dt, Pg, 'F')

            b = Pg + s*(Ffx @ Pg @ Fby.T)
            Pg,it,_ = self.solveC(b,dt,D,P)
            sw = max(sw,it)
            Pg, Ex, Ey = self._multExp(P, tm, dt, Pg, 'B', Ex, Ey)
            if track_min:
                gmin = min(gmin,Pg.min())

        return (Pg,gmin,sw) if track_min else (Pg,sw)

