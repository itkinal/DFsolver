import numpy as np
from typing import Callable
from dataclasses import field
from scipy.integrate import solve_ivp

class Gaussian():
    thx:    Callable[[float], float] = lambda t: t
    thy:    Callable[[float], float] = lambda t: t
    D:      Callable[[float], float] = lambda t: t
    m:      np.ndarray = None

    def __init__(self, p = None):
        if p is not None:
            for key, value in p.items():
                setattr(self, key, value)

    def exact_moments(self, mu0,C0,T):
        def rhs(t,z):
            ax,ay = self.thx(t), self.thy(t)
            D = self.D(t)
            mx,my,Cxx,Cxy,Cyy = z
            p = [ -ax*(mx - self.m[0]), -ay*(my - self.m[1]), -2*ax*Cxx + D[0,0],
                  -(ax + ay)*Cxy + D[0,1], -2*ay*Cyy+D[1,1]
            ]
            return p

        s=solve_ivp(rhs,(0,T),[mu0[0], mu0[1], C0[0,0], C0[0,1], C0[1,1]], rtol=1e-12, atol=1e-14)
        z=s.y[:,-1]
        return np.array([z[0],z[1]]), np.array([[z[2],z[3]],[z[3],z[4]]])

    @staticmethod
    def gaussian(X,Y,mu,C):
        Ci = np.linalg.inv(C)
        det = np.linalg.det(C)
        dx = X - mu[0]
        dy = Y - mu[1]
        p = np.exp(-0.5*(Ci[0,0]*dx*dx + 2*Ci[0,1]*dx*dy + Ci[1,1]*dy*dy))/(2*np.pi*np.sqrt(det))
        return p

