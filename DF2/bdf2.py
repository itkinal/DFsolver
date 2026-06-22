from scipy.sparse.linalg import splu
from DF.dfrog_solver import Lfull



class BDF2():
    P:     dict = {}

    def __init__(self, p = None):
        if p is not None:
            for key, value in p.items():
                setattr(self, key, value)

    def bdf2_solve(self, P,p0,T,nsteps, Lfull):
        dt=T/nsteps
        IN = P["IN"]
        pp=p0.reshape(-1).copy()
        pc=splu((IN - 0.5*dt*Lfull(P,dt)).tocsc()).solve((IN + 0.5*dt*Lfull(P,0.)) @ pp)
        for n in range(1,nsteps):
            pp,pc = pc, splu((1.5*IN-dt*Lfull(P,(n+1)*dt)).tocsc()).solve(2*pc - 0.5*pp)

        return pc.reshape(P["Nx"],P["Ny"])
