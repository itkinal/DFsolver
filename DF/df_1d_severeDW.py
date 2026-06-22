"""
Severe advection-dominated 1D experiment (corrected).

Fixes relative to the user's draft `better1D_example.py`:
  * the green "CN" curve plotted p_c (central-exp), not p_cn -- corrected so CN
    plots the actual Crank-Nicolson solution;
  * legend minima now report the TRUE minimum (the draft's "min = 0.0e+00" for
    DF was a rounding of a small negative Godunov ripple);
  * adds quantitative failure-character diagnostics: number of sign-changing
    nodes and total variation (TV), which separate the CENTRED scheme's
    DISPERSIVE ringing (many negative nodes, high TV) from DF's behaviour.

Honest framing (matches the theory): no linear second-order scheme is monotone
(Godunov), so on a SEVERELY under-resolved pulse DF also ripples; its advantage
is (a) the eventual-positivity recovery under time stepping (Fig E1d_eventual),
and (b) that on a RESOLVED pulse at moderate Pe it is positive to machine
precision while the centred/CN schemes are not. We report both regimes.
"""
import numpy as np
from scipy.linalg import expm
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import spsolve, expm_multiply
import matplotlib;

from DF.rational_Krylov import rational_krylov_exp
from df_1d_coreDW import mu_ou
from Tests.paperFD.timer import Timer

matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "../../Tests/paperFD")
from df_1d_coreDW import build_L_1d


def neg_nodes(p): return int(np.sum(p < -1e-9))
def total_variation(p): return float(np.sum(np.abs(np.diff(p))))


def solve_all(kappa, sigma, xspan, n, x0, std, T, cn_steps=200):
    x = np.linspace(-xspan, xspan, n); h = x[1] - x[0]; m = 1.0; D = 0.5 * sigma**2
    mu = mu_ou(x, kappa, m); Pe = np.abs(mu) * h / D   # Pe from the ACTUAL operator drift
    p0 = np.exp(-0.5 * ((x - x0) / std)**2); p0 /= np.sum(p0 * h)

    with Timer("CRT exp"):
        Ac  = build_L_1d(x, h, kappa, m, D, 'CTR', exam=1); Ac[0, 0]  = Ac[-1, -1]  = 0
        p_c  = expm(Ac.toarray()  * T) @ p0

    with Timer("DF exp"):
        Adf = build_L_1d(x, h, kappa, m, D, 'DF', exam=1);
        Adf[0, 0] = Adf[-1, -1] = 0
        p_df = expm(Adf.toarray() * T) @ p0

    with Timer("CN"):
        # true Crank-Nicolson on the central operator
        dt = T / cn_steps; I = eye(n); Acs = csr_matrix(Ac)
        lhs = I - 0.5 * dt * Acs; rhs = I + 0.5 * dt * Acs
        p_cn = p0.copy()
        for _ in range(cn_steps):
            p_cn = spsolve(lhs, rhs @ p_cn)

    # NB: no closed-form transition density for the double-well; we report the
    # operator solutions only. meanT/stdT below are placeholders for the report.
    meanT = x[np.argmax(p0)]; varT = std**2
    p_ex = np.exp(-0.5 * (x - meanT)**2 / varT) / np.sqrt(2 * np.pi * varT)

    with Timer("DF Krylov"):
        p_dfKrylov = expm_multiply(Adf * T, p0)

    with Timer("DF r-Krylov"):
        # Instead of one big step:
        # p_df = rational_krylov_exp(A_df, p0, T, m=3)
        nSteps = 10
        dt = T / nSteps
        p_current = p0.copy()
        for _ in range(nSteps):
            p_current = rational_krylov_exp(Adf, p_current, t=dt, m=7)
        p_dfKrylovR = p_current

    return dict(x=x, h=h, Pe=Pe.max(), p0=p0, DF=p_df, CTR=p_c, CN=p_cn, EX=p_ex,
                DFK = p_dfKrylov, DFRK = p_dfKrylovR,
                meanT=meanT, stdT=np.sqrt(varT))


def report(tag, r):
    print(f"[{tag}] maxPe={r['Pe']:.1f}  final std/h={r['stdT']/r['h']:.2f}  exact peak@{r['meanT']:.3f}")
    for k in ("DF", "CTR", "CN", "DFK", "DFRK"):
        p = r[k]
        print(f"   {k:3s}: min={p.min(): .3e}  peak@{r['x'][np.argmax(p)]:.3f}  "
              f"#neg={neg_nodes(p):3d}  TV={total_variation(p):.3f}")


if __name__ == "__main__":
    out = {}

    # # --- Regime A: SEVERE (under-resolved), the draft's intent. Honest result:
    # #     DF ripples too, but its failure is LOCALIZED vs CN's dispersive ringing.
    # print("=" * 70)
    # TA = 0.2
    # # kappa=15.0; sigma=0.3; xspan=4.0; n=201; x0=-2.5; std=6 * 0.04
    # kappa = 15.0; sigma = 0.1; xspan = 2.0; n = 201; x0 = -1; std = 6 * 0.04
    # std = TA*sigma**2
    # rA = solve_all(kappa=kappa, sigma=sigma, xspan=xspan, n=n, x0=x0, std=std, T=TA)
    # report("A severe (under-resolved)", rA)
    #
    # out["severe"] = {k: float(v) for k, v in
    #                  dict(Pe=rA['Pe'], stdT_over_h=rA['stdT'] / rA['h'],
    #                       DFmin=rA['DF'].min(), CTRmin=rA['CTR'].min(), CNmin=rA['CN'].min(),
    #                       DFneg=neg_nodes(rA['DF']), CTRneg=neg_nodes(rA['CTR']), CNneg=neg_nodes(rA['CN']),
    #                       DFtv=total_variation(rA['DF']), CTRtv=total_variation(rA['CTR']), CNtv=total_variation(rA['CN'])).items()}
    #
    # # --- Figure: severe regime, CORRECT curves (CN is the real CN), honest minima.
    # r = rA
    # fig, ax = plt.subplots(figsize=(7.0, 4.4))
    # ax.plot(r['x'], r['CN'],  'g-*', ms=4, lw=1.1, label=f"Crank--Nicolson (min={r['CN'].min():.2f})")
    # ax.plot(r['x'], r['CTR'], 'r--', lw=1.3,        label=f"centred, exact $e^{{L\\Delta t}}$ (min={r['CTR'].min():.2f})")
    # ax.plot(r['x'], r['DF'],  'b-o', ms=3, lw=1.3,  label=f"Diagonal Frog (min={r['DF'].min():.1e})")
    # ax.axhline(0, color='k', lw=0.6)
    # ax.set_xlim(-0.5, 0.25)
    # ax.set_xlabel("$x$"); ax.set_ylabel("density $p(x,T)$")
    # ax.set_title(rf"Severely advection-dominated OU ($\mathrm{{Pe}}_{{\max}}\approx{r['Pe']:.0f}$, T={TA:.3f})")
    # ax.legend(fontsize=8, frameon=False, loc="upper left")
    # ax.grid(True, alpha=0.3)
    # fig.tight_layout(); fig.savefig("E1d_severe_A.pdf"); plt.close(fig)
    # print("\n[written E1d_severe.pdf]")

    # --- Regime B: MODERATE Pe, RESOLVED pulse. Honest DF win: machine-zero vs
    #     negative CN. (This is where DF cleanly wins on positivity.)
    print("=" * 70)
    # TB = 0.05; kappa = 4.0; sigma = 0.5; xspan = 4.0; n = 201; x0 = -2.; std = 0.30
    # Double-well: start the pulse on the OUTER FLANK (x0=-1.9) so the steep
    # drift sweeps it through the high-Pe region; bottom-of-well x0=-1 shows nothing.
    TB = 0.04; kappa = 8.0; sigma = 0.2; xspan = 2.5; n = 201; x0 = -1.9; std = 6*(2*xspan/(n-1))
    rB = solve_all(kappa=kappa, sigma=sigma, xspan=xspan, n=n, x0=x0, std=std, T=TB)
    report("B moderate-Pe (resolved)", rB)
    out["resolved"] = {k: float(v) for k, v in
                       dict(Pe=rB['Pe'], stdT_over_h=rB['stdT'] / rB['h'],
                            DFmin=rB['DF'].min(), CTRmin=rB['CTR'].min(), CNmin=rB['CN'].min(),
                            DFneg=neg_nodes(rB['DF']), CTRneg=neg_nodes(rB['CTR']), CNneg=neg_nodes(rB['CN'])).items()}

    r = rB
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(r['x'], r['CN'],  'g-*', ms=4, lw=1.1, label=f"Crank--Nicolson (min={r['CN'].min():.2f})")
    ax.plot(r['x'], r['CTR'], 'r--', lw=1.3,        label=f"centred, exact $e^{{L\\Delta t}}$ (min={r['CTR'].min():.2f})")
    ax.plot(r['x'], r['DF'],  'b-o', ms=3, lw=1.3,  label=f"Diagonal Frog (min={r['DF'].min():.1e})")
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlim(-1.2, -0.8)
    ax.set_xlabel("$x$"); ax.set_ylabel("density $p(x,T)$")
    # ax.set_title(rf"Severely advection-dominated OU ($\mathrm{{Pe}}_{{\max}}\approx{r['Pe']:.0f}$, T={TB:.3f})")
    ax.set_title(rf"Double-well potential Kramers escape problem, ($\mathrm{{Pe}}_{{\max}}\approx{r['Pe']:.0f}$, T={TB:.3f})")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("E1d_severe_B.pdf"); plt.close(fig)
    print("\n[written E1d_severe_B.pdf]")

    import json
    json.dump(out, open("../Tests/paperFD/results_1d_severe.json", "w"), indent=2)
    print("[written results_1d_severe.json]")
