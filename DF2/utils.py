
class Utils:

    @staticmethod
    def convert2latex(header, caption, levels, dt,
        oDF, oBD, sw1, eDF, eBD, gmin, sw):

        # --- Print LaTeX Table Header ---
        print(r"\begin{table}[htbp]")
        print(r"\centering")
        print(r"\begin{tabular}{cccccccc}")
        print(r"\hline")
        print(header)
        print(r"\hline")

        for i, (N, ns) in enumerate(levels):
            # --- Print LaTeX Table Row ---
            # Scientific notation converted from .23e-04 to .23 \times 10^{-4}
            eDF_ltx = ("%.2e" % eDF[i]).replace("e", r" \times 10^{") + "}"
            gmin_ltx = ("%+.2e" % gmin[i]).replace("e", r" \times 10^{") + "}"
            eBD_ltx = ("%.2e" % eBD[i]).replace("e", r" \times 10^{") + "}"

            print(
                r"%4d & %.4f & $%s$ & %s & $%s$ & $%s$ & %s & %4d \\"
                % (N, dt[i], eDF_ltx, oDF[i], gmin_ltx, eBD_ltx, oBD[i], sw[i])
            )

        # --- Print LaTeX Table Footer ---
        print(r"\hline")
        print(r"\end{tabular}")
        print(rf"\caption{{{caption}}}")
        print(r"\label{tab:convergence_results}")
        print(r"\end{table}")


