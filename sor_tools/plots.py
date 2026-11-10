"""Figures shown in the experiment notebook."""

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2a78d6"
INK, INK_2, REFERENCE, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

PLOT_STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "lines.linewidth": 1.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
}


def reference_lines(ax, omegas):
    '''rho = 1 (divergence threshold) and Kahan's lower bound |omega - 1|.'''
    ax.axhline(1, color=REFERENCE, linestyle="--", linewidth=1)
    ax.plot(omegas, np.abs(omegas - 1), color=REFERENCE, linestyle=":", linewidth=1.2)
    ax.text(1.98, 1.03, "diverges above this line", color=INK_2, fontsize=8, va="bottom", ha="right")
    ax.text(0.5, 0.3, r"Kahan bound $|\omega - 1|$", color=INK_2, fontsize=8)


def mark(ax, x, y, text, dx, dy, filled=True, ha="left"):
    ax.plot(x, y, "o", markersize=7, color=BLUE if filled else SURFACE, markeredgecolor=BLUE,
            markeredgewidth=1.5, zorder=3)
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK, ha=ha)


def omega_plot(omegas, counts, rho_ex1, w_opt, gs_count, sor_count, gs_radius):
    """Plot the sweep counts and spectral radii for Example 1."""

    iters = np.array([np.nan if count is None else count for count in counts])

    with plt.rc_context(PLOT_STYLE):
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 5.6), sharex=True, gridspec_kw={"hspace": 0.12})

        top.plot(omegas, iters, color=BLUE)
        top.set_yscale("log")
        top.set_yticks([10, 20, 50, 100, 200, 500, 1000, 2000])
        top.set_yticklabels(["10", "20", "50", "100", "200", "500", "1000", "2000"])
        top.set_ylim(8, 3000)
        top.set_ylabel("Iterations to 7 decimal places")
        top.axvline(w_opt, color=REFERENCE, linestyle=":", linewidth=1)
        mark(top, 1.0, gs_count, f"Gauss-Seidel ($\\omega = 1$): {gs_count}", 0, 12, filled=False, ha="center")
        mark(top, 1.25, sor_count, f"SOR ($\\omega = 1.25$): {sor_count}", 10, -4)
        top.set_title(r"Example 1: iterations and spectral radius against $\omega$", loc="left")

        bottom.plot(omegas, rho_ex1, color=BLUE)
        reference_lines(bottom, omegas)
        bottom.axvline(w_opt, color=REFERENCE, linestyle=":", linewidth=1)
        mark(bottom, 1.0, gs_radius, r"$\rho(T_G) = 0.625$", -8, 6, filled=False, ha="right")
        mark(bottom, w_opt, w_opt - 1, rf"optimal $\omega \approx {w_opt:.3f}$, $\rho = {w_opt - 1:.3f}$", 8, -14)
        bottom.text(1.62, 0.42, r"$\rho = \omega - 1$ (Theorem 7.26)", color=INK_2, fontsize=8)
        bottom.set_ylim(0, 1.15)
        bottom.set_xlim(0, 2)
        bottom.set_ylabel(r"Spectral radius $\rho(T_\omega)$")
        bottom.set_xlabel(r"Relaxation parameter $\omega$")

    return fig
