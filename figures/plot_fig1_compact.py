#!/usr/bin/env python3
"""
plot_fig1_compact.py

Compact single-column Fig. 1 (3.375 in wide, PRL/revtex column width).
Same content as the Fig. 1 collage in the Oct-2026 draft, re-laid out:

    (a)  reduced VDOS g(w)/w: total, transverse and longitudinal in ONE panel
         (replaces the three stacked panels a-c of the draft)
    (b) (c) (d)   transverse   : w~0.4 | w~1.2 | zoom on the -1 TD of (b)
    (e) (f) (g)   longitudinal : w~0.4 | w~1.2 | <psi_4>(w), T and L
         (<psi_4> replaces the three insets of the draft panels a-c)

Needs fig1_compact_cache.npz -> run fig1_compact_data.py first.
Output: ../fig1.pdf (vector, quiver fields rasterised) and ../fig1.png (preview)
"""

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/private", "tmp", "matplotlib-cache"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "fig1_compact_cache.npz")
OUT_PDF = os.path.join(HERE, "..", "fig1.pdf")
OUT_PNG = os.path.join(HERE, "..", "fig1.png")

# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------
SHOW_OMEGA_IR = True  # mark the transverse Ioffe-Regel frequency in (a)
SHOW_MODE_CIRCLES = False  # circles in (a) at the frequencies of the mode panels
OMEGA_IR = 2.15  # value quoted in the main text

# ---------------------------------------------------------------------------
# Style: sizes are final print sizes (pt / in)
# ---------------------------------------------------------------------------
FS = 7.0  # tick labels, legend, annotations
FS_LAB = 8.0  # axis labels, panel letters, row/column titles
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": FS,
        "axes.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "axes.labelsize": FS_LAB,
        "axes.labelpad": 1.5,
        "legend.fontsize": FS - 0.5,
    }
)

C_TOT, C_T, C_L = "#CC79A7", "#E69F00", "#56B4E9"  # Okabe-Ito, as before
C_POS, C_NEG = "red", "blue"

# physical constants of the sample
system_size = 192.897128868165
area_density = 0.86
a0 = 1.0 / np.sqrt(area_density)
quiver_scale = 5.0e-3

# ---------------------------------------------------------------------------
# Layout (inches)
# ---------------------------------------------------------------------------
W = 3.375
x_row_lab = 0.17  # room for the rotated "Transverse"/"Longitudinal" labels
r_margin = 0.03
gap = 0.045
s = (W - x_row_lab - r_margin - 2 * gap) / 3  # square panel side
b_margin = 0.03
h_titles = 0.15  # column titles above the transverse row
h_xlab = 0.28  # x tick labels + x label of (a)
h_dos = 1.02
t_margin = 0.05
H = b_margin + 2 * s + gap + h_titles + h_xlab + h_dos + t_margin

y_row3 = b_margin
y_row2 = y_row3 + s + gap
y_dos = y_row2 + s + h_titles + h_xlab
x_cols = [x_row_lab + k * (s + gap) for k in range(3)]
dos_left = 0.37


def ax_in(fig, x, y, w, h):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------
def panel_letter(ax, letter, x=0.035, y=0.965, ha="left", boxed=True, extra=None):
    kw = dict(transform=ax.transAxes, fontsize=FS_LAB, ha=ha, va="top", zorder=30)
    if boxed:
        kw["bbox"] = dict(facecolor="white", alpha=0.85, edgecolor="none",
                          boxstyle="square,pad=0.08")
    s = rf"\mathrm{{({letter})}}"
    if extra:  # e.g. "(c)  w ~ 0.4" in one box
        s += r"\ \ " + extra
    ax.text(x, y, f"${s}$", **kw)


def scale_bar(ax, length, label, corner="right"):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr, yr = x1 - x0, y1 - y0
    xe = x1 - 0.07 * xr if corner == "right" else x0 + 0.07 * xr + length
    yb = y0 + 0.06 * yr
    ax.plot([xe - length, xe], [yb, yb], color="k", lw=1.6, solid_capstyle="butt", zorder=20)
    ax.text(xe - length / 2, yb + 0.025 * yr, label, ha="center", va="bottom", fontsize=FS - 0.5,
            zorder=20, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                                 boxstyle="square,pad=0.05"))


def kspace_inset(ax, q, O, size=0.38):
    ins = ax.inset_axes([1 - size, 1 - size, size, size])
    ins.set_zorder(25)  # keep defect markers of the parent panel underneath
    ins.imshow(O, origin="lower", extent=[q.min(), q.max(), q.min(), q.max()],
               cmap="magma", interpolation="nearest")
    lk = (2 * np.pi / a0) / 30
    xl, yl = ins.get_xlim(), ins.get_ylim()
    xe = xl[1] - 0.06 * (xl[1] - xl[0])
    yb = yl[0] + 0.08 * (yl[1] - yl[0])
    ins.plot([xe - lk, xe], [yb, yb], color="white", lw=0.9, solid_capstyle="butt")
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_visible(False)
    return ins


def defects_scatter(ax, pos, neg, size):
    kw = dict(marker="o", edgecolor="k", linewidths=0.3, s=size, zorder=8)
    if len(pos):
        ax.scatter(pos[:, 0], pos[:, 1], facecolors=C_POS, **kw)
    if len(neg):
        ax.scatter(neg[:, 0], neg[:, 1], facecolors=C_NEG, **kw)


def closest(target, pts):
    return pts[np.argmin(((pts - np.asarray(target)) ** 2).sum(1))]


def lattice(ax, pts, approx, links):
    xy = np.array([closest(c, pts) for c in approx])
    for i, j in links:
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], ls=(0, (2.5, 1.5)), lw=0.45,
                color="dimgray", alpha=0.75, zorder=4)


def quiver(ax, X, U, mask=None, **kw):
    if mask is not None:
        X, U = X[mask], U[mask]
    ax.quiver(X[:, 0], X[:, 1], U[:, 0], U[:, 1], angles="xy", scale_units="xy",
              scale=quiver_scale, rasterized=True, zorder=2, **kw)


def square_frame(ax, lim):
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# Hand-identified lattices (same approximate coordinates as the PRL script)
LAT_T04 = (
    [[81.5, 7.5], [129.0, 7.5], [1.0, 22.5], [54.5, 30], [106.5, 32.5], [165.0, 28.0],
     [27.0, 52.5], [84.0, 59.0], [134.0, 54.0], [180.0, 68.0], [48.5, 78.5], [113.5, 83.0],
     [163.0, 83.0], [10, 87], [25, 108], [73.82, 101.84], [139, 110], [10, 131], [48, 125],
     [108, 135], [155, 135], [187, 117], [182, 145], [22, 144], [69, 146], [4, 171],
     [54, 172], [110, 165], [34, 192], [170, 183]],
    [(0, 3), (0, 4), (1, 4), (1, 5), (3, 6), (3, 7), (4, 7), (4, 8), (5, 8), (5, 9),
     (6, 13), (6, 10), (7, 10), (7, 11), (8, 11), (8, 12), (13, 14), (10, 14), (10, 15),
     (11, 15), (11, 16), (12, 16), (14, 17), (14, 18), (15, 18), (15, 19), (16, 19),
     (16, 20), (20, 21), (17, 23), (18, 23), (18, 24), (19, 24), (23, 25), (23, 26),
     (26, 28), (9, 12), (24, 26), (26, 27), (19, 27), (20, 27), (20, 22), (22, 29),
     (12, 21), (25, 28)],
)
LAT_L04 = ([[17, 114], [117.0, 114], [19, 14], [117, 16]], [(0, 1), (1, 3), (3, 2), (2, 0)])
LAT_L12 = (
    [[51, 61], [84, 55], [113, 60], [58, 93], [86, 92], [118, 92], [57, 122], [88, 121],
     [123, 123]],
    [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5), (3, 6), (4, 7), (5, 8),
     (6, 7), (7, 8)],
)

# Zoom (d): the strongly localised -1 transverse TD at the left edge of (b)
ZOOM_CENTER_APPROX = (9.0, 131.0)
ZOOM_HALF = 24.0


def main():
    c = np.load(CACHE)
    X = c["coords"]
    bc = c["bin_centers"]
    omega_bp = float(c["omega_bp"])
    w_lo, w_mid = [float(v) for v in c["mode_omegas"]]

    fig = plt.figure(figsize=(W, H))

    # ======================= (a) reduced VDOS ==============================
    ax = ax_in(fig, dos_left, y_dos, W - dos_left - r_margin, h_dos)
    for key, col, z in (("total", C_TOT, 3), ("trans", C_T, 4), ("long", C_L, 5)):
        ax.plot(bc, c[f"rdos_{key}"], color=col, lw=0.9, zorder=z)
        ax.axhline(float(c[f"debye_{key}"]), color=col, ls=(0, (4, 2)), lw=0.7, alpha=0.9,
                   zorder=1)
    # circles: modes shown in (b),(c) [transverse] and (e),(f) [longitudinal]
    for w in (w_lo, w_mid) if SHOW_MODE_CIRCLES else ():
        for key in ("trans", "long"):
            y = np.interp(w, bc, c[f"rdos_{key}"])
            ax.scatter([w], [y], s=15, facecolors="none", edgecolors="k", linewidths=0.6,
                       alpha=0.85, zorder=10)
    ax.annotate(r"$\omega_\mathrm{BP}$", xy=(omega_bp, 7.0e-3), xytext=(omega_bp, 10.2e-3),
                ha="center", va="bottom", fontsize=FS,
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="k", shrinkA=0.5,
                                shrinkB=0, mutation_scale=5))
    if SHOW_OMEGA_IR:
        y_ir = np.interp(OMEGA_IR, bc, c["rdos_total"])
        ax.annotate(r"$\omega_\mathrm{IR}$", xy=(OMEGA_IR, y_ir + 0.9e-3),
                    xytext=(OMEGA_IR, y_ir + 4.2e-3), ha="center", va="bottom",
                    fontsize=FS,
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="k",
                                    shrinkA=0.5, shrinkB=0, mutation_scale=5))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 0.016)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.004))
    ax.yaxis.set_minor_locator(MultipleLocator(0.002))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"${int(round(v * 1000))}$"))
    ax.tick_params(top=True, right=True, which="both")
    ax.set_xlabel(r"$\omega$", labelpad=1.0)
    ax.set_ylabel(r"$g(\omega)/\omega\ (\times 10^{-3})$")
    handles = [Line2D([], [], color=C_TOT, lw=0.9, label=r"$\mathrm{Total}$"),
               Line2D([], [], color=C_T, lw=0.9, label=r"$\mathrm{Transverse}$"),
               Line2D([], [], color=C_L, lw=0.9, label=r"$\mathrm{Longitudinal}$"),
               Line2D([], [], color="0.35", lw=0.7, ls=(0, (4, 2)),
                      label=r"$\mathrm{Debye\ levels}$")]
    ax.legend(handles=handles, loc="upper right", frameon=False, handlelength=1.8,
              handletextpad=0.5, labelspacing=0.25, borderaxespad=0.35)
    panel_letter(ax, "a", x=0.012, y=0.965, boxed=False)

    # ======================= mode panels ===================================
    full = ((0, system_size), (0, system_size))
    half = system_size / 4
    ctr = system_size / 2
    mid = ((ctr - half, ctr + half), (ctr - half, ctr + half))

    def inside(P, lim):
        return ((P[:, 0] >= lim[0][0]) & (P[:, 0] <= lim[0][1]) &
                (P[:, 1] >= lim[1][0]) & (P[:, 1] <= lim[1][1]))

    specs = [  # (key, row-y, col, limits, letter, lattice)
        ("t22", y_row2, 0, full, "b", LAT_T04),
        ("t271", y_row2, 1, mid, "c", None),
        ("l22", y_row3, 0, full, "e", LAT_L04),
        ("l271", y_row3, 1, mid, "f", LAT_L12),
    ]
    mode_axes = {}
    for key, y, col, lim, letter, lat in specs:
        ax = ax_in(fig, x_cols[col], y, s, s)
        U = c[f"disp_{key}"].astype(float)
        pos, neg = c[f"pos_{key}"], c[f"neg_{key}"]
        m = None if lim is full else inside(X, lim)
        quiver(ax, X, U, m)
        if lim is not full:
            pos, neg = pos[inside(pos, lim)], neg[inside(neg, lim)]
        defects_scatter(ax, pos, neg, size=5.5 if lim is full else 7)
        if lat is not None:
            allp = np.vstack([c[f"pos_{key}"], c[f"neg_{key}"]])
            lattice(ax, allp, *lat)
        square_frame(ax, lim)
        kspace_inset(ax, c[f"q_{key}"], c[f"Ok_{key}"])
        if key.startswith("l"):
            scale_bar(ax, 10 * a0, r"$10\,a_0$")
        panel_letter(ax, letter)
        mode_axes[key] = ax

    # zoom box in (b)
    neg_t = c["neg_t22"]
    zc = closest(ZOOM_CENTER_APPROX, neg_t)
    axb = mode_axes["t22"]
    axb.add_patch(plt.Rectangle((zc[0] - ZOOM_HALF, zc[1] - ZOOM_HALF), 2 * ZOOM_HALF,
                                2 * ZOOM_HALF, fill=False, ec="k", lw=0.6, ls=(0, (2, 1.2)),
                                zorder=12, clip_on=True))

    # ======================= (d) zoom on the -1 TD ==========================
    ax = ax_in(fig, x_cols[2], y_row2, s, s)
    U = c["disp_t22"].astype(float)

    def wrap(P):
        return (P - zc + system_size / 2) % system_size - system_size / 2

    dX = wrap(X)
    m = (np.abs(dX) <= ZOOM_HALF * 1.02).all(1)
    quiver(ax, dX, U, m, width=0.0026)  # bolder arrows so the saddle/vortex read clearly
    for P, colr in ((c["pos_t22"], C_POS), (c["neg_t22"], C_NEG)):
        dP = wrap(P)
        dP = dP[(np.abs(dP) <= ZOOM_HALF).all(1)]
        ax.scatter(dP[:, 0], dP[:, 1], marker="o", facecolors=colr, edgecolor="k",
                   linewidths=0.4, s=16, zorder=8)
    square_frame(ax, ((-ZOOM_HALF, ZOOM_HALF), (-ZOOM_HALF, ZOOM_HALF)))
    for sp in ax.spines.values():
        sp.set_linestyle((0, (2, 1.2)))
    scale_bar(ax, 10 * a0, r"$10\,a_0$", corner="left")
    panel_letter(ax, "d")

    # ======================= (g) <psi_4>(omega) =============================
    lpad, bpad = 0.255, 0.235
    ax = ax_in(fig, x_cols[2] + lpad, y_row3 + bpad, s - lpad - 0.02, s - bpad - 0.03)
    ax.axvline(omega_bp, color="0.55", ls=(0, (3, 1.5)), lw=0.6, zorder=0)
    ax.text(omega_bp + 0.12, 0.97, r"$\omega_\mathrm{BP}$", ha="left", va="top",
            fontsize=FS - 0.5)
    for key, col in (("T", C_T), ("L", C_L)):
        ax.plot(c[f"psi4_w_{key}"], c[f"psi4_{key}"], "-o", color=col, lw=0.9, ms=2.2,
                mec=col, zorder=3, label=rf"$\mathrm{{{key}}}$")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.52), frameon=False,
              handlelength=1.2, handletextpad=0.4, labelspacing=0.2, borderaxespad=0.2,
              fontsize=FS)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"${v:g}$"))
    ax.tick_params(top=True, right=True, which="both")
    ax.set_xlabel(r"$\omega$", labelpad=0.5)
    ax.set_ylabel(r"$\langle\psi_4\rangle$", labelpad=1.0)
    ax.text(0.95, 0.95, r"$\mathrm{(g)}$", transform=ax.transAxes, fontsize=FS_LAB,
            ha="right", va="top")

    # ======================= titles & row labels ============================
    titles = [r"$\omega \approx 0.4$", r"$\omega \approx 1.2$",
              r"$\omega \approx 0.4\ \mathrm{(zoom)}$"]  # values quoted in the text
    for k, t in enumerate(titles):
        fig.text((x_cols[k] + s / 2) / W, (y_row2 + s + 0.025) / H, t, ha="center",
                 va="bottom", fontsize=FS_LAB)
    for y, t in ((y_row2, r"$\mathrm{Transverse}$"), (y_row3, r"$\mathrm{Longitudinal}$")):
        fig.text(0.075 / W, (y + s / 2) / H, t, rotation=90, ha="center", va="center",
                 fontsize=FS_LAB)

    fig.savefig(OUT_PDF, dpi=600)
    fig.savefig(OUT_PNG, dpi=300)
    print(f"figure size: {W:.3f} in x {H:.3f} in ({W * 72:.0f} pt x {H * 72:.0f} pt)")
    print(f"wrote {os.path.normpath(OUT_PDF)}")


def main_2col():
    """
    2 columns x 3 rows variant (same column width):
        (a) merged VDOS + one <psi_4> inset (total, T, L) | (b) zoom on the -1 TD of (c)
        (c) T, w~0.4 | (d) T, w~1.2
        (e) L, w~0.4 | (f) L, w~1.2
    Output: ../fig1_2col.pdf / ../fig1_2col.png
    """
    c = np.load(CACHE)
    X = c["coords"]
    bc = c["bin_centers"]
    omega_bp = float(c["omega_bp"])
    w_lo, w_mid = [float(v) for v in c["mode_omegas"]]

    # left strip holds the y label of (a) and the row labels, all centred at x_lab
    xl, rm, g = 0.36, 0.03, 0.05
    x_lab = 0.155
    s2 = (W - xl - rm - g) / 2  # every panel, including (a), is an s2 x s2 square
    bm, h_tit, g12, tm = 0.03, 0.0, 0.25, 0.06  # g12: x ticks + x label of (a)
    y3 = bm
    y2 = y3 + s2 + g
    y1 = y2 + s2 + h_tit + g12
    H2 = y1 + s2 + tm
    xc = [xl, xl + s2 + g]
    fig = plt.figure(figsize=(W, H2))

    # ---------------- (a) merged reduced VDOS + psi4 inset ----------------
    ax = ax_in(fig, xc[0], y1, s2, s2)
    for key, col, z in (("total", C_TOT, 3), ("trans", C_T, 4), ("long", C_L, 5)):
        ax.plot(bc, c[f"rdos_{key}"], color=col, lw=0.9, zorder=z)
        ax.axhline(float(c[f"debye_{key}"]), color=col, ls=(0, (4, 2)), lw=0.7, alpha=0.9,
                   zorder=1)
    for w in (w_lo, w_mid) if SHOW_MODE_CIRCLES else ():
        for key in ("trans", "long"):
            y = np.interp(w, bc, c[f"rdos_{key}"])
            ax.scatter([w], [y], s=13, facecolors="none", edgecolors="k", linewidths=0.6,
                       alpha=0.85, zorder=10)
    # frequency markers: arrows down onto the total curve, labels at a common height
    arrows = [(r"$\omega_\mathrm{BP}$", omega_bp)]
    if SHOW_OMEGA_IR:
        arrows.append((r"$\omega_\mathrm{IR}$", OMEGA_IR))
    for lab, w in arrows:
        y_tip = c["rdos_total"][np.abs(bc - w) < 0.08].max() + 0.55e-3
        ax.annotate(lab, xy=(w, y_tip), xytext=(w, 9.4e-3), ha="center", va="bottom",
                    fontsize=FS,
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="k", shrinkA=0.5,
                                    shrinkB=0, mutation_scale=5))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 0.018)  # head-room for the inset and the two arrows; ticks stay 0-16
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.004))
    ax.yaxis.set_minor_locator(MultipleLocator(0.002))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"${int(round(v * 1000))}$"))
    ax.tick_params(top=True, right=True, which="both")
    ax.set_xlabel(r"$\omega$", labelpad=1.0)
    ax.set_ylabel(r"$g(\omega)/\omega\ (\times 10^{-3})$")
    ax.yaxis.set_label_coords(-(xl - x_lab) / s2, 0.5)
    # One legend in the free top-left corner; black text, colour only on the line samples.
    def line_legend(entries, anchor, loc, transform):
        h = [Line2D([], [], color=col, lw=0.9, label=rf"$\mathrm{{{lab}}}$")
             for lab, col in entries]
        return ax.legend(handles=h, loc=loc, bbox_to_anchor=anchor,
                         bbox_transform=transform, frameon=False, fontsize=FS,
                         handlelength=1.2, handletextpad=0.4, labelspacing=0.15,
                         borderpad=0.0, borderaxespad=0.0)

    line_legend([("Total", C_TOT), ("Trans.", C_T), ("Long.", C_L)], (0.165, 0.955),
                "upper left", ax.transAxes)
    panel_letter(ax, "a", boxed=False)

    ins = ax.inset_axes([0.63, 0.60, 0.345, 0.375])
    ins.axvline(omega_bp, color="0.55", ls=(0, (3, 1.5)), lw=0.5, zorder=0)
    for key, col in (("tot", C_TOT), ("T", C_T), ("L", C_L)):
        ins.plot(c[f"psi4_w_{key}"], c[f"psi4_{key}"], "-o", color=col, lw=0.7, ms=1.5,
                 mec=col, zorder={"tot": 3, "T": 4, "L": 5}[key])
    ins.text(omega_bp + 0.15, 0.97, r"$\omega_\mathrm{BP}$", ha="left", va="top",
             fontsize=FS - 1.5)
    ins.set_xlim(0, 4)
    ins.set_ylim(0, 1.0)
    ins.xaxis.set_major_locator(MultipleLocator(2))
    ins.xaxis.set_minor_locator(MultipleLocator(1))
    ins.yaxis.set_major_locator(MultipleLocator(1.0))  # labels 0 and 1 only: keeps the
    ins.yaxis.set_minor_locator(MultipleLocator(0.5))  # inset clear of the legend
    ins.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"${v:g}$"))
    ins.tick_params(which="both", labelsize=FS - 1.5, pad=1.0, length=1.6)
    ins.tick_params(which="minor", length=1.0)
    ins.set_xlabel(r"$\omega$", fontsize=FS - 1, labelpad=0.0)
    ins.set_ylabel(r"$\langle\psi_4\rangle$", fontsize=FS - 1, labelpad=1.2)

    # ---------------- (c)-(f) mode panels ---------------------------------
    full = ((0, system_size), (0, system_size))
    half, ctr = system_size / 4, system_size / 2
    mid = ((ctr - half, ctr + half), (ctr - half, ctr + half))

    def inside(P, lim):
        return ((P[:, 0] >= lim[0][0]) & (P[:, 0] <= lim[0][1]) &
                (P[:, 1] >= lim[1][0]) & (P[:, 1] <= lim[1][1]))

    specs = [("t22", y2, 0, full, "c", LAT_T04), ("t271", y2, 1, mid, "d", None),
             ("l22", y3, 0, full, "e", LAT_L04), ("l271", y3, 1, mid, "f", LAT_L12)]
    w_text = {"22": r"\omega \approx 0.4", "271": r"\omega \approx 1.2"}  # as in the text
    mode_axes = {}
    for key, y, col, lim, letter, lat in specs:
        ax = ax_in(fig, xc[col], y, s2, s2)
        U = c[f"disp_{key}"].astype(float)
        pos, neg = c[f"pos_{key}"], c[f"neg_{key}"]
        m = None if lim is full else inside(X, lim)
        quiver(ax, X, U, m)
        if lim is not full:
            pos, neg = pos[inside(pos, lim)], neg[inside(neg, lim)]
        defects_scatter(ax, pos, neg, size=9 if lim is full else 11)
        if lat is not None:
            lattice(ax, np.vstack([c[f"pos_{key}"], c[f"neg_{key}"]]), *lat)
        square_frame(ax, lim)
        kspace_inset(ax, c[f"q_{key}"], c[f"Ok_{key}"], size=0.36)
        if key.startswith("l"):
            scale_bar(ax, 10 * a0, r"$10\,a_0$")
        panel_letter(ax, letter, extra=w_text[key[1:]])
        mode_axes[key] = ax

    zc = closest(ZOOM_CENTER_APPROX, c["neg_t22"])
    mode_axes["t22"].add_patch(plt.Rectangle(
        (zc[0] - ZOOM_HALF, zc[1] - ZOOM_HALF), 2 * ZOOM_HALF, 2 * ZOOM_HALF, fill=False,
        ec="k", lw=0.6, ls=(0, (2, 1.2)), zorder=12, clip_on=True))

    # ---------------- (b) zoom on the -1 TD -------------------------------
    ax = ax_in(fig, xc[1], y1, s2, s2)
    U = c["disp_t22"].astype(float)

    def wrap(P):
        return (P - zc + system_size / 2) % system_size - system_size / 2

    dX = wrap(X)
    quiver(ax, dX, U, (np.abs(dX) <= ZOOM_HALF * 1.02).all(1), width=0.0026)
    for P, colr in ((c["pos_t22"], C_POS), (c["neg_t22"], C_NEG)):
        dP = wrap(P)
        dP = dP[(np.abs(dP) <= ZOOM_HALF).all(1)]
        ax.scatter(dP[:, 0], dP[:, 1], marker="o", facecolors=colr, edgecolor="k",
                   linewidths=0.4, s=22, zorder=8)
    square_frame(ax, ((-ZOOM_HALF, ZOOM_HALF), (-ZOOM_HALF, ZOOM_HALF)))
    for sp in ax.spines.values():
        sp.set_linestyle((0, (2, 1.2)))
    scale_bar(ax, 10 * a0, r"$10\,a_0$", corner="left")
    panel_letter(ax, "b", extra=r"\omega \approx 0.4")

    # ---------------- titles & row labels ---------------------------------
    for y, t in ((y2, r"$\mathrm{Transverse}$"), (y3, r"$\mathrm{Longitudinal}$")):
        fig.text(x_lab / W, (y + s2 / 2) / H2, t, rotation=90, ha="center", va="center",
                 fontsize=FS_LAB)

    out_pdf = os.path.join(HERE, "..", "fig1_2col.pdf")
    fig.savefig(out_pdf, dpi=600)
    fig.savefig(os.path.join(HERE, "..", "fig1_2col.png"), dpi=300)
    print(f"figure size: {W:.3f} in x {H2:.3f} in ({W * 72:.0f} pt x {H2 * 72:.0f} pt)")
    print(f"wrote {os.path.normpath(out_pdf)}")


if __name__ == "__main__":
    import sys

    variant = sys.argv[1] if len(sys.argv) > 1 else "3col"
    if variant == "3col":
        main()
    elif variant == "2col":
        main_2col()
    else:
        raise SystemExit("usage: plot_fig1_compact.py [3col|2col]")
