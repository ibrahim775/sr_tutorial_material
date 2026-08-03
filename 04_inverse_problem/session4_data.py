from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.signal import savgol_filter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
FILL = "#dbe9fb"


def use_house_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.titleweight": "medium", "axes.titlelocation": "left",
        "axes.titlepad": 9,
        "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.grid": True, "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "figure.constrained_layout.use": True,
    })


use_house_style()


def _finish(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


X_SYM = sympy.Symbol("x", real=True)

TRUE_PSI_EXPR = sympy.exp(-X_SYM ** 4 / 4)


def true_psi(x):
    return np.exp(-np.asarray(x, dtype=float) ** 4 / 4)


def true_potential(x):
    """V = (x^6 - 3x^2)/2, taking E = 0. Minima at +-1, barrier at the origin."""
    x = np.asarray(x, dtype=float)
    return (x ** 6 - 3 * x ** 2) / 2


def image_density(noise_pct: float = 0.0, lo: float = -2.6, hi: float = 2.6,
                  n: int = 500, seed: int = 0):

    x = np.linspace(lo, hi, n)
    dens = true_psi(x) ** 2
    if noise_pct > 0:
        rng = np.random.default_rng(seed)
        dens = dens + rng.normal(0, noise_pct / 100 * dens.max(), n)
    return x, dens


def local_curvature(x, density, window: int = 61, poly: int = 3,
                    floor: float = 1e-6):
    """Turn an imaged density into the target psi''/(2 psi), and its weight.
    """
    x = np.asarray(x, dtype=float)
    dx = float(np.median(np.diff(x)))
    psi = np.sqrt(np.clip(np.asarray(density, dtype=float), 1e-14, None))
    d2 = savgol_filter(psi, window, poly, deriv=2, delta=dx)
    with np.errstate(all="ignore"):
        target = d2 / (2 * psi)
    weight = psi ** 2
    ok = np.isfinite(target) & (weight > floor * weight.max())
    return x[ok], target[ok], weight[ok]


# --------------------------------------------------------------- design ---

def potential_for(psi_expr):
    """The potential that has a given psi as an eigenstate, up to the zero of E.

    """
    psi_expr = sympy.sympify(psi_expr)
    V = sympy.simplify(sympy.diff(psi_expr, X_SYM, 2) / (2 * psi_expr))
    return sympy.simplify(sympy.expand(V))


def as_function(expr):
    f = sympy.lambdify(X_SYM, sympy.sympify(expr), "numpy")

    def evaluate(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            return np.asarray(f(x), dtype=float) * np.ones_like(x)

    return evaluate


def read_equation(equation: str):
    return as_function(sympy.sympify(str(equation)))


# -------------------------------------------------------------- figures ---

def plot_image(x, density, noise_pct=None):
    """What the experiment hands us."""
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.fill_between(x, 0, density, color=FILL, zorder=2)
    ax.plot(x, density, color=BLUE, lw=1.6, zorder=3)
    ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
    title = "Imaged density"
    if noise_pct:
        title += f",  noise {noise_pct:g}% of peak"
    _finish(ax, title, "x", r"$|\psi|^2$  [arb.]")
    plt.show()


def plot_raw_target(x, target, weight, truth=None, clip=8.0):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    for ax, wide in [(axes[0], True), (axes[1], False)]:
        ax.plot(x, target, ".", ms=3.0, color=BLUE, alpha=0.55, zorder=3,
                label="from the data")
        if truth is not None:
            ax.plot(x, truth(x), color=ORANGE, lw=2.0, zorder=4, label="truth")
        ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
        if wide:
            ax.set_title("as computed", color=INK)
        else:
            ax.set_ylim(-clip, clip)
            ax.set_title(f"same points, zoomed to $\\pm${clip:g}", color=INK)
        _finish(ax, None, "x", r"$\psi''/2\psi$" if ax is axes[0] else None)
    axes[1].legend(loc="upper center")
    plt.show()


def plot_recovered(x, V_found, truth=None, density=None, label="recovered",
                   title="The trapping potential"):
    x = np.asarray(x, dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))

    d = None if density is None else np.asarray(density, dtype=float) / np.max(density)
    core = np.ones_like(x, dtype=bool) if d is None else d > 0.01
    vals = [V_found(x[core])] + ([truth(x[core])] if truth is not None else [])
    finite = np.concatenate([v[np.isfinite(v)] for v in vals])
    lo, hi = float(finite.min()), float(finite.max())
    pad = max(0.35 * (hi - lo), 0.5)
    bottom, top = lo - 1.5 * pad, hi + pad
    ax.set_ylim(bottom, top)

    if d is not None:
        band = bottom + 0.20 * (top - bottom) * d
        ax.fill_between(x, bottom, band, color=FILL, zorder=1,
                        label="where the atoms are")

    if truth is not None:
        ax.plot(x, truth(x), color=AXIS, lw=5.0, solid_capstyle="round",
                zorder=2, label="true trap")
    ax.plot(x, V_found(x), color=BLUE, lw=2.2, zorder=3, label=label)
    ax.legend(loc="upper center", ncol=3)
    _finish(ax, title, "x", "V(x)  [arb.]")
    plt.show()


def plot_design(psi_expr, lo=-3.0, hi=3.0, n=500):
    """A state you invented, and the universe that would hold it."""
    V = potential_for(psi_expr)
    psi = as_function(psi_expr)
    Vf = as_function(V)
    x = np.linspace(lo, hi, n)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    p = psi(x)
    p = p / np.abs(p).max()
    axes[0].fill_between(x, 0, p ** 2, color=FILL, zorder=2, label=r"$|\psi|^2$")
    axes[0].plot(x, p, color=BLUE, lw=2.0, zorder=3, label=r"$\psi$")
    axes[0].axhline(0, color=AXIS, lw=0.9, zorder=1)
    axes[0].legend(loc="upper right")
    _finish(axes[0], f"the state you asked for:   $\\psi = {sympy.latex(sympy.sympify(psi_expr))}$",
            "x", r"$\psi$,  $|\psi|^2$")

    v = Vf(x)
    axes[1].plot(x, v, color=ORANGE, lw=2.2, zorder=3)
    fin = v[np.isfinite(v)]
    if len(fin):
        axes[1].set_ylim(np.percentile(fin, 1) - 0.5, np.percentile(fin, 97) + 0.5)
    _finish(axes[1], f"the trap it demands:   $V = {sympy.latex(V)}$", "x", "V(x)")
    plt.show()
    return V
