"""Data, physics and plotting for the supplemental notebook.

A tracer diffuses into a long bar from one end. With a diffusivity D and a surface
held at unit concentration from t = 0 onwards, the concentration at depth x and
time t is

    c(x, t) = erfc( x / (2 sqrt(D t)) ),

and the quantity a probe responds to, the concentration gradient, is

    g(x, t) = -dc/dx = (2 / sqrt(pi t)) exp(-x^2 / t)   [at D = 1/4].

We fix D = 1/4 cm^2/s throughout so that 2 sqrt(D t) = sqrt(t) and the answer we
are hunting for is the tidy erfc(x / sqrt(t)).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#2e8b6f"
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

D = 0.25          # cm^2 / s, chosen so that 2 sqrt(D t) = sqrt(t)
NOISE = 0.002     # absolute measurement noise on both channels


def _finish(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def true_profile(x, t):
    """Concentration c(x, t), the thing we are trying to rediscover."""
    return erfc(x / (2.0 * np.sqrt(D * np.asarray(t, dtype=float))))


def true_gradient(x, t):
    """-dc/dx, which is the flux divided by D."""
    t = np.asarray(t, dtype=float)
    return np.exp(-(x ** 2) / (4.0 * D * t)) / np.sqrt(np.pi * D * t)


def measurements(n: int = 300, seed: int = 0, x_hi: float = 2.5,
                 t_lo: float = 0.25, t_hi: float = 4.0):
    """A campaign of n probe readings, each at its own depth and time.

    Returns X (depth, time), the concentration, and the gradient.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.02, x_hi, n)
    t = rng.uniform(t_lo, t_hi, n)
    c = true_profile(x, t) + rng.normal(0.0, NOISE, n)
    g = true_gradient(x, t) + rng.normal(0.0, NOISE, n)
    return np.stack([x, t], axis=1), c, g


def measurable(c, lo: float = 0.01, hi: float = 0.99):
    """Mask for readings a real probe could resolve, away from both rails."""
    return (c > lo) & (c < hi)


def plot_measurements(X, c):
    x, t = X[:, 0], X[:, 1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))

    sc = ax1.scatter(x, c, c=t, cmap="viridis", s=16, edgecolor="none")
    fig.colorbar(sc, ax=ax1, label="time  $t$  [s]")
    _finish(ax1, "every reading, by depth", "depth  $x$  [cm]", "concentration  $c$")

    ax2.scatter(x / np.sqrt(t), c, s=16, color=BLUE, edgecolor="none", alpha=0.85)
    _finish(ax2, "the same readings against $x/\\sqrt{t}$",
            "$x/\\sqrt{t}$  [cm s$^{-1/2}$]", "concentration  $c$")
    plt.show()


def plot_fronts(fronts: dict):
    """Loss against complexity for several Pareto fronts, on one pair of axes."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    colors = [BLUE, ORANGE, GREEN, MUTED]
    for (label, front), color in zip(fronts.items(), colors):
        ax.plot(front["complexity"], front["loss"], "o-", color=color,
                label=label, markersize=4)
    ax.set_yscale("log")
    ax.legend()
    _finish(ax, "what each search had to pay", "complexity", "loss")
    plt.show()


def plot_check(X, y, prediction, label="found equation", ylabel="concentration  $c$"):
    x, t = X[:, 0], X[:, 1]
    order = np.argsort(x / np.sqrt(t))
    u = (x / np.sqrt(t))[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax1.scatter(u, np.asarray(y)[order], s=16, color=MUTED, edgecolor="none",
                label="measured", alpha=0.8)
    ax1.plot(u, np.asarray(prediction)[order], color=ORANGE, label=label)
    ax1.legend()
    _finish(ax1, "prediction along the similarity variable",
            "$x/\\sqrt{t}$", ylabel)

    residual = np.asarray(y) - np.asarray(prediction)
    ax2.scatter(x / np.sqrt(t), residual, s=16, color=BLUE, edgecolor="none", alpha=0.8)
    ax2.axhline(0.0, color=AXIS, linewidth=1.0)
    _finish(ax2, f"residual, rms {np.sqrt(np.mean(residual ** 2)):.2e}",
            "$x/\\sqrt{t}$", "measured $-$ predicted")
    plt.show()


def front(model, index=None):
    """The three columns worth looking at, for printing."""
    table = model.equations_ if index is None else model.equations_[index]
    return table[["complexity", "loss", "equation"]]
