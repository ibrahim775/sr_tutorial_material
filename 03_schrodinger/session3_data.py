from __future__ import annotations

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"        # categorical slot 1
ORANGE = "#eb6834"      # categorical slot 2

DENSITY = LinearSegmentedColormap.from_list(
    "density", [SURFACE, "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"])


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


# =========================================================================
# Part I, one dimension: the harmonic oscillator
# =========================================================================

COLS_1D = ["x", "V", "w"]

HARMONIC_LEVELS = [0.5, 1.5, 2.5]


def harmonic(x):
    """V = x^2/2, whose levels are E_n = n + 1/2."""
    return 0.5 * np.asarray(x, dtype=float) ** 2


def harmonic_energy(n: int) -> float:
    return n + 0.5


def harmonic_exact(n: int):
    """Eigenfunction n, up to normalisation: a Hermite polynomial times a Gaussian."""
    poly = {0: lambda x: np.ones_like(x),
            1: lambda x: 2.0 * x,
            2: lambda x: 4.0 * x ** 2 - 2.0}[n]

    def psi(x):
        x = np.asarray(x, dtype=float)
        return poly(x) * np.exp(-x ** 2 / 2)

    return psi


def poschl_teller(x):
    """V = -sech^2 x, kept for the exercises. One bound state, at E = -1/2."""
    return -1.0 / np.cosh(np.asarray(x, dtype=float)) ** 2


def collocation(potential=harmonic, lo: float = -4.0, hi: float = 4.0,
                n: int = 120, anchor: float = 20.0):

    x = np.linspace(lo, hi, n)
    X = np.column_stack([x, potential(x),
                         anchor * np.exp(2.0 * (np.abs(x) - hi))])
    return X, np.zeros(n)


def wavefunction(equation: str, node_factor=None, name: str = "f"):
    """psi(x) = N(x) f(x), with N whatever node structure the template imposed."""
    _, f = component(equation, name, var="x")
    if node_factor is None:
        return f
    return lambda g: node_factor(np.asarray(g, dtype=float)) * f(g)


def tail_ratio_1d(psi, edge: float = 8.0, n: int = 601) -> float:
    """|psi| at the edges of a box twice the fitting window, against its peak."""
    g = np.linspace(-edge, edge, n)
    v = np.abs(psi(g))
    if not np.all(np.isfinite(v)) or v.max() == 0:
        return np.inf
    return float(max(v[0], v[-1]) / v.max())


def classify_front_1d(model, node_factor=None, loss_tol: float = 1e-6,
                      decay_tol: float = 1e-2, name: str = "f") -> pd.DataFrame:
    """The one-dimensional twin of `classify_front`."""
    return _classify(model,
                     lambda eq: wavefunction(eq, node_factor, name),
                     tail_ratio_1d, loss_tol, decay_tol)


def physical_rows(front: pd.DataFrame) -> pd.DataFrame:
    """Rows that both solve the equation and decay, simplest first."""
    return front[front.bound].sort_values("complexity")


# ------------------------------------------------------- exact solutions ---

# (n, l): radial function up to normalisation, and its energy -1/(2 n^2)
EXACT = {
    (1, 0): (lambda r: np.exp(-r), "e^{-r}"),
    (2, 0): (lambda r: (2.0 - r) * np.exp(-r / 2), "(2-r)e^{-r/2}"),
    (2, 1): (lambda r: r * np.exp(-r / 2), "r e^{-r/2}"),
    (3, 2): (lambda r: r ** 2 * np.exp(-r / 3), "r^2 e^{-r/3}"),
}

LABEL = {(1, 0): "1s", (2, 0): "2s", (2, 1): "2p", (3, 2): "3d"}


def exact_radial(n: int, l: int):
    return EXACT[(n, l)][0]


def exact_energy(n: int) -> float:
    return -0.5 / n ** 2


def oscillator_energy(n_r: int, l: int) -> float:
    """Levels of the 3D isotropic oscillator, E = 2 n_r + l + 3/2.

    Worth having beside `exact_energy` because the contrast is the point: here
    the energy depends on n_r and l separately, which is what one expects of a
    central potential, and for hydrogen it collapses onto n = n_r + l + 1.
    """
    return 2.0 * n_r + l + 1.5


def coulomb(r):
    """Hydrogen, in atomic units."""
    return -1.0 / np.asarray(r, dtype=float)


def oscillator(r):
    """The 3D isotropic harmonic oscillator, for contrast."""
    return 0.5 * np.asarray(r, dtype=float) ** 2


# ------------------------------------------------------------ collocation ---

COLS = ["r", "L", "V", "w"]


def radial_grid(l: int, potential=None, lo: float = 0.4, hi: float = 24.0,
                n: int = 100, anchor: float = 30.0):

    r = np.linspace(lo, hi, n)
    X = np.column_stack([r, np.full_like(r, l * (l + 1.0)),
                         (potential or coulomb)(r),
                         anchor * np.exp(1.5 * (r - hi))])
    return X, np.zeros(n)


# ------------------------------------------- reading a component back out ---

def energy(equation: str) -> float:
    m = re.search(r"\bE = \[([^\]]+)\]", str(equation))
    if not m:
        raise ValueError(f"no E parameter in {equation!r}")
    return float(m.group(1))


def parameter(equation: str, name: str) -> float:
    m = re.search(rf"\b{re.escape(name)} = \[([^\]]+)\]", str(equation))
    if not m:
        raise ValueError(f"no {name} parameter in {equation!r}")
    return float(m.group(1))


def component(equation: str, name: str = "R", var: str = "r"):

    text = str(equation)
    m = re.search(rf"\b{re.escape(name)} = (.*?)(?:; [A-Za-z_]\w* = |$)", text)
    if not m:
        raise ValueError(f"no component {name!r} in {equation!r}")
    expr = sympy.sympify(m.group(1).strip().replace("#1", var))
    fn = sympy.lambdify(sympy.Symbol(var), expr, "numpy")

    def evaluate(grid):
        grid = np.asarray(grid, dtype=float)
        with np.errstate(all="ignore"):
            return np.asarray(fn(grid), dtype=float) * np.ones_like(grid)

    return expr, evaluate


def radial_function(equation: str, node_factor=None, name: str = "R"):
    """R(r) = N(r) f(r), with N whatever node structure the template imposed."""
    _, f = component(equation, name)
    if node_factor is None:
        return f
    return lambda g: node_factor(np.asarray(g, dtype=float)) * f(g)


def tail_ratio(R, lo: float = 0.4, hi: float = 60.0, n: int = 600) -> float:
    """|R| far outside the fitting window against its largest value inside."""
    g = np.linspace(lo, hi, n)
    v = np.abs(R(g))
    if not np.all(np.isfinite(v)) or v.max() == 0:
        return np.inf
    return float(v[-1] / v.max())


def _classify(model, to_function, tail, loss_tol: float, decay_tol: float):

    rows = []
    for _, r in model.equations_.reset_index(drop=True).iterrows():
        rec = {"complexity": int(r.complexity), "loss": float(r.loss),
               "E": np.nan, "decay": np.nan, "solved": bool(r.loss < loss_tol),
               "bound": False, "equation": str(r.equation)}
        try:
            rec["E"] = energy(r.equation)
            rec["decay"] = tail(to_function(r.equation))
            rec["bound"] = bool(rec["solved"] and rec["decay"] < decay_tol)
        except Exception:
            pass
        rows.append(rec)
    out = pd.DataFrame(rows)
    for c in ("E", "decay", "loss"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def classify_front(model, node_factor=None, loss_tol: float = 1e-6,
                   decay_tol: float = 1e-2, name: str = "R") -> pd.DataFrame:
    """Front rows with their energy, and whether each is actually a bound state."""
    return _classify(model,
                     lambda eq: radial_function(eq, node_factor, name),
                     tail_ratio, loss_tol, decay_tol)


def simplest_within(front: pd.DataFrame, factor: float = 20.0):

    ok = front[np.isfinite(front.loss)]
    if ok.empty:
        raise ValueError("no usable rows on this front")
    keep = ok[ok.loss <= ok.loss.min() * factor]
    return keep.sort_values("complexity").iloc[0]


def best_bound(front: pd.DataFrame):
    """Simplest row that both solves the equation and decays."""
    ok = front[front.bound]
    if ok.empty:
        raise ValueError("no bound state on this front; widen the search or "
                         "check the anchor")
    return ok.sort_values("complexity").iloc[0]


# ------------------------------------------------------------------ shapes ---

def angular(l: int, m: int, theta):
    """Real spherical harmonic, up to normalisation, for the cases we use."""
    theta = np.asarray(theta, dtype=float)
    if (l, m) == (0, 0):
        return np.ones_like(theta)
    if (l, m) == (1, 0):
        return np.cos(theta)
    if (l, m) == (2, 0):
        return 1.5 * np.cos(theta) ** 2 - 0.5
    raise ValueError(f"no closed form wired up for (l, m) = ({l}, {m})")


def density_slice(R, l: int = 0, m: int = 0, extent: float = 12.0, n: int = 420):
    """|psi|^2 on a plane through the nucleus, for drawing the orbital."""
    g = np.linspace(-extent, extent, n)
    xx, zz = np.meshgrid(g, g)
    rr = np.hypot(xx, zz)
    rr = np.where(rr < 1e-9, 1e-9, rr)
    theta = np.arccos(np.clip(zz / rr, -1.0, 1.0))
    with np.errstate(all="ignore"):
        psi = R(rr) * angular(l, m, theta)
    d = np.abs(psi) ** 2
    return np.where(np.isfinite(d), d, 0.0), extent


# ------------------------------------------------- figures, one dimension ---

def _unit(v):
    """Scale to peak one and fix the sign, since only shape was ever determined."""
    v = np.asarray(v, dtype=float)
    p = np.abs(v[np.isfinite(v)]).max() if np.any(np.isfinite(v)) else 0.0
    if p == 0 or not np.isfinite(p):
        return v
    v = v / p
    return -v if v[np.nanargmax(np.abs(v))] < 0 else v


def plot_potential(potential=harmonic, levels=HARMONIC_LEVELS, lo=-4.0, hi=4.0):
    """The well with its levels drawn between the classical turning points."""
    x = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(x, potential(x), color=INK_2, lw=2.0, zorder=3)
    for n, E in enumerate(levels):
        turn = np.sqrt(2.0 * E)
        ax.plot([-turn, turn], [E, E], color=BLUE, lw=3.0,
                solid_capstyle="round", zorder=4)
        ax.text(turn + 0.12, E, f"$E_{n}$ = {E:g}", va="center", ha="left",
                color=INK, fontsize=10.5)
    ax.set_ylim(-0.35, max(levels) + 1.15)
    ax.set_xlim(lo, hi)
    _finish(ax, "The harmonic oscillator and levels",
            "x", "energy")
    plt.show()


def plot_candidate(psi_found, psi_exact=None, title="Recovered against exact",
                   lo=-4.0, hi=4.0):
    x = np.linspace(lo, hi, 500)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    found = _unit(psi_found(x))
    if psi_exact is not None:
        exact = _unit(psi_exact(x))
        ok = np.isfinite(exact) & np.isfinite(found)
        if ok.any() and np.dot(exact[ok], found[ok]) < 0:
            found = -found
        ax.plot(x, exact, color=AXIS, lw=5.0, solid_capstyle="round", zorder=2,
                label="exact")
    ax.plot(x, found, color=BLUE, lw=2.0, zorder=3, label="recovered")
    ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
    ax.legend(loc="upper right")
    _finish(ax, title, "x", r"$\psi$  [scaled to peak 1]")
    plt.show()


def plot_runaway(entries, lo=-6.0, hi=6.0, window=4.0):

    x = np.linspace(lo, hi, 600)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.axvspan(-window, window, color=GRID, alpha=0.55, zorder=1,
               label="where the residual was checked")
    for (label, fn), col in zip(entries, [BLUE, ORANGE]):
        v = np.abs(np.asarray(fn(x), dtype=float))
        fin = v[np.isfinite(v)]
        v = v / (fin.max() if len(fin) and fin.max() > 0 else 1.0)
        ax.semilogy(x, np.clip(v, 1e-12, None), color=col, lw=2.2, zorder=3,
                    label=label)
    ax.set_ylim(1e-11, 1e3)
    ax.legend(loc="lower center", ncol=3)
    _finish(ax, "Both satisfy the equation at every point we sampled",
            "x", r"$|\psi|$,  scaled to peak 1")
    plt.show()


def plot_local_energy_1d(psi, potential=harmonic, E=None, lo=-3.0, hi=3.0):
    """Local energy of a recovered psi, differentiated numerically as a check.

    Computed on a grid wider than the one drawn, because `np.gradient` is
    one sided at the two end points and would put a spike at each edge that is
    an artefact of the difference stencil rather than anything about psi.
    """
    pad = 0.1 * (hi - lo)
    g = np.linspace(lo - pad, hi + pad, 640)
    p = np.asarray(psi(g), dtype=float)
    d2 = np.gradient(np.gradient(p, g), g)
    with np.errstate(all="ignore"):
        e = (-0.5 * d2 + potential(g) * p) / p
    keep = (g >= lo) & (g <= hi)
    g, e = g[keep], e[keep]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.plot(g, e, color=BLUE, lw=3.0, zorder=3, label="local energy")
    if E is not None:
        ax.axhline(E, color=ORANGE, lw=1.4, ls=(0, (5, 4)), zorder=4,
                   label=f"fitted E = {E:.5f}")
    fin = e[np.isfinite(e)]
    if len(fin):
        mid = float(np.median(fin))
        ax.set_ylim(mid - 0.25, mid + 0.25)
    ax.legend(loc="upper right")
    _finish(ax, r"$(\hat H\psi)/\psi$, flat for a solution", "x", "local energy")
    plt.show()


def plot_spectrum(found: dict, levels=HARMONIC_LEVELS):
    """The recovered ladder against the exact one, and the gaps between rungs.

    Left panel is the spectrum itself, exact as a wide grey rule with the
    recovered value laid over it. Right panel is the thing worth reading, which
    is the spacing: three searches that never exchanged a number, and the rungs
    come out one unit apart.
    """
    ns = sorted(found)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))

    ax = axes[0]
    for n in ns:
        ax.plot([n - 0.32, n + 0.32], [levels[n]] * 2, color=AXIS, lw=6.0,
                solid_capstyle="round", zorder=2,
                label="exact  $n + 1/2$" if n == ns[0] else None)
        ax.plot([n - 0.32, n + 0.32], [found[n]] * 2, color=BLUE, lw=2.4,
                solid_capstyle="round", zorder=3,
                label="recovered" if n == ns[0] else None)
        ax.text(n, found[n] + 0.10, f"{found[n]:.5f}", ha="center", va="bottom",
                color=INK, fontsize=10.5, zorder=4)
    ax.set_xticks(ns)
    ax.set_xticklabels([f"n = {n}" for n in ns])
    ax.set_xlim(min(ns) - 0.65, max(ns) + 0.65)
    ax.set_ylim(0.0, max(levels[n] for n in ns) + 0.55)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    _finish(ax, "The ladder, recovered", None, "energy")

    ax = axes[1]
    gaps = [found[b] - found[a] for a, b in zip(ns, ns[1:])]
    pos = np.arange(len(gaps))
    ax.axhline(1.0, color=AXIS, lw=1.4, ls=(0, (5, 4)), zorder=2)
    ax.text(-0.66, 1.0, "exact 1", va="bottom", ha="left", color=INK_2,
            fontsize=10.5)
    ax.bar(pos, gaps, width=0.42, color=BLUE, zorder=3)
    for p, g in zip(pos, gaps):
        ax.text(p, g + 0.03, f"{g:.5f}", ha="center", va="bottom", color=INK,
                fontsize=10.5)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{b} - {a}" for a, b in zip(ns, ns[1:])])
    ax.set_xlim(-0.7, len(gaps) - 0.3)
    ax.set_ylim(0, 1.35)
    ax.grid(axis="x", visible=False)
    _finish(ax, "Spacing between rungs, against an exact 1", None, "gap")
    plt.show()


# --------------------------------------------- figures, three dimensions ---

def plot_orbitals(entries, extent: float = 12.0, gamma: float = 0.42):
    """|psi|^2 through the nucleus, one panel per state.

    Each panel is normalised to its own peak, since the point is the shape rather
    than a comparison of magnitudes between states, and a mild gamma keeps the
    outer lobes visible without a log scale that would invent structure in the
    numerical floor.
    """
    fig, axes = plt.subplots(1, len(entries), figsize=(3.5 * len(entries), 3.9))
    axes = np.atleast_1d(axes)
    for ax, (title, R, l, m) in zip(axes, entries):
        d, ext = density_slice(R, l, m, extent=extent)
        peak = d.max()
        shown = (d / peak) ** gamma if peak > 0 else d
        ax.imshow(shown, origin="lower", extent=[-ext, ext, -ext, ext],
                  cmap=DENSITY, vmin=0.0, vmax=1.0, interpolation="bilinear")
        ax.set_title(title, color=INK)
        ax.set_xlabel("x  [a.u.]")
        ax.set_ylabel("z  [a.u.]" if ax is axes[0] else None)
        ax.grid(False)
        ax.set_xticks([-extent, 0, extent])
        ax.set_yticks([-extent, 0, extent])
    plt.show()


def plot_radial(R_found, n: int, l: int, hi: float = 22.0):
    """Recovered against exact, as R(r) and as the radial probability r^2 R^2.

    Both curves are scaled to peak at one, because the residual we minimised is
    invariant under R -> cR, so only the shape was ever determined.
    """
    r = np.linspace(0.05, hi, 500)
    R_true = exact_radial(n, l)

    def unit(v):
        v = np.asarray(v, dtype=float)
        p = np.abs(v).max()
        if p == 0 or not np.isfinite(p):
            return v
        v = v / p
        return -v if v[np.argmax(np.abs(v))] < 0 else v

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    for ax, f, lab in [(axes[0], lambda g, F: unit(F(g)), r"$R(r)$"),
                       (axes[1], lambda g, F: unit(g ** 2 * F(g) ** 2),
                        r"$r^2 R(r)^2$")]:
        ax.plot(r, f(r, R_true), color=AXIS, lw=5.0, solid_capstyle="round",
                zorder=2, label="exact")
        ax.plot(r, f(r, R_found), color=BLUE, lw=2.0, zorder=3, label="recovered")
        ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
        _finish(ax, lab, "r  [a.u.]", None)
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"{LABEL[(n, l)]}:  R(r)", color=INK)
    axes[1].set_title("radial probability", color=INK)
    plt.show()


def plot_local_energy(R, l: int, E=None, lo: float = 1.0, hi: float = 16.0,
                      potential=None):
    """Local energy of a recovered R, differentiated numerically as a check.

    Independent of whatever the search was optimising, so a flat line here is
    evidence rather than a restatement.
    """
    pad = 0.12 * (hi - lo)
    g = np.linspace(lo - pad, hi + pad, 600)
    R_ = R(g)
    d1 = np.gradient(R_, g)
    d2 = np.gradient(d1, g)
    with np.errstate(all="ignore"):
        e = (-0.5 * (d2 + 2 * d1 / g - l * (l + 1) * R_ / g ** 2)
             + (potential or coulomb)(g) * R_) / R_
    keep = (g >= lo) & (g <= hi)

    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.plot(g[keep], e[keep], color=BLUE, lw=2.0, zorder=3, label="local energy")
    if E is not None:
        ax.axhline(E, color=ORANGE, lw=1.4, ls=(0, (5, 4)), zorder=2,
                   label=f"fitted E = {E:.5f}")
    fin = e[keep][np.isfinite(e[keep])]
    if len(fin):
        mid = float(np.median(fin))
        ax.set_ylim(mid - 0.25, mid + 0.25)
    ax.legend(loc="upper right")
    _finish(ax, r"$(\hat H\psi)/\psi$, flat for a solution", "r  [a.u.]",
            "local energy  [a.u.]")
    plt.show()


def plot_angular_check(R, l: int, m: int = 0, radii=(2.0, 5.0, 9.0), potential=None):
    """Local energy of the full psi = R(r) Y_lm(theta), against polar angle.

    This has to differentiate the actual angular factor, not stand in a constant
    for it, or the check proves nothing: theta never enters and the lines would
    be flat whether or not the cancellation the notebook claims is real. The
    angular part of the Laplacian, (1/sin theta) d/dtheta(sin theta dY/dtheta),
    is built from `angular()` by finite differences and combined with the
    radial second derivative of R, so the only way this comes out flat is if the
    two genuinely add up to E for every theta, at every radius.

    The three lines are drawn at three widths and named in a legend rather than
    labelled where they overlap. The y-range is a physical quarter-hartree either
    side of the common value: left to autoscale it would zoom onto the finite
    difference noise floor and dress it up as structure.
    """
    th = np.linspace(0.05, np.pi - 0.05, 200)
    h = 1e-3

    def angular_laplacian(theta):
        # (1/sin theta) d/dtheta( sin theta dY/dtheta ), by finite differences
        # of Y itself, not by assuming the -l(l+1) eigenvalue it should equal.
        dY_plus = (angular(l, m, theta + h) - angular(l, m, theta)) / h
        dY_minus = (angular(l, m, theta) - angular(l, m, theta - h)) / h
        flux_plus = np.sin(theta + h / 2) * dY_plus
        flux_minus = np.sin(theta - h / 2) * dY_minus
        return (flux_plus - flux_minus) / h / np.sin(theta)

    style = [(5.5, AXIS, "-"), (2.8, BLUE, "-"), (1.4, ORANGE, (0, (5, 4)))]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    seen = []
    for i, rad in enumerate(radii):
        R0, Rp, Rm = R(rad), R(rad + h), R(rad - h)
        d1r = (Rp - Rm) / (2 * h)
        d2r = (Rp - 2 * R0 + Rm) / h ** 2
        Y0 = angular(l, m, th)
        lap = Y0 * (d2r + 2.0 / rad * d1r) + R0 * angular_laplacian(th) / rad ** 2
        with np.errstate(all="ignore"):
            e = (-0.5 * lap + (potential or coulomb)(rad) * R0 * Y0) / (R0 * Y0)
        lw, col, ls = style[i % len(style)]
        ax.plot(th, e, lw=lw, color=col, ls=ls, solid_capstyle="round",
                zorder=3 + i, label=f"r = {rad:g}")
        seen.append(e[np.isfinite(e)])
    fin = np.concatenate(seen) if seen else np.array([0.0])
    mid = float(np.median(fin))
    ax.set_ylim(mid - 0.25, mid + 0.25)
    ax.set_xlim(0, np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi])
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
    ax.legend(loc="upper right", ncol=3)
    _finish(ax, "Local energy of the full 3D state, against polar angle",
            r"$\theta$", "local energy  [a.u.]")
    plt.show()


def plot_levels(panels, ylim=None):
    span_of = lambda Es: (max(Es) - min(Es))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.3))
    axes = np.atleast_1d(axes)
    for ax, (title, levels) in zip(axes, panels):
        Es = [E for _, (_, E) in levels.items()]
        ls = [l for _, (l, _) in levels.items()]
        gap = span_of(Es)
        degenerate = len(Es) > 1 and gap < 1e-3

        if ylim is None:
            pad = max(0.55 * gap, 0.06 * max(1e-9, abs(np.mean(Es))), 1e-3)
            lo_, hi_ = min(Es) - 2.1 * pad, max(Es) + 2.1 * pad
        else:
            lo_, hi_ = ylim
        span = hi_ - lo_
        ax.set_ylim(lo_, hi_)

        # Print the gap as measured. Calling a 2e-5 splitting "zero" would be
        # claiming more than the search resolves; the contrast speaks anyway.
        ax.text(0.5, 0.055, f"splitting  {gap:.5f}", transform=ax.transAxes,
                ha="center", color=ORANGE, fontsize=11)

        # The dashed rule first, so the level marks sit on top of it, and the
        # labels ride above their own mark where nothing can collide with them.
        if degenerate:
            ax.plot([min(ls) - 0.42, max(ls) + 0.42], [np.mean(Es)] * 2,
                    color=ORANGE, lw=1.2, ls=(0, (5, 4)), zorder=2)
            ax.annotate("same energy", xy=(np.mean(ls), np.mean(Es)),
                        xytext=(np.mean(ls), np.mean(Es) - 0.22 * span),
                        ha="center", color=ORANGE, fontsize=10.5,
                        arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.0))
        for label, (l, E) in levels.items():
            ax.plot([l - 0.3, l + 0.3], [E, E], color=BLUE, lw=3.5,
                    solid_capstyle="round", zorder=4)
            ax.text(l, E + 0.035 * span, f"{label}  {E:+.5f}", va="bottom",
                    ha="center", color=INK, fontsize=10.5, zorder=5)

        ax.set_xlim(min(ls) - 0.75, max(ls) + 0.75)
        ax.set_xticks(sorted(set(ls)))
        ax.set_xticklabels([f"l = {l}" for l in sorted(set(ls))])
        ax.grid(axis="x", visible=False)
        _finish(ax, title, None, "energy  [a.u.]" if ax is axes[0] else None)
    plt.show()
