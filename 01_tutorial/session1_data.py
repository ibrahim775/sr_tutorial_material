from __future__ import annotations

import io
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"       # categorical slot 1
ORANGE = "#eb6834"     # categorical slot 2
AQUA = "#1baf7a"       # categorical slot 3

_BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
COMPLEXITY_CMAP = LinearSegmentedColormap.from_list("complexity", _BLUE_RAMP)


def use_house_style() -> None:
    """Applied on import. Sized for a projector: large type, hairline chrome."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 9,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.grid": True,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
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


# ======================================================================================
# Problem 1: driven, damped oscillator starting from rest
# ======================================================================================
X_FIT_MAX = 6.0        # everything past here is extrapolation, never shown to the search
X_VIEW_MAX = 12.0      # how far the plots look


def oscillator_truth(x):
    """y = 2.8 x exp(-x/2) cos(2.5 x), the thing we are trying to recover.

    A resonantly driven oscillator that starts at rest: the drive feeds amplitude in
    linearly while damping bleeds it away, so the envelope climbs from zero, peaks at
    x = 2 (one damping time), and rings down to nothing after that.
    """
    x = np.asarray(x, dtype=float)
    return 2.8 * x * np.exp(-x / 2.0) * np.cos(2.5 * x)


def oscillator_data(outliers: bool = False, seed: int = 0):
    """Noisy driven-and-damped oscillator, split at x = 6.

    Returns (x_fit, y_fit, x_out, y_out).  Only the first pair ever reaches PySR; the
    second is held back so we can ask what each model believes about x it never saw.
    With `outliers=True`, seven readings are corrupted the way a flaky sensor would.
    """
    g = np.random.default_rng(seed)
    x_fit = np.sort(g.uniform(0, X_FIT_MAX, 180))
    y_fit = oscillator_truth(x_fit) + g.normal(0, 0.10, 180)
    x_out = np.sort(g.uniform(X_FIT_MAX, X_VIEW_MAX, 120))
    y_out = oscillator_truth(x_out) + g.normal(0, 0.10, 120)

    if outliers:
        where = np.linspace(12, 170, 7).astype(int)
        y_fit[where] += np.array([2.5, -2.2, 2.8, -2.6, 2.4, -2.9, 2.7])
    return x_fit, y_fit, x_out, y_out


def extrapolation_r2(model, x_out, y_out, index=None):
    """R^2 of a fitted model against the clean signal on the held-out range."""
    pred = model.predict(np.asarray(x_out).reshape(-1, 1), index=index)
    clean = oscillator_truth(x_out)
    with np.errstate(all="ignore"):
        pred = np.nan_to_num(np.asarray(pred, dtype=float), nan=0.0,
                             posinf=1e12, neginf=-1e12)
    return float(1 - np.sum((pred - clean) ** 2) / np.sum((clean - clean.mean()) ** 2))


_MESSY_FRONT_CSV = """\
complexity,loss,score,equation
1,1.1301469,0.0,0.052830394
3,1.1244847,0.0025113689987216374,x * 0.024740662
4,0.23825672,1.551731417456819,cos(x / -0.3999987)
6,0.14942625,0.23327289386341263,cos(x * 2.4983828) * 1.4297336
8,0.13803428,0.039650449984668136,(cos(x * 2.491832) * 1.4559779) + -0.111233085
9,0.05971107,0.8379846297619462,sin(x / 0.332675) - sin(x + x)
11,0.045300595,0.13809863164209113,sin((x / 0.34198725) + 0.3357686) - sin(x + x)
13,0.035689335,0.11923413120115157,sin((x - -0.12285432) / 0.34225386) - sin((x + x) - 0.1462528)
15,0.015987447,0.4015240264243554,sin((x - -0.40645948) / 0.3597995) - sin(x + (x - cos(log(x))))
16,0.015767246,0.013869101468779057,sin((x - -0.3569613) / 0.3563433) - sin(x + (x - sin(cos(log(x)))))
17,0.014598306,0.07702925575544163,sin((x - -0.25627205) / 0.34993193) - sin((x + x) - (cos(log(x)) * 0.6559032))
18,0.0144707,0.008779579161775568,sin((x - -0.26071858) / 0.35012874) - sin((x + x) - sin(cos(log(x)) * 0.70620316))
19,0.013564581,0.06466385810523254,(sin((x - -0.2634795) / 0.35027787) * 0.96176213) - sin((x + x) - (cos(log(x)) * 0.65343434))
20,0.013338193,0.016830483277374693,(sin((x - -0.2721287) / 0.35080546) * 0.95511043) - sin((x + x) - sin(cos(log(x)) * 0.7422719))
21,0.013322527,0.0011752122075514868,(sin((x - -0.27124918) / 0.35068434) * 0.9558992) - sin((x + x) - sin(sin(cos(log(x)) * 0.7864833)))
"""

messy_front = pd.read_csv(io.StringIO(_MESSY_FRONT_CSV))


def has_nested_trig(equation: str) -> bool:
    """True when a sin or cos sits anywhere inside another sin or cos."""
    for match in re.finditer(r"(sin|cos)\(", equation):
        depth, i = 1, match.end()
        while i < len(equation) and depth > 0:
            depth += {"(": 1, ")": -1}.get(equation[i], 0)
            i += 1
        inner = equation[match.end():i - 1]
        if "sin(" in inner or "cos(" in inner:
            return True
    return False


messy_front["nested_trig"] = messy_front["equation"].map(has_nested_trig)


def _eval_equation(equation: str, x):
    """Evaluate a stored single-variable PySR equation string on a grid."""
    import sympy
    sym = sympy.Symbol("x")
    fn = sympy.lambdify(sym, sympy.sympify(equation), "numpy")
    with np.errstate(all="ignore"):
        out = np.asarray(fn(x), dtype=float) * np.ones_like(x)
    return out


_SERIES = [BLUE, ORANGE, AQUA, "#8a5cd6", "#c39b12"]


def _shade_extrapolation(ax, label=True):
    """Mark everything past the fitted range as territory the search never saw."""
    ax.axvspan(X_FIT_MAX, X_VIEW_MAX, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.axvline(X_FIT_MAX, color=AXIS, lw=1.1, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.annotate("extrapolation", (X_FIT_MAX + 0.15, 0.965), xycoords=("data", "axes fraction"),
                    fontsize=9.5, color=MUTED, ha="left", va="top")


def plot_oscillator_data(x_fit, y_fit, x_out, y_out):
    """The data as the colleague handed it over: measured on the left, dark on the right."""
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    _shade_extrapolation(ax)
    ax.plot(x_fit, y_fit, "o", ms=4.2, color=BLUE, mec=SURFACE, mew=0.6,
            label=f"fitted on  (x $\\leq$ {X_FIT_MAX:.0f})", zorder=3)
    ax.plot(x_out, y_out, "o", ms=4.2, mfc="none", color=MUTED, mew=1.1,
            label="held out", zorder=2)
    ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
    ax.set_xlim(0, X_VIEW_MAX)
    ax.legend(loc="upper right", ncol=2)
    _finish(ax, "An oscillator driven up from rest, then ringing down", "x", "y")
    plt.show()


def plot_fits(fits, x_fit, y_fit, x_out, y_out, title=None, span=2.6):
    """The workhorse figure: each fitted model drawn across the full range, with how well
    it holds up where it was never fitted carried in its legend entry.

    `fits` is a list of (label, model) pairs; each model is asked for its own selected
    equation.  Curves are clipped to a readable band, so a model that runs away simply
    leaves the frame, which is the honest picture of what it did.
    """
    grid = np.linspace(0, X_VIEW_MAX, 1400)
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    _shade_extrapolation(ax)
    ax.plot(grid, oscillator_truth(grid), color=MUTED, lw=1.6, ls=(0, (5, 3)),
            label="truth", zorder=2)
    ax.plot(x_fit, y_fit, "o", ms=3.4, color=MUTED, mec=SURFACE, mew=0.5, alpha=0.5, zorder=1)
    ax.plot(x_out, y_out, "o", ms=3.4, mfc="none", color=MUTED, mew=0.8, alpha=0.45, zorder=1)

    for i, (label, model) in enumerate(fits):
        colour = _SERIES[i % len(_SERIES)]
        curve = np.asarray(model.predict(grid.reshape(-1, 1)), dtype=float)
        raw = extrapolation_r2(model, x_out, y_out)
        if raw >= 0:
            score = f"$R^2 = {raw:.3f}$"
        else:
            score = f"$R^2 < 0$ ({raw:.1f})" if raw > -10 else "$R^2 < 0$"
        ax.plot(grid, _mask_outside(curve, -span, span), "-", color=colour, lw=2.3,
                label=f"{label}   ·   {score}", zorder=4 + i)

    ax.set_xlim(0, X_VIEW_MAX)
    ax.set_ylim(-span, span)
    ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
    ax.legend(loc="lower right", ncol=1, fontsize=9.5,
              title=f"$R^2$ measured on the held-out $x > {X_FIT_MAX:.0f}$", title_fontsize=9.0)
    _finish(ax, title or "Every fit, drawn past where it was fitted", "x", "y")
    plt.show()


def _mask_outside(curve, lo, hi, dilate: int = 8):
    curve = np.asarray(curve, dtype=float)
    bad = ~np.isfinite(curve) | (curve < lo) | (curve > hi)
    with np.errstate(all="ignore"):
        step = np.abs(np.diff(curve))
    finite = step[np.isfinite(step)]
    scale = np.median(finite) if finite.size else 0.0
    thresh = max(30.0 * scale, 0.02 * (hi - lo))
    big = ~np.isfinite(step) | (step > thresh)
    bad[:-1] |= big
    bad[1:] |= big
    if bad.any() and dilate:
        idx = np.flatnonzero(bad)
        for shift in range(1, dilate + 1):
            bad[np.clip(idx - shift, 0, len(bad) - 1)] = True
            bad[np.clip(idx + shift, 0, len(bad) - 1)] = True
    return np.where(bad, np.nan, curve)


def plot_complexity_sweep(front, x, y, title, predict=None, mark_row=None):
    """The centrepiece: every row of a Pareto front drawn over the data, coloured by
    complexity, beside loss against complexity.

    `predict` maps a front row to predictions on `x`; the default evaluates the stored
    equation string, which is what the pre-computed front needs.
    """
    predict = predict or (lambda row, grid: _eval_equation(row.equation, grid))
    grid = np.linspace(x.min(), x.max(), 2000)
    cx = front["complexity"].to_numpy(dtype=float)
    norm = mpl.colors.Normalize(vmin=cx.min(), vmax=cx.max())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9),
                                  gridspec_kw={"width_ratios": [1.55, 1]})
    ax.plot(x, y, "o", ms=3.4, color=MUTED, mec=SURFACE, mew=0.5, alpha=0.55, zorder=1)

    label_rows = {int(front.index[0]), int(front.index[-1])}
    if mark_row is not None:
        label_rows.add(int(mark_row))
    annotate_at = {int(front.index[0]): (0.30, "bottom", 9)}
    if mark_row is not None:
        annotate_at[int(mark_row)] = (0.62, "top", -11)
    lim = float(np.nanmax(np.abs(y))) * 1.08

    for idx, row in front.iterrows():
        curve = _mask_outside(predict(row, grid), -lim, lim)
        highlight = int(idx) in label_rows
        ax.plot(grid, curve, "-", color=COMPLEXITY_CMAP(norm(row.complexity)),
                lw=2.6 if highlight else 1.2, alpha=1.0 if highlight else 0.7,
                zorder=4 if highlight else 2)
        if int(idx) in annotate_at:
            frac, va, dy = annotate_at[int(idx)]
            j = np.isfinite(curve).nonzero()[0]
            if len(j):
                k = j[min(int(frac * len(j)), len(j) - 1)]
                ax.annotate(f"complexity {int(row.complexity)}", (grid[k], curve[k]),
                            xytext=(0, dy), fontsize=9.5, ha="center", va=va,
                            textcoords="offset points", fontweight="medium", color=INK_2)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color=AXIS, lw=0.9, zorder=0)
    _finish(ax, title, "x", "y")
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=COMPLEXITY_CMAP), ax=ax,
                 label="complexity", pad=0.015, fraction=0.045)

    ax2.plot(cx, front["loss"], "-", color=AXIS, lw=1.4, zorder=1)
    ax2.scatter(cx, front["loss"], s=52, c=cx, cmap=COMPLEXITY_CMAP, norm=norm,
                edgecolors=SURFACE, linewidths=1.1, zorder=3)
    if mark_row is not None:
        r = front.loc[mark_row]
        ax2.scatter([r.complexity], [r.loss], s=190, facecolors="none",
                    edgecolors=ORANGE, linewidths=2.2, zorder=4)
        ax2.annotate("Pareto elbow", (r.complexity, r.loss), xytext=(12, 14),
                     textcoords="offset points", fontsize=10, color=INK_2)
    ax2.set_yscale("log")
    _finish(ax2, "Loss against complexity", "complexity", "loss (MSE)")
    plt.show()


def plot_messy_front(front=None):
    """Loss against complexity for the unconstrained run, with the rows that put trig
    inside trig called out.

    The notebook fits that run live and passes its front in. `messy_front` above is only
    a fallback now: a real front from the same settings, kept so the segment still has
    something to talk about if the live fit misbehaves in the room.
    """
    front = messy_front if front is None else front
    bad = front["nested_trig"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.plot(front.complexity, front.loss, "-", color=AXIS, lw=1.4, zorder=1)
    ax.plot(front.complexity[~bad], front.loss[~bad], "o", ms=8, color=BLUE,
            mec=SURFACE, mew=1.4, label="Unconstrained", zorder=3)
    ax.plot(front.complexity[bad], front.loss[bad], "X", ms=11, color=ORANGE,
            mec=SURFACE, mew=1.4, label="Nested trig functions", zorder=4)
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    _finish(ax, "Unconstrained Pareto front",
            "complexity", "loss (MSE)")
    plt.show()


# ======================================================================================
# Problem 2: Kepler
# ======================================================================================
KEPLER_BODIES = ["Mercury", "Venus", "Earth", "Mars", "Ceres", "Jupiter",
                 "Saturn", "Uranus", "Neptune", "Pluto", "Eris", "Sedna"]
KEPLER_A = np.array([0.3871, 0.7233, 1.0000, 1.5237, 2.7658, 5.2029,
                     9.5367, 19.189, 30.070, 39.482, 67.781, 506.0])
KEPLER_T = np.array([0.2408, 0.6152, 1.0000, 1.8808, 4.6009, 11.862,
                     29.447, 84.017, 164.79, 247.94, 558.04, 11400.0])


def kepler_data():
    """Semi-major axis in AU and orbital period in years for twelve solar-system bodies."""
    return KEPLER_BODIES, KEPLER_A, KEPLER_T


def error_table(model, label, a=None, T=None):
    a = KEPLER_A if a is None else a
    T = KEPLER_T if T is None else T
    p = model.predict(a.reshape(-1, 1))
    return pd.DataFrame({"body": KEPLER_BODIES, "a [AU]": a, "T [yr]": T,
                         f"{label} pred": p.round(4),
                         f"{label} rel err %": (100 * np.abs(p - T) / T).round(3)})


def plot_kepler(mse_fit, log_fit, a=None, T=None):
    """Relative error body by body, which is the only view where the two fits differ.

    A log-log plot of period against semi-major axis was tried here and dropped: over four
    decades both fits lie on top of the data and on each other, so it carried no
    information. All of the difference is in the fractional error.
    """
    a = KEPLER_A if a is None else a
    T = KEPLER_T if T is None else T
    fits = [("fitted under MSE", mse_fit, ORANGE), ("fitted under the log loss", log_fit, BLUE)]
    xs = np.arange(len(a))

    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    for name, m, colour in fits:
        rel = 100 * np.abs(m.predict(a.reshape(-1, 1)) - T) / T
        ax.plot(xs, rel, "o-", ms=8, lw=1.8, color=colour, mec=SURFACE, mew=1.1,
                label=name, zorder=3)
    ax.axhline(0.1, color=AXIS, lw=1.0, zorder=1)
    ax.annotate("0.1%", (xs[-1] + 0.15, 0.1), xytext=(0, 4), textcoords="offset points",
                fontsize=9, color=MUTED)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(KEPLER_BODIES, rotation=45, ha="right")
    ax.set_xlim(-0.5, len(a) - 0.2)
    ax.legend(loc="lower left")
    _finish(ax, "Relative error body by body, inner planets on the left",
            None, "relative error [%]")
    plt.show()


# ======================================================================================
# Problem 4: spring-mass rigs
# ======================================================================================
K_1, M_1 = 4.0, 1.5


def single_rig(n: int = 200):
    """One rig, k = 4.0 N/m and m = 1.5 kg, giving E = 0.5 k x^2 + 0.5 m v^2."""
    g = np.random.default_rng(40)
    x = g.uniform(-1.5, 1.5, n)
    v = g.uniform(-2.0, 2.0, n)
    E = 0.5 * K_1 * x ** 2 + 0.5 * M_1 * v ** 2
    return x, v, E


def plot_components(model, x, v):
    """How far the two recovered halves of the template sit from the physics they
    should match. Overlaying the curves themselves is not informative here, since
    the recovered and true curves sit on top of each other at any scale a room can
    see; what is worth showing is exactly how small that gap is."""
    xs = np.linspace(-1.6, 1.6, 200)
    vs = np.linspace(-2.1, 2.1, 200)
    zero = np.zeros_like(xs)
    f_hat = model.predict(np.column_stack([xs, zero]))
    g_hat = model.predict(np.column_stack([np.zeros_like(vs), vs]))
    f_true = 0.5 * K_1 * xs ** 2
    g_true = 0.5 * M_1 * vs ** 2

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))
    for ax, grid, hat, truth, name, label, xlabel in [
        (axes[0], xs, f_hat, f_true, "f(x)", rf"$\frac{{1}}{{2}}kx^2$, k = {K_1:g}", "x"),
        (axes[1], vs, g_hat, g_true, "g(v)", rf"$\frac{{1}}{{2}}mv^2$, m = {M_1:g}", "v"),
    ]:
        resid = np.asarray(hat) - truth
        ax.plot(grid, resid, "-", lw=2.0, color=BLUE, zorder=3)
        ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
        ax.set_title(f"recovered {name} minus {label}", fontsize=11)
        _finish(ax, None, xlabel, "error [J]" if ax is axes[0] else None)
    plt.show()


K_TRUE = np.array([1.0, 5.0, 2.0])
M_TRUE = np.array([0.5, 2.0, 1.0])
K4, M4 = 3.0, 1.5


def rig_data(k_list=None, m_list=None, n_per: int = 100, seed: int = 4):
    """Three rigs stacked into one table of x, v, rig id and total energy."""
    k_list = K_TRUE if k_list is None else k_list
    m_list = M_TRUE if m_list is None else m_list
    g = np.random.default_rng(seed)
    blocks = []
    for c, (k, m) in enumerate(zip(k_list, m_list)):
        xx = g.uniform(-1.5, 1.5, n_per)
        vv = g.uniform(-2.0, 2.0, n_per)
        blocks.append(np.column_stack([xx, vv, np.full(n_per, float(c)),
                                       0.5 * k * xx ** 2 + 0.5 * m * vv ** 2]))
    D = np.vstack(blocks)
    g.shuffle(D)
    return D[:, :3], D[:, 3]


def read_coefficients(model, index, rig_column):
    """Probe a fitted expression at (x=1, v=0) and (x=0, v=1) to expose 1/2 k and 1/2 m.

    Whatever the expression does with the rig label, setting v=0 kills the velocity term,
    so the prediction at x=1 is that rig's 1/2 k. Works for any rig value handed in.
    """
    n = len(rig_column)
    at_x = np.column_stack([np.ones(n), np.zeros(n), rig_column])
    at_v = np.column_stack([np.zeros(n), np.ones(n), rig_column])
    return model.predict(at_x, index=index), model.predict(at_v, index=index)


def plot_rig_parabolas(model, index):
    """Energy against displacement at v = 0, one curve per rig, including the rig the
    search never saw. Unchanged in form from the version the class reveal is built on."""
    xs = np.linspace(-1.5, 1.5, 160)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for c in range(3):
        curve = model.predict(
            np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, float(c))]), index=index)
        ax.plot(xs, curve, "-", lw=2.0, color=BLUE,
                label="rigs 0-2 (measured)" if c == 0 else None)
        ax.annotate(f"rig {c}", (xs[-1], curve[-1]), xytext=(6, 0), fontsize=9.5,
                    textcoords="offset points", color=BLUE, va="center")
    curve3 = model.predict(
        np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, 3.0)]), index=index)
    ax.plot(xs, curve3, "--", lw=2.8, color=ORANGE, label="rig 3 (never measured)")
    ax.annotate("rig 3", (xs[-1], curve3[-1]), xytext=(6, 0), fontsize=9.5,
                textcoords="offset points", color=ORANGE, va="center")
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    ax.set_xlim(-1.7, 1.98)
    ax.legend(loc="lower center")
    _finish(ax, "What the fitted expression says each rig does",
            "displacement x [m]", "predicted E at v = 0 [J]")
    plt.show()


# ======================================================================================
# Potential from measured forces
# ======================================================================================
K_POT, B_POT = 4.0, 1.0


def potential_truth(x):
    """V(x) = 1/2 k x^2 + 1/4 b x^4, the thing we never get to measure directly."""
    return 0.5 * K_POT * x ** 2 + 0.25 * B_POT * x ** 4


def potential_data(n: int = 300, seed: int = 11):
    """Forces on an anharmonic spring: F = -dV/dx = -(k x + b x^3), plus noise."""
    g = np.random.default_rng(seed)
    x = np.sort(g.uniform(-2.0, 2.0, n))
    F = -(K_POT * x + B_POT * x ** 3) + g.normal(0, 0.05, n)
    return x, F


def read_potential(model, index):
    """Pull V out of a template row and return (sympy expr, callable shifted to V(0)=0).

    Template rows print as `V = (#1 * ...) * #1`, where `#1` is the component's first
    argument, so the string needs the name stripped and `#1` renamed before sympify.
    """
    import sympy
    equation = model.equations_.equation[index]
    rhs = equation.split("=", 1)[1] if "=" in equation else equation
    expr = sympy.sympify(rhs.replace("#1", "x"))
    fn = sympy.lambdify(sympy.Symbol("x"), expr, "numpy")

    def shifted(t):
        t = np.asarray(t, dtype=float)
        out = np.asarray(fn(t), dtype=float) * np.ones_like(t)
        return out - float(np.asarray(fn(0.0), dtype=float))

    return sympy.simplify(expr), shifted


def plot_potential(model, x, F, index):
    """Left: the forces we measured, and what the fitted V predicts for them.
    Right: the potential itself, which nothing in the data ever showed us."""
    _, V_fit = read_potential(model, index)
    grid = np.linspace(x.min(), x.max(), 600)
    pred = model.predict(x.reshape(-1, 1), index=int(index))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.7))
    ax.plot(x, F, "o", ms=3.4, color=MUTED, mec=SURFACE, mew=0.5, alpha=0.6,
            label="measured F", zorder=2)
    ax.plot(x, pred, "-", lw=2.0, color=BLUE, label="-dV/dx from the fit", zorder=3)
    ax.legend(loc="upper right")
    _finish(ax, "What we measured", "x", "F")

    ax2.plot(grid, potential_truth(grid), lw=3.4, color=GRID, label="true V", zorder=2)
    ax2.plot(grid, V_fit(grid), "--", lw=2.0, color=ORANGE, label="recovered V", zorder=3)
    ax2.legend(loc="upper center")
    _finish(ax2, "What we wanted, up to an additive constant", "x", "V")
    plt.show()


# ======================================================================================
# Problem 6: force law, distilled out of a network
# ======================================================================================
def force_law_data(n: int, seed: int):
    """F(r, s) = 1/r^2 + 0.3 s, returned as float32 numpy so the notebook can wrap it
    in tensors itself. No torch import here on purpose."""
    g = np.random.default_rng(seed)
    r = g.uniform(0.5, 2.0, n)
    s = g.uniform(-1.0, 1.0, n)
    X = np.column_stack([r, s]).astype(np.float32)
    y = (1.0 / r ** 2 + 0.3 * s).astype(np.float32).reshape(-1, 1)
    return X, y


def plot_distillation(X_test, y_true, mlp_pred, sym_pred):
    """The residual each model leaves behind on held-out data, against separation.

    A truth-against-prediction scatter was tried here and dropped: both series sit
    on the diagonal at any scale a room can read, so it said nothing the residual
    panel below doesn't say better."""
    y_true = np.asarray(y_true).ravel()
    series = [("the network", np.asarray(mlp_pred).ravel(), ORANGE),
              ("the distilled equation", np.asarray(sym_pred).ravel(), BLUE)]
    r = np.asarray(X_test)[:, 0]

    fig, ax2 = plt.subplots(figsize=(9.0, 4.6))
    for name, pred, colour in series:
        ax2.plot(r, np.abs(pred - y_true), "o", ms=6.5, color=colour, mec=SURFACE,
                 mew=0.6, label=name, zorder=3)
    ax2.set_yscale("log")
    ax2.legend(loc="upper right")
    _finish(ax2, "Absolute residual against separation, held-out data", "r", "|residual|")
    plt.show()
