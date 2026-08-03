"""Loading, feature views and figures for Session 2.

The measurements are NASA's airfoil self-noise dataset: a series of two-dimensional
and three-dimensional NACA 0012 blade sections put through anechoic wind-tunnel
tests at a range of speeds and angles of attack, published by Brooks, Pope and
Marcolini as NASA Reference Publication 1218 (1989) and distributed since through
the UCI repository.

    https://archive.ics.uci.edu/dataset/291/airfoil+self+noise

Every row is one one-third-octave band of one run. Five numbers describe the
condition, the sixth is the scaled sound pressure level in decibels that the
microphones recorded, and the whole file is 1503 rows, which is small enough that
a search fits in under a minute and the room can try things.

Nothing here downloads at runtime; the file ships under data/. The loader will
fall back to UCI if that copy is ever missing, but conference wifi should never
be on the critical path.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
CSV = DATA / "airfoil_self_noise.csv.gz"
UCI = "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip"

COLUMNS = ["freq", "aoa", "chord", "U", "delta", "spl"]

# What each column is and what it is measured in, kept next to the data because
# the units are the whole point of the Strouhal argument later on.
MEANING = {
    "freq":  ("centre frequency of the one-third-octave band", "Hz"),
    "aoa":   ("geometric angle of attack", "degrees"),
    "chord": ("chord length of the blade section", "m"),
    "U":     ("free-stream velocity", "m/s"),
    "delta": ("suction-side boundary-layer displacement thickness", "m"),
    "spl":   ("scaled sound pressure level", "dB"),
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
PLUM = "#8a5cd6"
GOLD = "#c39b12"

SERIES = [BLUE, ORANGE, AQUA, PLUM, GOLD]


def use_house_style() -> None:
    """Applied on import, matching Session 1 so the two decks look like one thing."""
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


# ----------------------------------------------------------------- loading ---

def load_airfoil() -> pd.DataFrame:
    """The 1503 measurements, plus the one derived column we will argue about.

    `St` is the Strouhal number built from the displacement thickness,
    ``f * delta / U``, which is the dimensionless group the acoustics literature
    writes this problem in. It costs nothing to carry it here, and having it in
    the frame from the start means the notebook can compare a search that is told
    about it against one that is not, on identical rows.
    """
    if CSV.exists():
        df = pd.read_csv(CSV)
    else:
        raw = urllib.request.urlopen(UCI, timeout=30).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next(n for n in z.namelist() if n.endswith(".dat"))
            df = pd.read_csv(io.BytesIO(z.read(name)), sep="\t", names=COLUMNS)

    df["St"] = df["freq"] * df["delta"] / df["U"]
    return df


def split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 0):
    """A plain random split into fit rows and held-out rows.

    Deliberately plain. Holding out an entire wind speed or an entire chord is a
    more searching test and is worth doing once the room has something worth
    testing, but it drags every score down to somewhere in the 0.5s, and a lab
    whose first three numbers all look like failure is a bad lab. The random
    split is the one that makes progress legible.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    cut = int(round(len(df) * (1 - test_size)))
    return np.sort(order[:cut]), np.sort(order[cut:])


# ------------------------------------------------------------ feature views ---
#
# Three ways of handing the same 1503 rows to a search. The notebook walks down
# this list, and the only thing that changes between the runs is which of these
# came out of `view()`.

VIEWS = {
    "raw": (["freq", "aoa", "chord", "U", "delta"],
            "the five columns exactly as measured"),
    "log": (["log_f", "aoa", "log_c", "log_U", "log_d"],
            "the same five, with the ones that span decades put on a log scale"),
    "strouhal": (["log_St", "aoa", "log_c", "log_U", "log_d"],
                 "log frequency replaced by log Strouhal number"),
}


def view(df: pd.DataFrame, which: str):
    """Return ``(X, names)`` for one of the three feature views.

    Frequency, chord, velocity and thickness are all strictly positive and the
    first and last span more than a decade, so taking logs is not a trick, it is
    the scale the quantities actually live on. Decibels are already a log
    quantity, which is why the target is left alone.
    """
    if which not in VIEWS:
        raise ValueError(f"view must be one of {sorted(VIEWS)}, got {which!r}")

    cols = {
        "log_f": np.log10(df["freq"]),
        "log_c": np.log10(df["chord"]),
        "log_U": np.log10(df["U"]),
        "log_d": np.log10(df["delta"]),
        "log_St": np.log10(df["St"]),
        "aoa": df["aoa"],
    }
    names = VIEWS[which][0]
    X = np.column_stack([df[n] if n in df else cols[n] for n in names])
    return X.astype(float), list(names)


# ------------------------------------------------------------------ scoring ---

def r2(pred, true) -> float:
    """Coefficient of determination, with runaway predictions kept finite."""
    true = np.asarray(true, dtype=float)
    pred = np.nan_to_num(np.asarray(pred, dtype=float), nan=0.0,
                         posinf=1e12, neginf=-1e12)
    return float(1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2))


def explained_by(v, y, bins: int = 25) -> float:
    """How much of `y` one variable accounts for on its own, shape unspecified.

    Bins the rows by `v` and asks how much variance the bin means remove. Unlike a
    correlation this does not care whether the relationship is a straight line, so
    it answers the question we actually want to ask of a candidate variable, which
    is whether knowing it tells you anything at all about the answer.
    """
    y = pd.Series(np.asarray(y, dtype=float))
    binned = pd.qcut(pd.Series(np.asarray(v, dtype=float)), bins, duplicates="drop")
    means = y.groupby(binned, observed=True).transform("mean")
    return float(1 - ((y - means) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def front_scores(model, X, y) -> pd.DataFrame:
    """Every row of a Pareto front scored on rows the search never saw.

    `loss` in `model.equations_` is the training loss, so it falls monotonically
    with complexity and cannot tell you when an equation has started memorising.
    This re-scores each row against held-out data, which can and does go back
    down at the top of the front.
    """
    rows = []
    for i, row in model.equations_.reset_index(drop=True).iterrows():
        try:
            score = r2(model.predict(X, index=i), y)
        except Exception:                                  # a row that will not evaluate
            score = float("nan")
        rows.append({"complexity": int(row["complexity"]), "loss": float(row["loss"]),
                     "test_r2": score, "equation": row["equation"]})
    return pd.DataFrame(rows)


def best_on_front(front: pd.DataFrame):
    """The front row that actually scores best on held-out data.

    Not the same thing as `get_best()`, which never sees the test rows. Using this
    to *choose* would be cheating; using it to *report* is the honest way to say
    how far a given search got.
    """
    return front.loc[front["test_r2"].idxmax()]


def shortest_above(front: pd.DataFrame, bar: float):
    """The shortest equation on a front that clears `bar` on held-out rows.

    This is the question worth asking when comparing two searches: not "how good
    did it eventually get", which is mostly a statement about how long you left it
    running, but "how much equation did I have to buy to reach this accuracy".
    Returns None if nothing on the front gets there.
    """
    ok = front[front["test_r2"] >= bar]
    return None if ok.empty else ok.loc[ok["complexity"].idxmin()]


def price_of(fronts: dict, bar: float) -> pd.DataFrame:
    """`shortest_above` across several fronts at once, as a table to read out loud.

    `fronts` maps a label to a scored front. Rows that never clear the bar come
    back with a missing complexity rather than being dropped, because "this one
    never got there" is an answer.
    """
    rows = []
    for label, front in fronts.items():
        hit = shortest_above(front, bar)
        rows.append({"features": label,
                     "complexity needed": None if hit is None else int(hit["complexity"]),
                     "test_r2 there": None if hit is None else round(float(hit["test_r2"]), 3),
                     "best on front": round(float(front["test_r2"].max()), 3)})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- plots ---

def _finish(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def plot_overview(df: pd.DataFrame):
    """What the wind tunnel measured, one panel per knob.

    The point of looking at this before fitting anything is that no single panel
    shows a clean curve. Every one of them is a band, because the other four
    columns are varying underneath it.
    """
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 2.9), sharey=True)
    panels = [("freq", "frequency (Hz)", True), ("delta", "displacement thickness (m)", True),
              ("chord", "chord (m)", True), ("U", "free-stream U (m/s)", False),
              ("aoa", "angle of attack (deg)", False)]
    for ax, (col, label, logx) in zip(axes, panels):
        ax.scatter(df[col], df["spl"], s=7, color=BLUE, alpha=0.30, lw=0)
        if logx:
            ax.set_xscale("log")
        _finish(ax, xlabel=label)
    axes[0].set_ylabel("sound pressure level (dB)")
    axes[0].set_title("1503 wind-tunnel measurements, five knobs and one microphone",
                      color=INK, loc="left")
    return fig


def plot_collapse(df: pd.DataFrame):
    """Frequency against Strouhal number, the argument for the whole session.

    Same target, same rows, same kind of axis. On the left the level is smeared
    across two decades of frequency because a 2 kHz tone means something different
    to a 30 cm blade than to a 2.5 cm one. On the right the frequency has been
    made dimensionless with the boundary-layer thickness and the flow speed, and
    the band tightens into something with a shape.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, col, label in [(ax1, "freq", "frequency  $f$  (Hz)"),
                           (ax2, "St", "Strouhal number  $f\\,\\delta^*/U$")]:
        ax.scatter(df[col], df["spl"], s=9, color=BLUE, alpha=0.28, lw=0, zorder=2)
        x = np.log10(df[col])
        edges = np.quantile(x, np.linspace(0, 1, 22))
        mid = 0.5 * (edges[:-1] + edges[1:])
        means = [df["spl"][(x >= a) & (x <= b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(10 ** mid, means, color=ORANGE, lw=2.4, zorder=3)
        ax.set_xscale("log")
        share = explained_by(x, df["spl"])
        _finish(ax, title=f"{label.split('  ')[0]}  —  explains {share:.0%} on its own",
                xlabel=label)
    ax1.set_ylabel("sound pressure level (dB)")
    return fig


def plot_fronts(entries, baseline=None, title=None):
    """Held-out score against complexity, one line per search.

    `entries` is a list of ``(label, scored_front)`` pairs, where each front came
    out of `front_scores`. Reading it left to right answers the only question
    that matters when comparing two searches, which is how much equation you had
    to buy to reach a given accuracy.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    if baseline is not None:
        ax.axhline(baseline, color=MUTED, lw=1.3, ls=(0, (5, 3)), zorder=2)
        ax.annotate(f"boosted trees, {baseline:.3f}", (0.995, baseline), xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points", ha="right", va="bottom",
                    fontsize=9.5, color=MUTED)
    for i, (label, front) in enumerate(entries):
        colour = SERIES[i % len(SERIES)]
        ax.plot(front["complexity"], front["test_r2"], color=colour, marker="o",
                ms=4.5, lw=2.0, label=label, zorder=3)
    ax.set_ylim(-0.05, 1.02)
    ax.legend(loc="lower right")
    return _finish(ax, title=title or "How much equation you have to buy",
                   xlabel="complexity (nodes in the expression)",
                   ylabel="$R^2$ on held-out rows")


def plot_parity(entries, y_true, title=None):
    """Predicted against measured on the held-out rows, one panel per model.

    `entries` is a list of ``(label, predictions)``. The diagonal is where a
    perfect model would sit, and the useful thing to look for is not the spread
    but its shape, since a model that is merely noisy looks quite different from
    one that has flattened the ends of the range.
    """
    fig, axes = plt.subplots(1, len(entries), figsize=(3.5 * len(entries), 3.6),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = np.min(y_true) - 2, np.max(y_true) + 2
    for i, (ax, (label, pred)) in enumerate(zip(axes, entries)):
        pred = np.asarray(pred, dtype=float)
        # Scored before clipping and drawn after, so the number in the title is the
        # real one and a runaway point still lands on the edge of the frame instead
        # of stretching the axes past the range anyone can read.
        score = r2(pred, y_true)
        drawn = np.clip(np.nan_to_num(pred, nan=lo), lo, hi)
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=2)
        ax.scatter(y_true, drawn, s=10, color=SERIES[i % len(SERIES)], alpha=0.35, lw=0, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        _finish(ax, title=f"{label}   $R^2$ {score:.3f}", xlabel="measured (dB)")
    axes[0].set_ylabel("predicted (dB)")
    if title:
        fig.suptitle(title, color=INK, x=0.005, ha="left", fontsize=12)
    return fig


def scoreboard(rows) -> pd.DataFrame:
    """The running tally, sorted by held-out score.

    `rows` is a list of dicts with at least `name` and `test_r2`; anything else
    you put in comes along. This exists so the open bench at the end of the
    session has somewhere to put its answers.
    """
    table = pd.DataFrame(list(rows))
    return table.sort_values("test_r2", ascending=False).reset_index(drop=True)
