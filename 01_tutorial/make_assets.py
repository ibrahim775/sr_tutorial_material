from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"

BLUE, ORANGE = "#2f6fb5", "#e1701a"


P1_SEED = 0          # numpy seed for the oscillator data
P1_JULIA_SEED = 3    # PySR random_state for the messy run
P6_SEED = 6          # numpy + torch seed for the Problem 6 network

P1_SETTINGS = dict(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp"],
    maxsize=30,
    niterations=100,
)


def oscillator_data(n: int = 200):
    """y = A x exp(-x/tau) cos(omega x) + noise.  Identical code in the notebook."""
    g = np.random.default_rng(P1_SEED)
    x = np.sort(g.uniform(0, 10, n))
    y = 2.8 * x * np.exp(-x / 2.0) * np.cos(2.5 * x) + g.normal(0, 0.05, n)
    return x, y


def force_law_data(n: int, seed: int):
    """F(r, s) = 1/r^2 + 0.3 s.  Identical code in the notebook."""
    g = np.random.default_rng(seed)
    r = g.uniform(0.5, 2.0, n)
    s = g.uniform(-1.0, 1.0, n)
    X = np.column_stack([r, s]).astype(np.float32)
    y = (1.0 / r**2 + 0.3 * s).astype(np.float32).reshape(-1, 1)
    return X, y


def has_nested_trig(equation: str) -> bool:
    """True if a sin/cos appears anywhere inside another sin/cos.  Also in the notebook."""
    for match in re.finditer(r"(sin|cos)\(", equation):
        depth, i = 1, match.end()
        while i < len(equation) and depth > 0:
            depth += {"(": 1, ")": -1}.get(equation[i], 0)
            i += 1
        if "sin(" in equation[match.end():i - 1] or "cos(" in equation[match.end():i - 1]:
            return True
    return False


# --------------------------------------------------------------------------------------
# artifact 1: Problem 1's messy Pareto front
# --------------------------------------------------------------------------------------
def build_problem1() -> None:
    import shutil
    import tempfile

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pysr import PySRRegressor

    x, y = oscillator_data()
    scratch = tempfile.mkdtemp(prefix="pysr_assets_")

    print("fitting the unconstrained oscillator search (serial + deterministic)...")
    t0 = time.time()
    model = PySRRegressor(
        **P1_SETTINGS,
        deterministic=True,
        parallelism="serial",
        random_state=P1_JULIA_SEED,
        output_directory=scratch,
        run_id="problem1_messy",
        verbosity=0,
        progress=False,
    )
    model.fit(x.reshape(-1, 1), y, variable_names=["x"])
    elapsed = time.time() - t0
    shutil.rmtree(scratch, ignore_errors=True)

    front = model.equations_[["complexity", "loss", "score", "equation"]].copy()
    front.to_csv(ASSETS / "problem1_messy_front.csv", index=False)

    flagged = front["equation"].map(has_nested_trig)
    meta = {
        "what": "Unconstrained PySR front for y = 2 exp(-x/3) cos(2.5 x) + N(0, 0.05).",
        "why_precomputed": (
            "The junk on this front is stochastic and the run costs about "
            f"{elapsed:.0f} s. Fixed here so the notebook's before/after is exact."
        ),
        "pysr_settings": P1_SETTINGS,
        "reproducibility": {
            "deterministic": True,
            "parallelism": "serial",
            "random_state": P1_JULIA_SEED,
            "numpy_seed": P1_SEED,
        },
        "n_rows": int(len(front)),
        "n_rows_with_nested_trig": int(flagged.sum()),
        "best_loss": float(front["loss"].min()),
        "wall_seconds": round(elapsed, 1),
    }
    (ASSETS / "problem1_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.plot(front["complexity"], front["loss"], "-", color="#9aa4b2", lw=2, zorder=1)
    ok = ~flagged
    ax.plot(front.loc[ok, "complexity"], front.loc[ok, "loss"], "o", ms=8,
            color=BLUE, mec="white", mew=2, label="plain", zorder=3)
    ax.plot(front.loc[flagged, "complexity"], front.loc[flagged, "loss"], "X", ms=11,
            color=ORANGE, mec="white", mew=2, label="trig inside trig", zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("complexity")
    ax.set_ylabel("loss (MSE)")
    ax.set_title("Unconstrained search: what fills the Pareto front", fontsize=11)
    ax.grid(True, which="major", color="#e6e8ec", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "problem1_messy_front.png")
    plt.close(fig)

    print(f"  wrote problem1_messy_front.csv  ({len(front)} rows, "
          f"{int(flagged.sum())} with nested trig, {elapsed:.0f} s)")


# --------------------------------------------------------------------------------------
# artifact 2: Problem 6's fallback MLP checkpoint
# --------------------------------------------------------------------------------------
def build_problem6() -> None:
    import torch
    import torch.nn as nn

    class ForceNet(nn.Module):
        """Same definition as the notebook's, so the state_dict keys line up."""

        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                                     nn.Linear(32, 32), nn.ReLU(),
                                     nn.Linear(32, 1))

        def forward(self, x):
            return self.mlp(x)

    X_train, y_train = force_law_data(400, seed=P6_SEED)
    X_test, y_test = force_law_data(400, seed=P6_SEED + 1)

    torch.manual_seed(P6_SEED)
    net = ForceNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    Xt, yt = torch.tensor(X_train), torch.tensor(y_train)

    print("training the fallback MLP...")
    t0 = time.time()
    for _ in range(400):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(Xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        test_mse = nn.functional.mse_loss(net(torch.tensor(X_test)),
                                          torch.tensor(y_test)).item()

    torch.save(
        {
            "state_dict": net.state_dict(),
            "architecture": "ForceNet: mlp = Sequential(Linear(2,32), ReLU, Linear(32,32), ReLU, Linear(32,1))",
            "target": "F(r, s) = 1/r**2 + 0.3*s",
            "train_seed": P6_SEED,
            "epochs": 400,
            "optimizer": "Adam(lr=1e-2)",
            "train_mse": float(loss.item()),
            "test_mse": float(test_mse),
            "torch_version": torch.__version__,
        },
        ASSETS / "problem6_mlp.pt",
    )
    print(f"  wrote problem6_mlp.pt  (train mse {loss.item():.3e}, "
          f"test mse {test_mse:.3e}, {time.time() - t0:.1f} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["p1", "p6"], default=None)
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "p1"):
        build_problem1()
    if args.only in (None, "p6"):
        build_problem6()
    print(f"assets in {ASSETS}")


if __name__ == "__main__":
    main()
