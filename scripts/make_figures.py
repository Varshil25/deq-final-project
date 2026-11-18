"""Generate the figures used in the reports, from the committed result files.

    python scripts/make_figures.py

Writes vector PDFs into reports/figures/ so LaTeX embeds them without
rasterising. Panel (a) needs residual traces, which the training runs do not
save, so it trains briefly and measures them here; everything in panel (b)
comes straight from results/*.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deq.mlp_deq import backward, forward, init_params, solve_forward  # noqa: E402
from experiments.train_mnist_numpy import Adam  # noqa: E402
from scripts.get_mnist import load  # noqa: E402

FIGDIR = ROOT / "reports" / "figures"

# Colourblind-safe; each series also carries its own marker and dash pattern so
# the figure survives greyscale printing.
STYLE = {
    "fixed_point": dict(color="#0173B2", marker="o", ls="-", label="Fixed-point"),
    "anderson":    dict(color="#029E73", marker="s", ls="--", label="Anderson (scaled $\\lambda$)"),
    "absolute":    dict(color="#DE8F05", marker="^", ls=":", label="Anderson (fixed $\\lambda$)"),
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 3.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def residual_traces(epochs=1, train_size=10000, seed=0):
    """Train briefly, then record the residual per iteration on one batch."""
    d = load()
    xtr = d["train_x"][:train_size].astype(np.float64) / 255.0
    ytr = d["train_y"][:train_size].astype(np.int64)

    p = init_params(784, 128, 10, seed=seed)
    opt = Adam(p, lr=1e-3)
    kw = dict(solver="anderson", tol=1e-3, max_iter=50)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in range(0, train_size, 128):
            idx = rng.permutation(train_size)[i : i + 128]
            _, cache = forward(xtr[idx], ytr[idx], p, **kw)
            grads, _ = backward(cache, p, mode="exact", **kw)
            opt.step(grads)

    batch = xtr[:128]
    out = {}
    for name in ("fixed_point", "anderson"):
        # tol=0 so the solver runs the full budget and the whole curve is seen
        res = solve_forward(batch, p, solver=name, tol=0.0, max_iter=30)
        out[name] = res.residuals
    return out


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 1.95))

    # (a) forward-solver convergence
    traces = residual_traces()
    for name in ("fixed_point", "anderson"):
        r = traces[name]
        s = STYLE[name]
        ax1.semilogy(range(1, len(r) + 1), r, **s, markevery=3)
    ax1.axhline(1e-3, color="0.45", lw=0.8, ls="-.")
    ax1.text(1, 1.3e-3, "tolerance", ha="left", va="bottom", fontsize=6, color="0.35")
    ax1.set_xlabel("Forward solver iteration")
    ax1.set_ylabel("Relative residual")
    ax1.set_title("(a) Forward convergence", loc="left", fontweight="bold")
    ax1.annotate("Fixed-point", xy=(24, traces["fixed_point"][23]),
                 xytext=(8, 14), textcoords="offset points",
                 color=STYLE["fixed_point"]["color"], fontsize=7)
    ax1.annotate("Anderson", xy=(21, traces["anderson"][20]),
                 xytext=(-2, -16), textcoords="offset points",
                 color=STYLE["anderson"]["color"], fontsize=7, ha="center")
    ax1.set_xlim(0, 36)

    # (b) backward iterations per epoch, from the committed runs
    files = {
        "fixed_point": "mnist_fixed_point.json",
        "anderson": "mnist_anderson.json",
        "absolute": "mnist_anderson_absolute_lam.json",
    }
    for key, fname in files.items():
        h = json.load(open(ROOT / "results" / fname))["history"]
        ax2.plot([r["epoch"] for r in h], [r["mean_bwd_iters"] for r in h], **STYLE[key])
    ax2.axhline(50, color="0.45", lw=0.8, ls="-.")
    ax2.text(5, 51, "iteration cap", ha="right", va="bottom", fontsize=6, color="0.35")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean backward iterations")
    ax2.set_title("(b) Backward cost", loc="left", fontweight="bold")
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_ylim(0, 58)
    ax2.legend(frameon=False, loc="center left", handlelength=1.8,
               borderaxespad=0.3, labelspacing=0.3)

    fig.tight_layout(pad=0.4)
    out = FIGDIR / "solvers.pdf"
    fig.savefig(out)
    fig.savefig(FIGDIR / "solvers.png", dpi=200)
    print(f"written {out}")

    f = traces["fixed_point"]; a = traces["anderson"]
    hit = lambda r: next((i + 1 for i, v in enumerate(r) if v < 1e-3), None)
    print(f"iterations to reach 1e-3 -- fixed-point: {hit(f)}, anderson: {hit(a)}")


if __name__ == "__main__":
    main()
