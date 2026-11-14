"""Central-difference check of the hand-derived gradients.

This is the correctness evidence for Progress Report I: if these errors are
small, the implicit gradient in `backward` is right. Run with

    python -m deq.gradcheck
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mlp_deq import DEQParams, backward, forward, init_params


def numerical_grad(p: DEQParams, name, x, y, eps=1e-6, solver_kw=None):
    """dL/d(param) by central differences, one entry at a time."""
    solver_kw = solver_kw or {}
    arr = getattr(p, name)
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = arr[idx]

        arr[idx] = orig + eps
        loss_plus, _ = forward(x, y, p, **solver_kw)
        arr[idx] = orig - eps
        loss_minus, _ = forward(x, y, p, **solver_kw)
        arr[idx] = orig

        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(ana, num, floor=1e-8):
    """Max of |a - n| / max(|a|, |n|) over entries that are not both ~zero.

    With a ReLU layer most units are inactive on any given batch, so the
    majority of entries have a true gradient of order 1e-10. Dividing the
    difference of two such entries by their own magnitude reports a relative
    error near 1 while the absolute error is 1e-16, which says nothing about
    correctness. Entries below `floor` are therefore judged on absolute error
    only, which is the usual convention for a gradient check.
    """
    scale = np.maximum(np.abs(ana), np.abs(num))
    mask = scale > floor
    if not mask.any():
        return 0.0, 0
    return float(np.max(np.abs(ana - num)[mask] / scale[mask])), int(mask.sum())


def run_check(d_in=6, d_hidden=5, n_classes=3, batch=4, eps=1e-6, seed=0, tol=1e-10):
    """Tight solver tolerance: the gradient can only be as exact as the solve."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, d_in))
    y = rng.integers(0, n_classes, size=batch)
    p = init_params(d_in, d_hidden, n_classes, seed=seed)

    solver_kw = dict(solver="anderson", tol=tol, max_iter=500)

    _, cache = forward(x, y, p, **solver_kw)
    analytic, _ = backward(cache, p, mode="exact", **solver_kw)

    rows = []
    for name in p.names:
        num = numerical_grad(p, name, x, y, eps=eps, solver_kw=solver_kw)
        ana = getattr(analytic, name)
        rel_err, n_checked = relative_error(ana, num)
        rows.append(
            dict(
                param=name,
                max_abs_err=float(np.max(np.abs(ana - num))),
                max_rel_err=rel_err,
                n_entries=int(ana.size),
                n_nonzero=n_checked,
            )
        )
    return rows


def main():
    rows = run_check()
    width = max(len(r["param"]) for r in rows)
    print(f"{'param':<{width}}  {'max abs err':>12}  {'max rel err':>12}  {'entries':>12}")
    for r in rows:
        checked = f"{r['n_nonzero']}/{r['n_entries']}"
        print(
            f"{r['param']:<{width}}  {r['max_abs_err']:12.3e}  "
            f"{r['max_rel_err']:12.3e}  {checked:>12}"
        )

    worst = max(r["max_rel_err"] for r in rows)
    print(f"\nworst relative error: {worst:.3e}", "PASS" if worst < 1e-5 else "FAIL")

    out = Path(__file__).resolve().parent.parent / "results" / "gradcheck.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"written to {out}")
    return 0 if worst < 1e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
