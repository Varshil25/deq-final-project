"""Phase 1: train the hand-written NumPy DEQ on MNIST.

Every gradient here comes from `deq.backward` — no autograd. Results are
written to results/mnist_<solver>.json so the report tables can be generated
without rerunning anything.

    python experiments/train_mnist_numpy.py --solver anderson --epochs 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deq.mlp_deq import DEQParams, backward, forward, init_params, predict  # noqa: E402
from scripts.get_mnist import load  # noqa: E402


class Adam:
    """Plain Adam, written out so the whole pipeline stays dependency-free."""

    def __init__(self, params: DEQParams, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {n: np.zeros_like(getattr(params, n)) for n in params.names}
        self.v = {n: np.zeros_like(getattr(params, n)) for n in params.names}
        self.t = 0

    def step(self, grads: DEQParams):
        self.t += 1
        for n in self.p.names:
            g = getattr(grads, n)
            self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * g
            self.v[n] = self.b2 * self.v[n] + (1 - self.b2) * g * g
            mhat = self.m[n] / (1 - self.b1**self.t)
            vhat = self.v[n] / (1 - self.b2**self.t)
            getattr(self.p, n)[...] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def evaluate(x, y, p, batch=1000, **kw):
    correct, iters = 0, []
    for i in range(0, len(x), batch):
        pred, res = predict(x[i : i + batch], p, **kw)
        correct += int((pred == y[i : i + batch]).sum())
        iters.append(res.n_iter)
    return correct / len(x), float(np.mean(iters))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="anderson", choices=["anderson", "fixed_point"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--train-size", type=int, default=0, help="0 = all 60k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--lam-scaling",
        default="relative",
        choices=["relative", "absolute"],
        help="Anderson Tikhonov term; 'absolute' reproduces the stalling bug",
    )
    ap.add_argument("--tag", default="", help="suffix for the results filename")
    args = ap.parse_args()

    d = load()
    xtr = d["train_x"].astype(np.float64) / 255.0
    ytr = d["train_y"].astype(np.int64)
    xte = d["test_x"].astype(np.float64) / 255.0
    yte = d["test_y"].astype(np.int64)
    if args.train_size:
        xtr, ytr = xtr[: args.train_size], ytr[: args.train_size]

    p = init_params(784, args.hidden, 10, seed=args.seed)
    opt = Adam(p, lr=args.lr)
    kw = dict(solver=args.solver, tol=args.tol, max_iter=args.max_iter)
    if args.solver == "anderson":
        kw["lam_scaling"] = args.lam_scaling

    rng = np.random.default_rng(args.seed)
    history = []

    for epoch in range(args.epochs):
        order = rng.permutation(len(xtr))
        t0 = time.time()
        losses, fwd_iters, bwd_iters = [], [], []

        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            loss, cache = forward(xtr[idx], ytr[idx], p, **kw)
            grads, n_bwd = backward(cache, p, mode="exact", **kw)
            opt.step(grads)

            losses.append(loss)
            fwd_iters.append(cache["solver_result"].n_iter)
            bwd_iters.append(n_bwd)

        epoch_time = time.time() - t0
        test_acc, test_iters = evaluate(xte, yte, p, **kw)
        rec = dict(
            epoch=epoch + 1,
            train_loss=float(np.mean(losses)),
            test_acc=test_acc,
            mean_fwd_iters=float(np.mean(fwd_iters)),
            mean_bwd_iters=float(np.mean(bwd_iters)),
            test_fwd_iters=test_iters,
            epoch_time_s=epoch_time,
        )
        history.append(rec)
        print(
            f"epoch {rec['epoch']}  loss {rec['train_loss']:.4f}  "
            f"test acc {100*rec['test_acc']:.2f}%  "
            f"fwd {rec['mean_fwd_iters']:.1f}  bwd {rec['mean_bwd_iters']:.1f}  "
            f"{rec['epoch_time_s']:.1f}s"
        )

    out = ROOT / "results" / f"mnist_{args.solver}{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(config=vars(args), history=history), indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
