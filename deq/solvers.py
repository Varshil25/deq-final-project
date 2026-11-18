"""Fixed-point solvers for z* = f(z*).

Both solvers take a callable f mapping an array of shape (B, d) to the same
shape, and return the solution together with the residual history so that
convergence can be plotted and iteration counts logged.

Nothing here is specific to the forward pass: the backward pass solves
u = u J + g, which is also a fixed point, so it reuses the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SolverResult:
    z: np.ndarray
    n_iter: int
    residuals: list[float] = field(default_factory=list)

    @property
    def final_residual(self) -> float:
        return self.residuals[-1] if self.residuals else float("nan")


def _rel_residual(f_z: np.ndarray, z: np.ndarray) -> float:
    """Relative residual ||f(z) - z|| / ||f(z)||, averaged over the batch."""
    num = np.linalg.norm(f_z - z, axis=1)
    den = np.linalg.norm(f_z, axis=1) + 1e-8
    return float(np.mean(num / den))


def fixed_point_solve(f, z0, max_iter=50, tol=1e-3):
    """Plain iteration z_{k+1} = f(z_k). The baseline solver."""
    z = z0
    residuals: list[float] = []
    for k in range(max_iter):
        f_z = f(z)
        res = _rel_residual(f_z, z)
        residuals.append(res)
        z = f_z
        if res < tol:
            return SolverResult(z, k + 1, residuals)
    return SolverResult(z, max_iter, residuals)


def anderson_solve(f, z0, m=5, lam=1e-4, beta=1.0, max_iter=50, tol=1e-3,
                   lam_scaling="relative"):
    """Anderson acceleration (Anderson, 1965).

    Keeps the last m iterates and residuals, and takes the step that minimises
    the residual over their affine hull. The least-squares problem is solved
    per batch element, which is why the history arrays carry a batch axis.

    `lam_scaling="absolute"` restores the naive Tikhonov term that stalls the
    backward solve; it exists so that behaviour can be reproduced on demand
    rather than only described. See the comment in the loop.
    """
    bsz, d = z0.shape
    m = min(m, max_iter)

    X = np.zeros((bsz, m, d), dtype=z0.dtype)  # iterates
    F = np.zeros((bsz, m, d), dtype=z0.dtype)  # f(iterates)

    X[:, 0], F[:, 0] = z0, f(z0)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0])

    # Bordered system: [[0, 1^T], [1, G G^T + lam I]] [[beta],[alpha]] = [[1],[0]]
    H = np.zeros((bsz, m + 1, m + 1), dtype=z0.dtype)
    H[:, 0, 1:] = 1.0
    H[:, 1:, 0] = 1.0
    y = np.zeros((bsz, m + 1, 1), dtype=z0.dtype)
    y[:, 0] = 1.0

    residuals = [_rel_residual(F[:, 0], X[:, 0]), _rel_residual(F[:, 1], X[:, 1])]

    k = 1
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]                      # (B, n, d)
        GGt = G @ G.transpose(0, 2, 1)

        # The Tikhonov term must scale with the problem. With a fixed absolute
        # lam, once the residuals become small (which happens in the backward
        # solve, where most examples contribute a near-zero gradient late in
        # training) lam dominates the normal equations, alpha collapses towards
        # uniform averaging and the solver stalls at the iteration cap.
        if lam_scaling == "relative":
            scale = np.einsum("bii->b", GGt)[:, None, None] / n + 1e-30
        elif lam_scaling == "absolute":
            scale = 1.0
        else:
            raise ValueError(f"unknown lam_scaling {lam_scaling!r}")
        H[:, 1 : n + 1, 1 : n + 1] = GGt + lam * scale * np.eye(n, dtype=z0.dtype)[None]

        alpha = np.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[:, 1 : n + 1, 0]

        idx = k % m
        X[:, idx] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (
            alpha[:, None] @ X[:, :n]
        )[:, 0]
        F[:, idx] = f(X[:, idx])

        res = _rel_residual(F[:, idx], X[:, idx])
        residuals.append(res)
        if res < tol:
            return SolverResult(X[:, idx], k + 1, residuals)

    return SolverResult(X[:, k % m], max_iter, residuals)


SOLVERS = {"fixed_point": fixed_point_solve, "anderson": anderson_solve}
