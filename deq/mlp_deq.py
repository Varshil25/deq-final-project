"""Fully connected deep equilibrium model with hand-derived gradients.

Every derivative in this file is written out by hand. No automatic
differentiation is used anywhere — that is the point of the assignment track.

Notation (row-vector / batch-first convention):

    x in R^{B x d_in}      input
    z in R^{B x d}         hidden state
    W in R^{d x d}         recurrent weight
    U in R^{d x d_in}      input injection
    b in R^{d}             bias
    V in R^{C x d}, c in R^{C}   output head

    a      = z W^T + x U^T + b            pre-activation
    f(z,x) = ReLU(a)                      equilibrium layer
    z*     = f(z*, x)                     fixed point
    logits = z* V^T + c

Jacobian of the layer with respect to z, for one batch element:

    d f_i / d z_j = 1[a_i > 0] * W_ij

so the vector-Jacobian product of a row vector u is

    (u J)_j = sum_i u_i 1[a_i > 0] W_ij = ((u * mask) W)_j

which is `vjp_z` below. The same masked vector (u * mask) is what the
parameter gradients contract against, which is why `vjp_params` takes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solvers import SOLVERS, SolverResult


@dataclass
class DEQParams:
    W: np.ndarray
    U: np.ndarray
    b: np.ndarray
    V: np.ndarray
    c: np.ndarray

    def copy(self) -> "DEQParams":
        return DEQParams(*(a.copy() for a in self.as_tuple()))

    def as_tuple(self):
        return (self.W, self.U, self.b, self.V, self.c)

    @property
    def names(self):
        return ("W", "U", "b", "V", "c")


def init_params(d_in, d_hidden, n_classes, seed=0, spectral_scale=0.9) -> DEQParams:
    """Initialise, rescaling W so the layer starts as a contraction.

    If the spectral norm of W exceeds 1 the fixed-point iteration is not
    guaranteed to converge, and in practice the solver stalls on the very
    first batch. Rescaling to `spectral_scale` < 1 avoids that.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0 / np.sqrt(d_hidden), (d_hidden, d_hidden))
    W *= spectral_scale / np.linalg.norm(W, ord=2)
    U = rng.normal(0, 1.0 / np.sqrt(d_in), (d_hidden, d_in))
    b = np.zeros(d_hidden)
    V = rng.normal(0, 1.0 / np.sqrt(d_hidden), (n_classes, d_hidden))
    c = np.zeros(n_classes)
    return DEQParams(W, U, b, V, c)


# --------------------------------------------------------------------------
# the equilibrium layer
# --------------------------------------------------------------------------

def layer(z, x, p: DEQParams):
    """f(z, x) = ReLU(z W^T + x U^T + b)."""
    a = z @ p.W.T + x @ p.U.T + p.b
    return np.maximum(a, 0.0), a


def solve_forward(x, p: DEQParams, solver="anderson", **kw) -> SolverResult:
    """Find z* = f(z*, x), starting from zero."""
    z0 = np.zeros((x.shape[0], p.W.shape[0]))
    inj = x @ p.U.T + p.b                      # constant across iterations
    f = lambda z: np.maximum(z @ p.W.T + inj, 0.0)
    return SOLVERS[solver](f, z0, **kw)


def vjp_z(u, mask, p: DEQParams):
    """u J, where J = df/dz evaluated at the fixed point."""
    return (u * mask) @ p.W


def vjp_params(u, mask, z_star, x):
    """Gradients of the layer's parameters given the adjoint u.

    dL/dW = (u * mask)^T z*,   dL/dU = (u * mask)^T x,   dL/db = sum(u * mask)
    """
    um = u * mask
    return um.T @ z_star, um.T @ x, um.sum(axis=0)


# --------------------------------------------------------------------------
# loss and output head
# --------------------------------------------------------------------------

def softmax_cross_entropy(logits, y):
    """Mean cross-entropy, and dL/dlogits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1.0
    return loss, dlogits / n


# --------------------------------------------------------------------------
# forward + backward
# --------------------------------------------------------------------------

def forward(x, y, p: DEQParams, solver="anderson", **solver_kw):
    """Returns (loss, cache) where cache holds what the backward pass needs."""
    res = solve_forward(x, p, solver=solver, **solver_kw)
    z_star = res.z
    _, a = layer(z_star, x, p)
    mask = (a > 0).astype(z_star.dtype)

    logits = z_star @ p.V.T + p.c
    loss, dlogits = softmax_cross_entropy(logits, y)

    cache = dict(x=x, z_star=z_star, mask=mask, dlogits=dlogits, solver_result=res)
    return loss, cache


def backward(cache, p: DEQParams, solver="anderson", mode="exact", neumann_k=5, **solver_kw):
    """Implicit gradient.

    The adjoint u solves the linear fixed point

        u = u J + g,    g = dL/dz*

    `mode` selects how (I - J)^{-1} is applied:
      exact        — solve the fixed point to tolerance
      neumann      — truncate the series at `neumann_k` terms
      jacobian_free— take (I - J)^{-1} ~ I, i.e. u = g
    """
    x, z_star, mask = cache["x"], cache["z_star"], cache["mask"]
    dlogits = cache["dlogits"]

    # output head
    dV = dlogits.T @ z_star
    dc = dlogits.sum(axis=0)
    g = dlogits @ p.V                     # dL/dz*

    if mode == "exact":
        bwd = SOLVERS[solver](lambda u: vjp_z(u, mask, p) + g, np.zeros_like(g), **solver_kw)
        u, n_bwd_iter = bwd.z, bwd.n_iter
    elif mode == "neumann":
        u, term = g.copy(), g.copy()
        for _ in range(neumann_k):
            term = vjp_z(term, mask, p)
            u = u + term
        n_bwd_iter = neumann_k
    elif mode == "jacobian_free":
        u, n_bwd_iter = g, 0
    else:
        raise ValueError(f"unknown mode {mode!r}")

    dW, dU, db = vjp_params(u, mask, z_star, x)
    grads = DEQParams(dW, dU, db, dV, dc)
    return grads, n_bwd_iter


def predict(x, p: DEQParams, solver="anderson", **solver_kw):
    res = solve_forward(x, p, solver=solver, **solver_kw)
    return (res.z @ p.V.T + p.c).argmax(axis=1), res
