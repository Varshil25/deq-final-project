# End-to-end plan

Project: implementing deep equilibrium models (DEQ) from scratch — Efficient
Implementation track, Direction 2.

Three submissions, three phases. Each phase ends with a report built from the
shared LaTeX template in `reports/`, so every submission looks identical.

| Deadline | Submission | Points | Phase |
| --- | --- | --- | --- |
| Nov 17, 11:59pm | Progress Report I | 15 | Phase 1 — NumPy DEQ, MNIST |
| Dec 1, 11:59pm | Progress Report II | 15 | Phase 2 — PyTorch conv DEQ, CIFAR-10 |
| Dec 8, 11:59pm | Final Report | 50 | Phase 3 — ablations and write-up |

**Note:** HW3&4 (20 pts) is also due Dec 1, the same night as Progress Report
II. Phase 2 is planned to finish its experiments by **Nov 28** so the last
three days are free for the homework.

---

## Phase 1 — NumPy DEQ (Nov 14 → Nov 17)

The goal of this phase is *correctness*, not accuracy. A DEQ whose gradients
provably match finite differences is the foundation everything else rests on.

### Build

| Module | What it does |
| --- | --- |
| `deq/solvers.py` | `fixed_point_solve`, `anderson_solve` — forward fixed-point solvers, returning the solution plus residual history |
| `deq/mlp_deq.py` | `f(z,x) = ReLU(Wz + Ux + b)`, its forward pass, and its hand-derived Jacobian-vector products |
| `deq/backward.py` | Exact implicit gradient: solve `u = uJ + dl/dz*`, then apply to `df/dtheta` |
| `deq/gradcheck.py` | Central-difference check of every parameter gradient |
| `experiments/train_mnist_numpy.py` | SGD/Adam training loop, logs accuracy and solver iteration counts |

No autograd anywhere in Phase 1 — every derivative is written out by hand.
That is the part of the assignment being graded on this track.

### Checks that must pass before the report

1. Max relative gradient error vs central differences below `1e-5` for `W`,
   `U`, `b` and the output layer.
2. Anderson reaches a residual of `1e-3` in fewer iterations than plain
   fixed-point iteration on the same batch.
3. MNIST test accuracy above 95% — enough to show the model trains.

### Report I contents (2 pages)

Summary · what was built (layer, both solvers, hand-derived backward pass) ·
gradient-check table · MNIST accuracy and solver-iteration table · problems hit
· plan for Phase 2.

### Day by day

- **Nov 14** — solvers and forward pass, residual plots working.
- **Nov 15** — backward pass derived on paper, then coded; gradient check passing.
- **Nov 16** — MNIST training runs, both solvers, results saved to `results/`.
- **Nov 17** — fill `reports/report1.tex`, build, submit.

---

## Phase 2 — PyTorch convolutional DEQ (Nov 18 → Dec 1)

This phase answers the project's actual research question: what does replacing
the exact implicit gradient with a cheap approximation cost?

### Build

| Module | What it does |
| --- | --- |
| `deq/torch_layer.py` | Conv equilibrium layer: two 3×3 convs, group norm, input injection |
| `deq/torch_deq.py` | Custom `torch.autograd.Function` — solver in forward, implicit gradient in backward |
| `deq/gradients.py` | Three backward variants: exact, truncated Neumann (k=1…5), Jacobian-free (k=0) |
| `deq/unrolled.py` | Weight-tied baseline applying the layer K times, ordinary backprop |
| `experiments/train_cifar.py` | Training loop with memory, timing and iteration-count instrumentation |
| `experiments/grad_similarity.py` | Cosine similarity of each approximate gradient against the exact one |

Autograd is used here only for the vector-Jacobian product of a *single* layer
evaluation. The solver and the implicit gradient are still mine.

### Experiments

- Four methods × CIFAR-10, same architecture, same budget: unrolled, exact,
  Neumann (k=1 and k=5), Jacobian-free.
- Record per method: test accuracy, peak memory, time per epoch, forward and
  backward iteration counts, gradient cosine similarity.

### Report II contents (2 pages)

Summary leading with the accuracy/time trade-off · what was added since Report
I · the comparison table · forward-iteration-count figure · problems · plan for
Phase 3.

### Week by week

- **Nov 18–21** — conv layer and custom autograd Function; port the gradient
  check to PyTorch and confirm the custom backward matches autograd on a tiny
  unrolled case.
- **Nov 22–25** — the three gradient variants and the unrolled baseline.
- **Nov 26–28** — CIFAR-10 runs for all four methods; results to `results/`.
- **Nov 29–30** — write Report II. **Leave Dec 1 for HW3&4.**

---

## Phase 3 — Ablations and final write-up (Dec 2 → Dec 8)

Phase 3 adds nothing new to the model. It makes the existing result solid and
writes it up properly.

### Experiments

- **Solver tolerance:** 1e-2, 1e-3, 1e-4 — how exact does the forward solve
  need to be?
- **Anderson memory:** m = 2, 5, 10.
- **Neumann order:** k = 0 (Jacobian-free) through 5, as one sweep — the
  headline figure of the report.
- **Stability:** forward iteration count over training; add Jacobian
  regularization if it grows and show the before/after.

### Report contents (5 pages)

Introduction · related works · methodology (equilibrium formulation, Anderson
pseudocode, hand-derived backward pass, the three approximations) ·
experimental results (setup, gradient correctness, main comparison table,
ablations) · conclusions.

### Day by day

- **Dec 2–4** — run all ablations, save every result as JSON.
- **Dec 5–6** — generate final figures and tables from the JSON.
- **Dec 7** — write the 5 pages.
- **Dec 8** — proofread, check page count, verify every citation, submit.

---

## Standing rules

- **Results are files, not notebook output.** Every run writes a JSON file to
  `results/` with its config and numbers. Tables and figures are generated from
  those files, so a rerun never means retyping a table.
- **Verify the references before each submission.** The entries in
  `reports/refs.bib` were written from memory. Confirm authors, venue and year
  against the real papers.
- **Check the page count before submitting:** `make check` in `reports/` prints
  it for every document. Limits are 2, 2 and 5 pages at 11 pt.
- **Commit after every working change**, so a broken experiment is one `git
  checkout` away from the last good state.
