# Deep Equilibrium Models from Scratch

Final project — Deep Learning. A deep equilibrium model (DEQ) implemented by
hand: the fixed-point forward solver and the implicit-differentiation backward
pass are written from scratch, with no automatic differentiation run through
the solver.

**Varshil Patel · TUID 916646801 · Instructor: Hongchang Gao**

> Assignment track: *Efficient Implementation, Direction 2* — implement an
> advanced architecture, including its backpropagation, yourself.

## Status

| Phase | Deliverable | Due | State |
| --- | --- | --- | --- |
| — | Proposal | Nov 10 | Submitted |
| 1 | Progress Report I — NumPy DEQ, MNIST | Nov 17 | **Complete — submitted** |
| 2 | Progress Report II — PyTorch conv DEQ, CIFAR-10 | Dec 1 | Not started |
| 3 | Final Report — ablations and write-up | Dec 8 | Not started |

See [`PLAN.md`](PLAN.md) for the full three-phase plan.

### Phase 1 results

- Hand-derived gradients match central differences to **1.3e-06** worst
  relative error (`results/gradcheck.json`).
- MNIST, 5 epochs, hidden width 128: **97.5%** test accuracy.
- Anderson acceleration converges in ~30% fewer iterations than plain
  fixed-point iteration, but costs ~2x the wall-clock time at this scale.

## Layout

```
deq/            the model: solvers, layer, hand-derived gradients
  solvers.py      fixed_point_solve, anderson_solve (used by both passes)
  mlp_deq.py      the equilibrium layer, forward, and the implicit backward
  gradcheck.py    central-difference verification of every gradient
experiments/    training and evaluation scripts
scripts/        data download helpers
results/        one JSON per run — report tables are generated from these
reports/        LaTeX: one shared style + one file per submission
```

## Quick start

```bash
pip install -r requirements.txt
make gradcheck      # verify the hand-derived gradients — must print PASS
make phase1         # data, gradcheck, both solvers, then build Report I
make help           # all targets
```

## Reports

All four documents share `reports/deqreport.sty`, so formatting, header and
bibliography are identical across submissions. Name, TUID and instructor are
set once in that file; `reports/refs.bib` is the single shared bibliography.

```bash
make reports        # build every PDF
make check          # build and print page counts against the limits
```

| Source | Submission | Due | Page limit |
| --- | --- | --- | --- |
| `reports/proposal.tex` | Proposal | Nov 10 | 2 |
| `reports/report1.tex` | Progress Report I | Nov 17 | 2 |
| `reports/report2.tex` | Progress Report II | Dec 1 | 2 |
| `reports/final.tex` | Final Report | Dec 8 | 5 |

The PDF actually handed in for each deadline is kept in `reports/submitted/`.

Reports II and the final report are laid out with their sections in place and
`\TODO` markers where the numbers go.

## Requirements

Python 3.10+ and NumPy for Phase 1; PyTorch is added in Phase 2. The reports
need a LaTeX distribution with `latexmk` and `bibtex`.
