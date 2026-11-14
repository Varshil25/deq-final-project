# Deep Equilibrium Models from Scratch

Final project — Deep Learning. Implementing a deep equilibrium model (DEQ)
including the fixed-point forward solver and the implicit-differentiation
backward pass, without running automatic differentiation through the solver.

**Varshil Patel · TUID 916646801 · Instructor: Hongchang Gao**

See [`PLAN.md`](PLAN.md) for the three-phase plan and deadlines.

## Layout

```
deq/            model, solvers, hand-derived gradients
experiments/    training and evaluation scripts
results/        one JSON per run — tables are generated from these
reports/        LaTeX: shared style + one file per submission
scripts/        helpers (data download, figure generation)
```

## Reports

All four documents share `reports/deqreport.sty`, so the formatting, header and
bibliography are identical across submissions. Name, TUID and instructor are
set once in that file.

```bash
cd reports
make report1       # build one
make all           # build everything
make check         # build and print page counts (limits: 2, 2, 5)
```

| File | Submission | Due | Limit |
| --- | --- | --- | --- |
| `proposal.tex` | Proposal | Nov 10 (submitted) | 2 pages |
| `report1.tex` | Progress Report I | Nov 17 | 2 pages |
| `report2.tex` | Progress Report II | Dec 1 | 2 pages |
| `final.tex` | Final Report | Dec 8 | 5 pages |

Each template has its sections laid out with `\TODO` markers where numbers go.

## Running the code

```bash
pip install -r requirements.txt
python -m deq.gradcheck                      # verify hand-derived gradients
python experiments/train_mnist_numpy.py      # Phase 1
```

## Requirements

Python 3.10+, NumPy for Phase 1; PyTorch added in Phase 2. A LaTeX
distribution with `latexmk` for the reports.
