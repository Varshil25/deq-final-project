# Top-level entry points. `make help` lists them.
PY ?= python3

.PHONY: help data gradcheck train-mnist phase1 figures reports check clean

help:
	@echo "make data        download and cache MNIST into data/"
	@echo "make gradcheck   verify the hand-derived gradients (must PASS)"
	@echo "make train-mnist train the NumPy DEQ with both solvers"
	@echo "make phase1      data + gradcheck + train-mnist, then build Report I"
	@echo "make figures     regenerate the report figures from results/"
	@echo "make reports     build every report PDF"
	@echo "make check       build reports and print page counts vs the limits"
	@echo "make clean       remove build artifacts (keeps data/ and results/)"

data:
	$(PY) scripts/get_mnist.py

gradcheck:
	$(PY) -m deq.gradcheck

train-mnist: data
	$(PY) experiments/train_mnist_numpy.py --solver anderson --epochs 5
	$(PY) experiments/train_mnist_numpy.py --solver fixed_point --epochs 5

phase1: gradcheck train-mnist figures
	$(MAKE) -C reports report1

figures:
	$(PY) scripts/make_figures.py

reports: figures
	$(MAKE) -C reports all

check:
	$(MAKE) -C reports check

clean:
	$(MAKE) -C reports clean
	find . -name __pycache__ -type d -exec rm -rf {} +
