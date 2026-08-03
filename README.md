# Material for SR tutorial

Professor: Miles Cranmer

Tutorial lead: Jose M Munoz Arias

---
2026 IAIFI summer school.

The tutorial is split in two parts:
- a parts in where we discuss nice concepts from PySR (sr_tutorial_material/01_tutorial).
- a part where each one of us will get PySR running on our own machine and we will run some experiments (sr_tutorial_material/02_hands_on).
Extra:
- a couple of extra notebooks with more advanced topics that I really think are cool.

## Setup

Requires [conda](https://docs.conda.io/projects/miniconda/en/latest/). Paste this into a
terminal before the tutorial — the first PySR import downloads and precompiles a Julia
backend, which is slow.

```bash
conda create -y -n sr_tutorial python=3.11
conda activate sr_tutorial

pip install "pysr>=1.5" "sympy>=1.14" numpy scipy pandas matplotlib \
  scikit-learn xgboost torch torch-symbolic nonlinear_benchmarks \
  jupyterlab ipykernel nbformat
```

`sympy` must be >= 1.14 — `symtorch` silently no-ops on older versions instead of
raising an error.

To use it later:

```bash
conda activate sr_tutorial
jupyter lab
```