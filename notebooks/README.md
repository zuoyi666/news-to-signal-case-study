# Notebooks

Canonical notebooks directory for research writeups and exploratory analysis.

Canonical notebook:

- `notebooks/case_study.ipynb`

Recommendation: keep notebooks as readable analysis narratives; keep production code and experiments in `src/` + `cli.py`.

By default, notebook output variables in this repo (`FIGURES_DIR`, `TABLES_DIR`) are
resolved to the latest `artifacts/runs/exp_YYYYMMDD_name/` directory so outputs are
aligned with CLI runs.
