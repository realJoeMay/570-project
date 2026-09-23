# AGENTS.md

## Project

Parish Ministry Benchmark is a Python data-mining project that collects Catholic parish data, extracts and categorizes ministries using AI, identifies comparable parishes through clustering, and generates parish comparison reports.

## Structure

The project is notebook-first. Each major pipeline step has one numbered notebook at the project root:

1. `01_build_site_list.ipynb`
2. `02_extract_ministries.ipynb`
3. `03_validate_ministries.ipynb`
4. `04_extract_features.ipynb`
5. `05_cluster_parishes.ipynb`
6. `06_generate_reports.ipynb`

Data produced by each step belongs in the corresponding numbered directory under `data/`.

Reusable Python code belongs in `utils/`. Keep workflow, exploration, and analysis in notebooks; move code into `utils/` when it is reusable or makes a notebook unnecessarily complex.

Final generated reports belong in `reports/`.

## Development Guidelines

* Use Python and pandas for data processing.
* Prefer simple, readable solutions over unnecessary abstractions.
* Keep notebooks focused on their specific pipeline step.
* Do not duplicate reusable functions across notebooks.
* Preserve downloaded source data in `raw/` directories and do not modify it.
* Use relative project paths rather than machine-specific absolute paths.
* Do not silently modify the schema of datasets consumed by later pipeline steps.
* Add dependencies to `pyproject.toml`.
* Follow existing naming and formatting conventions.
