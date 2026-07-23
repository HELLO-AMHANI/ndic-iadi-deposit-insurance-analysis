# NDIC + IADI Analytics Project

**Author:** Promise O. Amhanesi

## Research Question
How does NDIC's fund adequacy, coverage ratio, and premium system compare against IADI Core Principles and peer member DICs?

## Tools
- Python (pandas, scikit-learn, XGBoost, SHAP, Plotly)
- SQL (SQLite)
- MS Excel
- GitHub

## IADI Framework
This project benchmarks Nigeria's NDIC against the [IADI Core Principles for Effective Deposit Insurance Systems](https://www.iadi.org/en/core-principles-and-guidance/core-principles/).

## Data Sources (to be updated)
- NDIC Annual Reports — https://www.ndic.gov.ng/publications
- World Bank GFDD — https://databank.worldbank.org
- IADI Annual Survey — https://www.iadi.org/en/about-iadi/annual-survey

## Project Structure
data/raw/       → downloaded source files
data/clean/     → cleaned CSVs used by notebooks
sql/            → ingest and transform scripts
notebooks/      → Python Jupyter notebooks
excel/          → Excel workbooks
results/figs/   → charts and outputs
docs/           → report.pdf, slides.pdf, demo link
dashboard/      → Streamlit app (optional)

## Status
🚧 In progress — Stage 1: Data Sourcing and SQL ingestion
