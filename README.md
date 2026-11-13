# SupeRR Selector Season 2 — Round 1: Pressure Performance System

A dual-innings Residual+Skip Neural Network Win-Probability Leverage and Empirical-Bayes player evaluation framework for identifying the most reliable T20 batters and bowlers under pressure (IPL & T20I, 2022–2026).

## Structure
- `src/01_clean_and_eda.py`: Data hygiene, dismissal/extra standardization, DLS & Super Over tagging, and exploratory analysis.
- `src/02_hybrid_pipeline.py`: 5-fold match-grouped Residual+Skip Neural Win-Probability models (1st & 2nd innings), 6-outcome counterfactual leverage, context-adjusted baselines (`xR_adj`, `xW`), own-baseline Empirical-Bayes shrinkage, and final Top-10 rankings.
- `src/03_generate_rr_report.py`: Generates Rajasthan Royals-themed Matplotlib charts and compiles the 5-page submission PDF via WeasyPrint.
- `outputs/`: Final ranked CSVs (`hybrid_batter_rankings.csv`, `hybrid_bowler_rankings.csv`) and figures.

## Quick Start
```bash
pip install -r requirements.txt
python src/01_clean_and_eda.py
python src/02_hybrid_pipeline.py
python src/03_generate_rr_report.py
```
