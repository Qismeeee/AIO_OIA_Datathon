# AIO_OIA — Datathon 2026 Round 1

**Team:** AIO_OIA | **Competition:** [Datathon 2026 — The Gridbreakers](https://www.kaggle.com/competitions/datathon-2026-round-1)

Best Kaggle RMSE: **658,785**

---

## Structure

```
├── notebooks/
│   └── 09_eda_complete.ipynb     # Full EDA — 7 stories, 16 figures (Part 2)
├── scripts/
│   ├── run_baseline.py           # Pre-COVID seasonal model
│   ├── creative_submissions.py   # DOW correction + seasonal variants
│   ├── lgbm_v3_proper.py         # LightGBM residual model
│   ├── dow_optimize.py           # DOW + ensemble optimization
│   └── mcq_official.py           # MCQ Part 1 answers (data-driven)
├── src/
│   └── data/loader.py            # Data loading utilities
├── submissions/
│   └── ens_fine_a012.csv         # Best submission (RMSE 658,785)
├── report/figures/               # All generated figures
│   └── eda/                      # EDA story figures
├── AIO_OIA.tex                   # NeurIPS-format report
└── requirements.txt
```

## Reproduce Best Submission

```bash
pip install -r requirements.txt

# 1. Generate DOW-corrected seasonal predictions
python scripts/creative_submissions.py

# 2. Train LightGBM residual model
python scripts/lgbm_v3_proper.py

# 3. Optimal ensemble (alpha=0.124)
python scripts/dow_optimize.py

# Best output: submissions/ens_fine_a012.csv
```

## Method

Three-stage pipeline:

1. **Pre-COVID seasonal profile** (2013-2018): Normalize each year by its annual mean, average by calendar date, re-normalize. Excludes COVID years (2019-2022) which have severely distorted patterns.

2. **Day-of-week correction**: Monthly x weekday residual factors from stable pre-COVID years. 15.6 pp weekly spread (Wed +7.1%, Sat -8.5%).

3. **LightGBM ensemble**: Train on DOW residuals with 48 features. Blend weight a* = 0.124 derived analytically from Kaggle parabola fit.

## Data

Place raw data in `src/data/raw/`:
- `sales.csv` — training data (2012-07-04 to 2022-12-31)
- `sales_test.csv` — test period (2023-01-01 to 2024-07-01)
- `orders.csv`, `order_items.csv`, `products.csv`, `customers.csv`, etc.

Data not included (competition terms).

## EDA Notebook

`notebooks/09_eda_complete.ipynb` — 4-level analysis across 7 stories:

| Story | Topic | Level |
|-------|-------|-------|
| 1 | Revenue trajectory & structural break | Descriptive + Diagnostic |
| 2 | Customer activation & revenue per buyer | Descriptive + Diagnostic |
| 3 | Stockout risk analysis | Diagnostic + Prescriptive |
| 4 | Geographic revenue concentration | Descriptive + Diagnostic |
| 5 | Promotion effectiveness | Diagnostic + Predictive |
| 6 | Executive scorecard & 12-month roadmap | Prescriptive |
| 7 | Tet + DOW + forecast bridge | Predictive + Prescriptive |

## Key Results

| Submission | Kaggle RMSE |
|------------|-------------|
| Lag-365 from 2022 | 902,623 |
| Pre-COVID seasonal (tuned) | 691,282 |
| + DOW correction | 682,679 |
| + LightGBM ensemble (a=0.124) | **658,785** |
