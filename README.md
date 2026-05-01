# AIO_OIA — Datathon 2026 Round 1

**Team:** AIO_OIA | **Competition:** [Datathon 2026 — The Gridbreakers](https://www.kaggle.com/competitions/datathon-2026-round-1)

Best Kaggle MAE: **628,317**

---

## Structure

```
├── docs/Style/
│   ├── report_vi.tex              # Báo cáo chính (NeurIPS template, tiếng Việt)
│   ├── report_vi.pdf              # PDF báo cáo
│   └── neurips_2025.sty           # NeurIPS LaTeX style file
├── notebooks/
│   ├── 09_eda_complete.ipynb      # Full EDA — 4 stories + figures (Part 2)
│   ├── 10_mcq_verify.ipynb        # MCQ Part 1 answers
│   ├── story1_revenue_overview.ipynb
│   ├── story2_customer_churn.ipynb
│   ├── story3_product_returns.ipynb
│   ├── story4_inventory_stockout.ipynb
│   └── story5_marketing_roi.ipynb
├── scripts/
│   ├── v22_targeted_fix.py        # Best submission pipeline (MAE 628,317)
│   ├── run_baseline.py            # Pre-COVID seasonal model
│   ├── creative_submissions.py    # DOW correction + seasonal variants
│   ├── lgbm_v3_proper.py          # LightGBM residual model
│   └── mcq_official.py            # MCQ answers (data-driven)
├── src/                           # Data loading & feature engineering
├── submissions/
│   └── v22_b045.csv               # Best submission (MAE 628,317)
├── report/figures/                 # All figures used in report
│   └── eda/                       # EDA story figures
└── requirements.txt
```

## Reproduce Best Submission

```bash
pip install -r requirements.txt

# Place raw data in src/data/raw/:
#   sales.csv, sales_test.csv, orders.csv, order_items.csv,
#   products.csv, customers.csv, etc.

# Step 1: Generate base ensemble (v17 blend)
python scripts/creative_submissions.py
python scripts/lgbm_v3_proper.py

# Step 2: Apply COGS beta fine-tuning (best submission)
python scripts/v22_targeted_fix.py

# Output: submissions/v22_b045.csv (MAE 628,317)
```

## Method

Pipeline dự báo Revenue hàng ngày cho 548 ngày (01/2023–07/2024):

1. **Calendar-only LightGBM**: Fourier harmonics, Tết proximity, DOM cycle, holiday flags. Trained with 100× sample weight on Golden Era (2014–2018). No lag features (would leak into test set).

2. **Level calibration**: 2023 daily mean = 4,135,973 VND, 2024 Jan–Jul mean = 4,967,327 VND.

3. **3-way ensemble**: cal_lgbm (65%) + v1_lgbm with lunar calendar (12%) + diverse pool of 20 models (23%).

4. **August loss leader correction**: August odd-year Revenue × 0.809, COGS/Revenue = 1.369 (consistent 5/5 odd years in training data).

5. **COGS derivation**: COGS = Revenue × empirical COGS/Revenue ratio by month and year parity (odd/even), blended with model COGS at β=0.45.

## Data

Place raw data in `src/data/raw/`. Data not included (competition terms).

## Report

`docs/Style/report_vi.tex` — NeurIPS format, 4 pages:
- Part 2: 4 EDA stories (Descriptive → Diagnostic → Predictive → Prescriptive)
- Part 3: Forecasting pipeline, ablation study, feature importance

Compile: `cd docs/Style && pdflatex report_vi.tex`
