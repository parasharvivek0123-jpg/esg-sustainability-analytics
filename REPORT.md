# ESG Sustainability Analytics — Methodology Report

**Dataset:** 866 ESG sustainability reports, 263 S&P 500 companies, 2014–2023 (Kaggle, pre-scored on E/S/G).
**Goal:** build a defensible analytics + ML layer on top of the raw scores — describe, predict, and segment.
**Pipeline:** Clean → Validate → Feature Engineering → EDA → Risk Index → Forecast → Validate & Explain → Segment.

---

## Step 1 — Data Cleaning
Dropped the leftover index column, safe-parsed the string-encoded `ner_entities` lists with `ast.literal_eval` (not `eval`, avoiding code-execution risk), standardized dtypes, and removed duplicate `ticker+year` rows. Result: **864 × 9**.

## Step 2 — Data Quality Checks
Verified the `total ≈ e + s + g` relationship, value ranges, and missing values. A regression of `total` on the three pillars returns coefficients ≈ (1, 1, 1) with R² = 0.999 — **approximately additive**, but ~22% of rows differ from an exact sum by more than rounding, so the relationship is treated with a tolerance check and outliers are flagged rather than assumed away.

## Step 3 — Feature Engineering
Added 18 engineered columns (27 total): text signals (`word_count`, `char_count`, `entity_count`, `entity_diversity`), pillar composition (`e_share`, `s_share`, `g_share`, `dominant_pillar`, `pillar_std`), and **leakage-safe** lag/time features (`prev_total`, `prev_e/s/g`, `roll_mean_total`, `yoy_change`, `years_since_2014`, `report_seq`).

## Step 4 — Exploratory Data Analysis
Four charts (`figures/01`–`04`): score distributions, pillar correlation, temporal trends, and the top-10 disclosers. Social scores are consistently highest, Environmental lowest, and disclosure volume rises over time.

## Step 5 — ESG Risk Index
A weighted composite (0.40 E + 0.30 S + 0.30 G), min-max normalized to 0–100, plus a `risk_band`, ranking all 263 companies.

## Step 6 — Next-Year Forecasting
Predicts next-year `total_score` from **only past/structural features** — current-year e/s/g/total and `yoy_change` are excluded (yoy_change hides the target and would inflate R² to 1.0). Linear Regression reaches **R² ≈ 0.913** on a held-out split, beating both a RandomForest and a naive persistence baseline; the model is saved to `outputs/models/esg_model.pkl`.

## Step 7 — Model Validation & Explainability
A single split isn't proof. Three checks:
- **5-fold cross-validation:** Linear R² = **0.922 ± 0.010** (RandomForest 0.904 ± 0.012) — the Step-6 number is stable, not a lucky split.
- **Time-based, expanding-window validation** (train on years `< y`, test on year `y`): out-of-sample R² = 0.90 (2021), 0.87 (2022), 0.79 (2023). The mild decay on the most recent, smallest fold is expected and reported honestly.
- **Permutation importance:** `prev_total` dominates (≈3.47 R² drop when shuffled), followed by `prev_s` and `prev_e` — ESG scores are highly persistent year-over-year.

Charts: `figures/05_model_validation.png`, `figures/06_permutation_importance.png`.

## Step 8 — Company Segmentation (Unsupervised)
One profile per company (mean E/S/G, diversity, risk over its reports), standardized, clustered with **KMeans**. `k` is chosen by silhouette score, which peaks at **k = 3** (silhouette 0.34). The 263 companies split into interpretable archetypes — *Environment-led leaders*, *Social-led leaders*, and a larger *Social-led laggards* group — visualized with a PCA map and a profile heatmap. Deliverable: `outputs/esg_segments.csv` for peer benchmarking.

Charts: `figures/07_segment_map.png`, `figures/08_segment_profiles.png`.

---

## Headline results
| Metric | Value |
|--------|-------|
| Forecast R² (held-out) | 0.913 |
| Forecast R² (5-fold CV) | 0.922 ± 0.010 |
| Forecast R² (time-based OOS) | ≈ 0.79–0.90 |
| Dominant predictor | `prev_total` (persistence) |
| Segments (silhouette-chosen) | k = 3 archetypes, 263 companies |

**Story:** descriptive (clean → risk index) → predictive (validated forecast) → strategic (segmentation), with leakage control and honest, cross-validated evaluation throughout.

Tools: Python, pandas, NumPy, scikit-learn, matplotlib, seaborn.
