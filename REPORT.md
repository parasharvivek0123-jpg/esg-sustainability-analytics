# ESG Sustainability Analytics — Methodology Report

**Dataset:** 866 ESG sustainability reports, 263 S&P 500 companies, 2014–2023 (Kaggle, pre-scored on E/S/G).
**Goal:** build a defensible, end-to-end layer on top of the raw scores — analytics, ML, and a GenAI app.
**Pipeline:** Clean → Validate → Feature Engineering → EDA → Risk Index → Forecast → GenAI Copilot → Dashboard.

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

## Step 7 — GenAI Copilot (RAG)
An analyst-facing assistant that answers plain-English questions grounded in **this** dataset, using **Retrieval-Augmented Generation**:
- **Retrieval** (`esg_rag.py`, runs offline): each report is turned into a TF-IDF vector; a query is matched by cosine similarity and the top-k most relevant reports are returned.
- **Generation** (`esg_copilot.py`): the retrieved rows / a compact stats summary are passed as context to **Mistral** (`mistral-small-latest`, free tier) with a system instruction to answer *only* from the provided data — which reduces hallucination.
- **Security:** the API key is read from a git-ignored `.env` via `os.getenv`, never hard-coded or committed.

## Step 8 — Interactive Dashboard (Streamlit)
`app.py` packages the results into a demo-able web app in pure Python: sidebar filters (company, year range), ESG KPI tiles, an E/S/G trend line, a top-10 ranking bar chart, and a filtered data table. Streamlit is chosen over a BI tool because the trained model and the RAG copilot integrate directly in Python. Run with `streamlit run app.py`.

---

## Headline results
| Metric | Value |
|--------|-------|
| Forecast R² (held-out) | 0.913 |
| Dominant predictor | `prev_total` (persistence) |
| Copilot | RAG (TF-IDF retrieval + Mistral generation), data-grounded |
| Dashboard | Streamlit app, filters + KPIs + charts + table |

**Story:** descriptive (clean → risk index) → predictive (leakage-safe forecast) → conversational (RAG copilot) → product (dashboard). Data analytics + ML + GenAI, end to end.

Tools: Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Plotly, Streamlit, Mistral (GenAI).
