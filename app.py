"""Step 8 — Interactive ESG dashboard.  Run: streamlit run app.py"""
import streamlit as st
import pandas as pd, plotly.express as px, joblib

st.set_page_config(page_title="ESG Analytics", layout="wide")
df = pd.read_pickle("data/features.pkl")   # clean + feature-engineered data
model = joblib.load("outputs/models/esg_model.pkl")   # Step 6 forecasting model
FEATURES = ["prev_total", "prev_e", "prev_s", "prev_g", "roll_mean_total",
            "word_count", "entity_count", "years_since_2014", "report_seq"]

st.title("ESG Analytics Dashboard")
st.caption(f"{df.shape[0]} reports · {df['ticker'].nunique()} companies · 2014-2023")

# ---- Sidebar filters ----
tickers = st.sidebar.multiselect("Companies", sorted(df.ticker.unique()))
yr = st.sidebar.slider("Year range", int(df.year.min()), int(df.year.max()),
                       (int(df.year.min()), int(df.year.max())))
d = df[df.year.between(*yr)]
if tickers:
    d = d[d.ticker.isin(tickers)]

# ---- KPI row ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg E", f"{d.e_score.mean():.1f}")
c2.metric("Avg S", f"{d.s_score.mean():.1f}")
c3.metric("Avg G", f"{d.g_score.mean():.1f}")
c4.metric("Avg Total", f"{d.total_score.mean():.1f}")

# ---- Trend chart ----
trend = d.groupby("year")[["e_score", "s_score", "g_score"]].mean().reset_index()
st.plotly_chart(px.line(trend, x="year", y=["e_score", "s_score", "g_score"],
                markers=True, title="ESG Trend Over Time"), use_container_width=True)

# ---- Top companies ----
top = (d.groupby("ticker")["total_score"].mean()
         .sort_values(ascending=False).head(10).reset_index())
st.plotly_chart(px.bar(top, x="total_score", y="ticker", orientation="h",
                title="Top 10 Companies by ESG Score"), use_container_width=True)

# ===================================================================
# Step 6 model, live in the app — Next-Year ESG Forecast
# ===================================================================
st.header("🔮 Next-Year ESG Forecast")
st.caption("Leakage-safe Linear Regression (Step 6) — predicts next-year total_score "
           "from past/structural features only.")

hist = d.dropna(subset=["prev_total", "roll_mean_total"]).copy()
if len(hist) == 0:
    st.info("Not enough history in the current filter to forecast (need companies with a prior report).")
else:
    from sklearn.metrics import r2_score, mean_absolute_error
    hist["predicted"] = model.predict(hist[FEATURES])

    m1, m2, m3 = st.columns(3)
    m1.metric("Model R²", f"{r2_score(hist.total_score, hist.predicted):.3f}")
    m2.metric("MAE (points)", f"{mean_absolute_error(hist.total_score, hist.predicted):.2f}")
    m3.metric("Reports scored", f"{len(hist)}")

    left, right = st.columns(2)
    # predicted vs actual — how good the model is
    fig = px.scatter(hist, x="total_score", y="predicted", hover_data=["ticker", "year"],
                     title="Predicted vs Actual total_score", opacity=0.6)
    lo, hi = float(hist.total_score.min()), float(hist.total_score.max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(dash="dash", color="gray"))
    left.plotly_chart(fig, use_container_width=True)

    # genuine next-year forecast from each company's latest report
    latest = hist.sort_values("year").groupby("ticker").tail(1).copy()
    roll = df.groupby("ticker")["total_score"].mean()
    nxt = pd.DataFrame({
        "prev_total": latest.total_score.values,
        "prev_e": latest.e_score.values,
        "prev_s": latest.s_score.values,
        "prev_g": latest.g_score.values,
        "roll_mean_total": roll.reindex(latest.ticker).values,
        "word_count": latest.word_count.values,
        "entity_count": latest.entity_count.values,
        "years_since_2014": (latest.year + 1 - 2014).values,
        "report_seq": (latest.report_seq + 1).values,
    })
    latest["next_year"] = latest.year + 1
    latest["forecast"] = model.predict(nxt[FEATURES])
    latest["change"] = latest.forecast - latest.total_score
    fc = (latest[["ticker", "year", "total_score", "next_year", "forecast", "change"]]
          .round(2).sort_values("forecast", ascending=False)
          .rename(columns={"year": "latest_year", "total_score": "latest_total"}))
    right.markdown("**Per-company next-year forecast**")
    right.dataframe(fc, use_container_width=True, height=430)

st.divider()
st.subheader("Filtered reports")
st.dataframe(d[["ticker", "year", "e_score", "s_score", "g_score", "total_score"]])
