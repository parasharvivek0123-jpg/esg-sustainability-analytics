"""Step 8 — Interactive ESG dashboard.  Run: streamlit run app.py"""
import streamlit as st
import pandas as pd, plotly.express as px

st.set_page_config(page_title="ESG Analytics", layout="wide")
df = pd.read_pickle("data/features.pkl")   # clean + feature-engineered data

st.title("🌱 ESG Analytics Dashboard")
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

st.dataframe(d[["ticker", "year", "e_score", "s_score", "g_score", "total_score"]])
