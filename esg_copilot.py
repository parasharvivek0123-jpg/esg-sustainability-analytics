"""Step 7 — GenAI ESG Copilot (Mistral, free tier).

Grounds the LLM in a factual summary of the dataset so answers stay on-data.
Requires MISTRAL_API_KEY in a .env file (never hard-code / commit the key).
"""
import os
import pandas as pd
from mistralai import Mistral                   # pip install mistralai  (free tier)
from dotenv import load_dotenv

load_dotenv()                                   # .env se API key (kabhi code me mat likho)
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

df = pd.read_pickle("data/features.pkl")   # clean + feature-engineered data


def build_context(df):
    """A short factual summary of the data to ground the model."""
    top = df.groupby("ticker")["total_score"].mean().sort_values(ascending=False)
    trend = df.groupby("year")["total_score"].mean().round(2).to_dict()
    return (f"Dataset: {df.shape[0]} sustainability reports, "
            f"{df['ticker'].nunique()} companies, years {df['year'].min()}-{df['year'].max()}.\n"
            f"Top 5 ESG companies: {top.head(5).round(2).to_dict()}\n"
            f"Avg total_score by year: {trend}\n"
            f"Pillar averages: E={df.e_score.mean():.1f}, "
            f"S={df.s_score.mean():.1f}, G={df.g_score.mean():.1f}")


def ask(question):
    context = build_context(df)
    resp = client.chat.complete(              # Mistral: .chat.complete()
        model="mistral-small-latest",         # free tier model (also: open-mistral-7b)
        messages=[
            {"role": "system", "content":
             "You are an ESG analytics assistant. Answer ONLY from the "
             "provided data context. If the data doesn't cover it, say so."},
            {"role": "user", "content": f"DATA:\n{context}\n\nQUESTION: {question}"}
        ],
        temperature=0.2,          # low = factual, high = creative
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    print(ask("Which companies have the highest ESG focus and how did scores trend over time?"))
