"""Step 7 — RAG retrieval over ESG reports (TF-IDF + cosine similarity).

Runs offline (no API key). Pairs with esg_copilot.py for the generation step.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_pickle("data/features.pkl")   # clean + feature-engineered data

# 1. Turn each report into a vector (TF-IDF = lightweight embedding)
vec = TfidfVectorizer(max_features=5000)
doc_matrix = vec.fit_transform(df["preprocessed_content"].astype(str))


def retrieve(query, k=3):
    """Find the k reports most similar to the query."""
    q = vec.transform([query])
    sims = cosine_similarity(q, doc_matrix).ravel()
    idx = sims.argsort()[::-1][:k]              # top-k similar
    return df.iloc[idx][["ticker", "year", "total_score"]]


# Feed these retrieved rows as context into ask() in esg_copilot.py
if __name__ == "__main__":
    print(retrieve("carbon emission reduction targets"))
