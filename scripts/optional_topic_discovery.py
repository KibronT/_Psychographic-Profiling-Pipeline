"""
optional_topic_discovery.py - Cluster articles per outlet, label clusters with TF-IDF.

This is *not* part of the core handoff pipeline. The outlet-level psychographic
profile is built sentence-by-sentence in scripts 03-05; you do not need clusters
for that. This script exists for researchers who want a complementary
topic-discovery view of WHAT each outlet is covering.

Pipeline: SBERT article embeddings -> UMAP (15 dims) -> HDBSCAN -> per-cluster
TF-IDF bigram labels. Outputs per-outlet CSV (one row per cluster) and a 2D
projection PNG for visualization.

Input:  outputs/intermediate/articles.parquet
Output: outputs/topics/clusters_<outlet>.csv
        outputs/topics/clusters_<outlet>.png
"""

import re
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import (
    INTERMEDIATE,
    OUTLETS,
    OUTLET_DISPLAY,
    OUTPUTS,
    RANDOM_STATE,
    SBERT_MODEL,
    ensure_dirs,
)


TOPICS_DIR = OUTPUTS / "topics"

# Per-outlet HDBSCAN params (carried forward from prior iteration)
HDBSCAN_PARAMS = {
    "abs":     {"min_cluster_size": 5, "min_samples": 2},
    "gma":     {"min_cluster_size": 8, "min_samples": 3},
    "rappler": {"min_cluster_size": 8, "min_samples": 3},
}

UMAP_N_COMPONENTS = 15
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0

TAGALOG_STOPWORDS = {
    "na", "sa", "ng", "ang", "mga", "ay", "po", "din", "rin", "kay",
    "kaya", "para", "dahil", "kung", "naman", "lang", "yung", "nung",
}


def cluster_outlet(articles_outlet: pd.DataFrame, params: dict, embeddings: np.ndarray) -> pd.DataFrame:
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)
    out = articles_outlet.copy().reset_index(drop=True)
    out["cluster"] = labels
    return out, reduced


def label_clusters(df: pd.DataFrame) -> dict[int, str]:
    eng_stops = set(TfidfVectorizer(stop_words="english").get_stop_words())
    stops = list(eng_stops | TAGALOG_STOPWORDS)
    labels = {}
    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            labels[cluster_id] = "(noise)"
            continue
        texts = df[df["cluster"] == cluster_id]["full_text"].tolist()
        if not texts:
            continue
        try:
            vec = TfidfVectorizer(ngram_range=(2, 2), stop_words=stops, max_features=200)
            X = vec.fit_transform(texts)
            scores = np.asarray(X.mean(axis=0)).flatten()
            top_idx = scores.argsort()[::-1][:3]
            terms = vec.get_feature_names_out()
            top_terms = [terms[i] for i in top_idx if scores[i] > 0]
            labels[cluster_id] = ", ".join(top_terms) if top_terms else "(unlabeled)"
        except ValueError:
            labels[cluster_id] = "(unlabeled)"
    return labels


def plot_clusters(reduced_2d: np.ndarray, labels: np.ndarray, outlet: str, out_path: Path):
    # Project to 2D for visualization using the same reducer in 2D space
    reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=RANDOM_STATE)
    coords = reducer_2d.fit_transform(reduced_2d)
    fig, ax = plt.subplots(figsize=(9, 7))
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", max(1, len(unique_labels)))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(coords[mask, 0], coords[mask, 1], s=15, c=[cmap(i)],
                   label=f"noise" if lab == -1 else f"c{lab}", alpha=0.7)
    ax.set_title(f"{OUTLET_DISPLAY[outlet]} - article clusters", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    if len(unique_labels) <= 25:
        ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1), fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ensure_dirs()
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    articles = pd.read_parquet(INTERMEDIATE / "articles.parquet")
    print(f"Loaded {len(articles)} articles")

    model = SentenceTransformer(SBERT_MODEL)

    for outlet in OUTLETS:
        sub = articles[articles["outlet"] == outlet].reset_index(drop=True)
        print(f"\n{outlet}: embedding {len(sub)} articles...")
        embs = model.encode(sub["full_text"].tolist(), batch_size=128,
                            show_progress_bar=True, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)

        print(f"  Clustering (UMAP-{UMAP_N_COMPONENTS} + HDBSCAN)...")
        clustered, reduced = cluster_outlet(sub, HDBSCAN_PARAMS[outlet], embs)
        labels = label_clusters(clustered)

        summary = []
        for cluster_id in sorted(clustered["cluster"].unique()):
            n = int((clustered["cluster"] == cluster_id).sum())
            summary.append({
                "cluster_id": int(cluster_id),
                "n_articles": n,
                "tfidf_label": labels[cluster_id],
            })
        sum_df = pd.DataFrame(summary).sort_values("n_articles", ascending=False)
        sum_df.to_csv(TOPICS_DIR / f"clusters_{outlet}.csv", index=False)
        n_real = (sum_df["cluster_id"] != -1).sum()
        print(f"  Found {n_real} clusters (plus noise)")

        plot_clusters(reduced, clustered["cluster"].values, outlet,
                      TOPICS_DIR / f"clusters_{outlet}.png")

    print(f"\nWrote topic outputs to {TOPICS_DIR}")


if __name__ == "__main__":
    main()
