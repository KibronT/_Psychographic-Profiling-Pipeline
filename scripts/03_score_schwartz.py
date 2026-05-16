"""
03_score_schwartz.py - Score every sentence against Schwartz value centroids.

Input:  outputs/intermediate/sentences.parquet
        outputs/intermediate/sentence_embeddings.npy
        reference/schwartz_exemplars.json
Output: outputs/intermediate/schwartz_sentence_scores.parquet
            sentence_id, schwartz_<Value> cosine sim (9 cols),
            schwartz_dominant, schwartz_salience

Centroids are the mean of L2-normalized exemplar embeddings, then renormalized.
Salience = max similarity across the 9 values. The pipeline downstream filters
sentences below the median salience before aggregation.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import (
    ACTIVE_SCHWARTZ_SIZE,
    INTERMEDIATE,
    RANDOM_STATE,
    SBERT_MODEL,
    ensure_dirs,
    load_schwartz_exemplars,
)


def build_centroids(model, exemplars: dict, n_per_value: int, rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    """Sample n_per_value exemplars per value (or all if pool is smaller),
    embed, and return (value_names, centroids matrix V x 384)."""
    value_names = list(exemplars["values"].keys())
    centroids = []
    for v in value_names:
        pool = exemplars["values"][v]["exemplars"]
        if n_per_value < len(pool):
            idx = rng.choice(len(pool), size=n_per_value, replace=False)
            chosen = [pool[i] for i in idx]
        else:
            chosen = pool
        embs = model.encode(chosen, normalize_embeddings=True, convert_to_numpy=True)
        c = embs.mean(axis=0)
        c = c / np.linalg.norm(c)
        centroids.append(c)
    return value_names, np.vstack(centroids).astype(np.float32)


def main():
    ensure_dirs()
    sentences = pd.read_parquet(INTERMEDIATE / "sentences.parquet")
    embeddings = np.load(INTERMEDIATE / "sentence_embeddings.npy")
    exemplars = load_schwartz_exemplars()
    print(f"Loaded {len(sentences)} sentences, {embeddings.shape} embeddings")
    print(f"Sampling {ACTIVE_SCHWARTZ_SIZE} exemplars per value (random_state={RANDOM_STATE})")

    model = SentenceTransformer(SBERT_MODEL)
    rng = np.random.default_rng(RANDOM_STATE)
    value_names, centroids = build_centroids(model, exemplars, ACTIVE_SCHWARTZ_SIZE, rng)
    print(f"Built {len(value_names)} value centroids")

    # Cosine similarity (embeddings + centroids are L2-normalized)
    sims = embeddings @ centroids.T  # N x V

    out = sentences[["sentence_id"]].copy()
    for i, v in enumerate(value_names):
        out[f"schwartz_{v}"] = sims[:, i]
    out["schwartz_dominant"] = [value_names[i] for i in sims.argmax(axis=1)]
    out["schwartz_salience"] = sims.max(axis=1)

    out_path = INTERMEDIATE / "schwartz_sentence_scores.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nWrote {len(out)} sentence scores to {out_path}")


if __name__ == "__main__":
    main()
