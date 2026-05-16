"""
04_score_mft.py - Score every sentence against MFT foundation centroids.

Input:  outputs/intermediate/sentences.parquet
        outputs/intermediate/sentence_embeddings.npy
        reference/mft_exemplars.json
Output: outputs/intermediate/mft_sentence_scores.parquet
            sentence_id,
            mft_<foundation> cosine sim (5 cols, combined virtue+vice),
            mft_<foundation>_virtue cosine sim (5 cols),
            mft_<foundation>_vice cosine sim (5 cols),
            mft_dominant, mft_salience, mft_dominant_pole (virtue or vice)

The combined per-foundation similarity is the max of virtue and vice
similarities for that foundation - i.e., a sentence framed in moral terms
of Care counts as Care whether it expresses virtue (compassion) or vice
(cruelty). The pole label tells you which one.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import (
    ACTIVE_MFT_SIZE,
    INTERMEDIATE,
    RANDOM_STATE,
    SBERT_MODEL,
    ensure_dirs,
    load_mft_exemplars,
)


def build_pole_centroids(model, exemplars: dict, n_per_pole: int, rng: np.random.Generator) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build separate virtue and vice centroids per foundation.
    n_per_pole exemplars are sampled per pole (half the active foundation size).
    Returns (foundation_names, virtue_centroids F x 384, vice_centroids F x 384)."""
    foundation_names = list(exemplars["foundations"].keys())
    virtue_centroids = []
    vice_centroids = []
    for fname in foundation_names:
        for pole, store in [("virtue", virtue_centroids), ("vice", vice_centroids)]:
            pool = exemplars["foundations"][fname]["exemplars"][pole]
            if n_per_pole < len(pool):
                idx = rng.choice(len(pool), size=n_per_pole, replace=False)
                chosen = [pool[i] for i in idx]
            else:
                chosen = pool
            embs = model.encode(chosen, normalize_embeddings=True, convert_to_numpy=True)
            c = embs.mean(axis=0)
            c = c / np.linalg.norm(c)
            store.append(c)
    return foundation_names, np.vstack(virtue_centroids).astype(np.float32), np.vstack(vice_centroids).astype(np.float32)


def main():
    ensure_dirs()
    sentences = pd.read_parquet(INTERMEDIATE / "sentences.parquet")
    embeddings = np.load(INTERMEDIATE / "sentence_embeddings.npy")
    exemplars = load_mft_exemplars()
    print(f"Loaded {len(sentences)} sentences, {embeddings.shape} embeddings")

    n_per_pole = ACTIVE_MFT_SIZE // 2  # virtue + vice split
    print(f"Sampling {n_per_pole} virtue + {n_per_pole} vice per foundation (random_state={RANDOM_STATE})")

    model = SentenceTransformer(SBERT_MODEL)
    rng = np.random.default_rng(RANDOM_STATE)
    foundation_names, virtue_c, vice_c = build_pole_centroids(model, exemplars, n_per_pole, rng)
    print(f"Built {len(foundation_names)} foundation centroids (virtue + vice each)")

    virtue_sims = embeddings @ virtue_c.T  # N x F
    vice_sims = embeddings @ vice_c.T      # N x F
    combined = np.maximum(virtue_sims, vice_sims)  # N x F

    out = sentences[["sentence_id"]].copy()
    for i, fname in enumerate(foundation_names):
        out[f"mft_{fname}"] = combined[:, i]
        out[f"mft_{fname}_virtue"] = virtue_sims[:, i]
        out[f"mft_{fname}_vice"] = vice_sims[:, i]
    out["mft_dominant"] = [foundation_names[i] for i in combined.argmax(axis=1)]
    out["mft_salience"] = combined.max(axis=1)
    # pole label for the dominant foundation
    dom_idx = combined.argmax(axis=1)
    out["mft_dominant_pole"] = np.where(
        virtue_sims[np.arange(len(combined)), dom_idx] >= vice_sims[np.arange(len(combined)), dom_idx],
        "virtue",
        "vice",
    )

    out_path = INTERMEDIATE / "mft_sentence_scores.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nWrote {len(out)} sentence scores to {out_path}")


if __name__ == "__main__":
    main()
