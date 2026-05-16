"""
02_embed_sentences.py - Split articles into sentences, embed with SBERT.

Input:  outputs/intermediate/articles.parquet
Output: outputs/intermediate/sentences.parquet  (sentence_id, article_id, outlet, sentence_text)
        outputs/intermediate/sentence_embeddings.npy  (N x 384 float32, row-aligned to sentences.parquet)

Sentence splitting uses a simple regex on .!? - fine for SBERT input quality
and orders of magnitude faster than spaCy. Embeddings are L2-normalized so
cosine similarity reduces to a dot product downstream.
"""

import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import INTERMEDIATE, SBERT_MODEL, ensure_dirs


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
MIN_SENT_WORDS = 4


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = SENT_SPLIT.split(text)
    return [p.strip() for p in parts if len(p.split()) >= MIN_SENT_WORDS]


def main():
    ensure_dirs()
    articles = pd.read_parquet(INTERMEDIATE / "articles.parquet")
    print(f"Loaded {len(articles)} articles")

    rows = []
    for _, row in articles.iterrows():
        for s in split_sentences(row["full_text"]):
            rows.append(
                {"article_id": row["article_id"], "outlet": row["outlet"], "sentence_text": s}
            )
    sentences = pd.DataFrame(rows)
    sentences.insert(0, "sentence_id", sentences.index.map(lambda i: f"s{i:07d}"))
    print(f"Split into {len(sentences)} sentences")

    print(f"Loading SBERT model: {SBERT_MODEL}")
    model = SentenceTransformer(SBERT_MODEL)
    print("Embedding sentences (this may take a few minutes)...")
    embeddings = model.encode(
        sentences["sentence_text"].tolist(),
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    sent_path = INTERMEDIATE / "sentences.parquet"
    emb_path = INTERMEDIATE / "sentence_embeddings.npy"
    sentences.to_parquet(sent_path, index=False)
    np.save(emb_path, embeddings)
    print(f"\nWrote {len(sentences)} sentences to {sent_path}")
    print(f"Wrote embeddings shape {embeddings.shape} to {emb_path}")


if __name__ == "__main__":
    main()
