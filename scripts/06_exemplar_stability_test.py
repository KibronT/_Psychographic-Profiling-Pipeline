"""
06_exemplar_stability_test.py - Find the smallest stable exemplar size.

Runs scoring + aggregation at sizes [15, 25, 35, 50] for both frameworks
(MFT sizes are halved per pole). Computes per-outlet rankings by % dominant.
Compares consecutive sizes via Spearman rank correlation.

Decision rule (per user spec):
  Primary: pick the smallest size where mean Spearman across outlets to the
           next-larger size is >= 0.95.
  Secondary check: whether top-3 categories per outlet are unchanged
                   between that size and the next-larger size.
  Fallback: if no size reaches >= 0.95, use the largest (50) and flag
            non-convergence in stability/README.md.

Outputs: outputs/stability/stability_results.csv
         outputs/stability/stability_convergence.png
         outputs/stability/README.md
         outputs/stability/active_sizes.json   <-- read this to lock sizes
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

from utils import (
    INTERMEDIATE,
    OUTLETS,
    RANDOM_STATE,
    SBERT_MODEL,
    STABILITY,
    ensure_dirs,
    load_mft_exemplars,
    load_schwartz_exemplars,
)


SIZES = [15, 25, 35, 50]
SPEARMAN_THRESHOLD = 0.95


def schwartz_rankings(model, sentences: pd.DataFrame, embeddings: np.ndarray, exemplars: dict, size: int):
    """Returns dict: outlet -> ranked list of (value, pct_dominant) sorted desc."""
    rng = np.random.default_rng(RANDOM_STATE)
    value_names = list(exemplars["values"].keys())
    centroids = []
    for v in value_names:
        pool = exemplars["values"][v]["exemplars"]
        if size < len(pool):
            idx = rng.choice(len(pool), size=size, replace=False)
            chosen = [pool[i] for i in idx]
        else:
            chosen = pool
        embs = model.encode(chosen, normalize_embeddings=True, convert_to_numpy=True)
        c = embs.mean(axis=0)
        centroids.append(c / np.linalg.norm(c))
    centroids = np.vstack(centroids).astype(np.float32)
    sims = embeddings @ centroids.T
    dominant = np.array(value_names)[sims.argmax(axis=1)]
    salience = sims.max(axis=1)
    out = {}
    for outlet in OUTLETS:
        mask = (sentences["outlet"] == outlet).values
        s_sub = salience[mask]
        d_sub = dominant[mask]
        cut = np.median(s_sub)
        keep = d_sub[s_sub >= cut]
        counts = pd.Series(keep).value_counts().reindex(value_names, fill_value=0)
        pct = counts / counts.sum() * 100
        out[outlet] = pct.sort_values(ascending=False).index.tolist()
    return out, value_names


def mft_rankings(model, sentences: pd.DataFrame, embeddings: np.ndarray, exemplars: dict, size: int):
    rng = np.random.default_rng(RANDOM_STATE)
    foundation_names = list(exemplars["foundations"].keys())
    n_per_pole = size // 2
    virtue_c, vice_c = [], []
    for fname in foundation_names:
        for pole, store in [("virtue", virtue_c), ("vice", vice_c)]:
            pool = exemplars["foundations"][fname]["exemplars"][pole]
            if n_per_pole < len(pool):
                idx = rng.choice(len(pool), size=n_per_pole, replace=False)
                chosen = [pool[i] for i in idx]
            else:
                chosen = pool
            embs = model.encode(chosen, normalize_embeddings=True, convert_to_numpy=True)
            c = embs.mean(axis=0)
            store.append(c / np.linalg.norm(c))
    virtue_c = np.vstack(virtue_c).astype(np.float32)
    vice_c = np.vstack(vice_c).astype(np.float32)
    combined = np.maximum(embeddings @ virtue_c.T, embeddings @ vice_c.T)
    dominant = np.array(foundation_names)[combined.argmax(axis=1)]
    salience = combined.max(axis=1)
    out = {}
    for outlet in OUTLETS:
        mask = (sentences["outlet"] == outlet).values
        s_sub = salience[mask]
        d_sub = dominant[mask]
        cut = np.median(s_sub)
        keep = d_sub[s_sub >= cut]
        counts = pd.Series(keep).value_counts().reindex(foundation_names, fill_value=0)
        pct = counts / counts.sum() * 100
        out[outlet] = pct.sort_values(ascending=False).index.tolist()
    return out, foundation_names


def spearman_between(rank_a: list[str], rank_b: list[str], categories: list[str]) -> float:
    """Spearman rank correlation treating rank position as the variable."""
    a_pos = {c: rank_a.index(c) for c in categories}
    b_pos = {c: rank_b.index(c) for c in categories}
    a = [a_pos[c] for c in categories]
    b = [b_pos[c] for c in categories]
    rho, _ = spearmanr(a, b)
    return rho


def evaluate_framework(name: str, rank_fn, exemplars: dict, model, sentences, embeddings) -> tuple[pd.DataFrame, int, str]:
    """Returns (results_df, chosen_size, note)."""
    print(f"\n=== {name.upper()} stability test ===")
    rankings_by_size = {}
    for size in SIZES:
        print(f"  Scoring at size {size}...")
        ranks, cats = rank_fn(model, sentences, embeddings, exemplars, size)
        rankings_by_size[size] = (ranks, cats)

    rows = []
    for i in range(len(SIZES) - 1):
        sa, sb = SIZES[i], SIZES[i + 1]
        ranks_a, cats = rankings_by_size[sa]
        ranks_b, _ = rankings_by_size[sb]
        rhos = []
        top3_match = []
        for outlet in OUTLETS:
            rho = spearman_between(ranks_a[outlet], ranks_b[outlet], cats)
            rhos.append(rho)
            top3_match.append(set(ranks_a[outlet][:3]) == set(ranks_b[outlet][:3]))
        rows.append({
            "framework": name,
            "size_a": sa,
            "size_b": sb,
            "mean_spearman": float(np.mean(rhos)),
            "min_spearman": float(np.min(rhos)),
            **{f"spearman_{o}": r for o, r in zip(OUTLETS, rhos)},
            "all_top3_unchanged": all(top3_match),
        })

    df = pd.DataFrame(rows)
    chosen_size = SIZES[-1]
    note = f"No consecutive pair reached mean Spearman >= {SPEARMAN_THRESHOLD}; using largest size (50)."
    for r in rows:
        if r["mean_spearman"] >= SPEARMAN_THRESHOLD:
            chosen_size = r["size_a"]
            note = (
                f"Size {chosen_size} reaches mean Spearman {r['mean_spearman']:.3f} with size "
                f"{r['size_b']} (>= {SPEARMAN_THRESHOLD}). Top-3 unchanged across all outlets: "
                f"{r['all_top3_unchanged']}."
            )
            break
    print(f"  -> chosen size: {chosen_size}  ({note})")
    return df, chosen_size, note


def plot_convergence(results: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for fw in results["framework"].unique():
        sub = results[results["framework"] == fw]
        x = [f"{a}→{b}" for a, b in zip(sub["size_a"], sub["size_b"])]
        ax.plot(x, sub["mean_spearman"], marker="o", label=fw)
    ax.axhline(SPEARMAN_THRESHOLD, color="grey", linestyle="--", alpha=0.6, label=f"threshold = {SPEARMAN_THRESHOLD}")
    ax.set_ylabel("Mean Spearman across outlets")
    ax.set_xlabel("Consecutive size comparison")
    ax.set_title("Exemplar-size stability: ranking correlation between sizes")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ensure_dirs()
    sentences = pd.read_parquet(INTERMEDIATE / "sentences.parquet")
    embeddings = np.load(INTERMEDIATE / "sentence_embeddings.npy")
    print(f"Loaded {len(sentences)} sentences, {embeddings.shape} embeddings")

    model = SentenceTransformer(SBERT_MODEL)

    s_results, s_chosen, s_note = evaluate_framework(
        "schwartz", schwartz_rankings, load_schwartz_exemplars(), model, sentences, embeddings
    )
    m_results, m_chosen, m_note = evaluate_framework(
        "mft", mft_rankings, load_mft_exemplars(), model, sentences, embeddings
    )

    all_results = pd.concat([s_results, m_results], ignore_index=True)
    all_results.to_csv(STABILITY / "stability_results.csv", index=False)
    plot_convergence(all_results, STABILITY / "stability_convergence.png")

    active = {"ACTIVE_SCHWARTZ_SIZE": int(s_chosen), "ACTIVE_MFT_SIZE": int(m_chosen)}
    with open(STABILITY / "active_sizes.json", "w") as f:
        json.dump(active, f, indent=2)

    readme = f"""# Exemplar Stability Test Results

## Decision rule
Pick the smallest exemplar size where the mean Spearman rank correlation
between per-outlet rankings at that size and the next-larger size is
>= {SPEARMAN_THRESHOLD}. Fallback: use the largest tested size (50) if no size
converges.

## Sizes tested
{SIZES}  (MFT sizes are split evenly between virtue and vice poles.)

## Chosen sizes
- Schwartz: **{s_chosen}** per value
  - {s_note}
- MFT: **{m_chosen}** per foundation
  - {m_note}

## How to apply
Edit `scripts/utils.py` and set:

```python
ACTIVE_SCHWARTZ_SIZE = {s_chosen}
ACTIVE_MFT_SIZE = {m_chosen}
```

Then rerun `03_score_schwartz.py`, `04_score_mft.py`, and `05_build_profiles.py`
to produce profiles at the chosen sizes.

## Files
- `stability_results.csv` - full Spearman + top-3 results across all size pairs
- `stability_convergence.png` - visualization of mean Spearman by size pair
- `active_sizes.json` - machine-readable chosen sizes
"""
    with open(STABILITY / "README.md", "w") as f:
        f.write(readme)

    print(f"\nWrote results to {STABILITY}")
    print(f"\n>>> Set these in scripts/utils.py:")
    print(f"    ACTIVE_SCHWARTZ_SIZE = {s_chosen}")
    print(f"    ACTIVE_MFT_SIZE = {m_chosen}")


if __name__ == "__main__":
    main()
