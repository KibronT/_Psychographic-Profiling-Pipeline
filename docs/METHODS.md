# Methods

How the pipeline turns raw article JSONL into outlet-level psychographic
profiles. Six steps, each implemented by one script in `scripts/`.

## Step 1 - Loading and cleaning (`01_clean.py`)

- Per-outlet JSONL files in `data/{outlet}/` are read line by line.
- Each record's `headline` and `text` are concatenated into a single
  `full_text` field (headline + ". " + body).
- Whitespace is normalized and known outlet footers (e.g. "GMA News Online",
  "Rappler.com") are stripped from the body before concatenation.
- Articles with fewer than 20 words after concatenation are dropped.

**Output:** `outputs/intermediate/articles.parquet` (one row per article).

## Step 2 - Sentence splitting and SBERT embedding (`02_embed_sentences.py`)

- Each `full_text` is split on sentence-final punctuation (`.`, `!`, `?`)
  followed by whitespace and a capital letter. Sentences with fewer than
  4 words are dropped.
- Every sentence is embedded with `sentence-transformers/all-MiniLM-L6-v2`
  (384-dim vectors), L2-normalized so cosine similarity reduces to a dot
  product downstream.

**Output:** `outputs/intermediate/sentences.parquet` + a row-aligned
`sentence_embeddings.npy` (float32).

## Step 3 - Schwartz scoring (`03_score_schwartz.py`)

- Loads the Schwartz exemplar pool from `reference/schwartz_exemplars.json`
  (9 values × up to 50 sentences each).
- Samples `ACTIVE_SCHWARTZ_SIZE` exemplars per value with
  `random_state=42`, embeds them, computes the centroid (mean of L2-
  normalized exemplars, then renormalized).
- Every sentence is scored against all 9 centroids by cosine similarity.
- Dominant value = argmax across centroids; salience = max similarity.

**Output:** `outputs/intermediate/schwartz_sentence_scores.parquet`.

## Step 4 - MFT scoring (`04_score_mft.py`)

Same shape as Step 3, with one extension:

- Each MFT foundation has two reference pools - virtue and vice. Both are
  sampled and embedded separately, producing two centroids per foundation.
- The combined per-foundation similarity is `max(virtue_sim, vice_sim)` -
  a sentence framed in moral terms of Care counts as Care whether it
  expresses compassion (virtue) or cruelty (vice).
- A `mft_dominant_pole` column records whether the dominant foundation's
  signal came from the virtue or vice centroid.

**Output:** `outputs/intermediate/mft_sentence_scores.parquet`.

## Step 5 - Aggregation and figures (`05_build_profiles.py`)

For each outlet:

- **Salience filter.** Sentences below the per-outlet, per-framework median
  salience are dropped before aggregation. This removes the bottom 50% of
  sentences - the neutral/factual ones that don't morally frame anything
  and don't anchor to any value.
- **Per-outlet profile.** For each value/foundation:
    - `mean_similarity` - average cosine sim across retained sentences
    - `pct_dominant` - share of retained sentences whose dominant label is this category
    - `sentence_count_dominant` - raw count of dominant-label sentences
    - (MFT only) `pct_virtue_of_dominant`, `pct_vice_of_dominant`
- **Comparison tables.** `pct_dominant` is the headline number, pivoted into
  one table per framework with 9 (or 5) rows × 3 outlet columns.

**Outputs:** `outputs/profiles/` (CSVs), `outputs/figures/` (paired PNGs),
`outputs/diagnostics/` (salience cutoffs and exemplar coverage counts).

## Step 6 - Exemplar stability test (`06_exemplar_stability_test.py`)

Drafted exemplar pools max out at 50 per category, but the production
pipeline does not necessarily use all 50. The stability test picks the
smallest size that produces stable rankings:

1. Run scoring at sizes [15, 25, 35, 50] (MFT halved per pole: 7+8, 12+13,
   17+18, 25+25 virtue/vice respectively).
2. For each consecutive pair (e.g., 15→25), compute Spearman rank
   correlation between per-outlet category rankings.
3. **Primary rule:** pick the smallest size where the mean Spearman across
   the 3 outlets to the next-larger size is ≥ 0.95.
4. **Secondary check:** whether the top-3 categories per outlet are
   unchanged between that size and the next-larger size. Reported but not
   used as the decision rule.
5. **Fallback:** if no consecutive pair reaches ≥ 0.95, use the largest
   tested size (50) and flag non-convergence in
   `outputs/stability/README.md`.

The chosen sizes are written to `outputs/stability/active_sizes.json` and
should be copied into `scripts/utils.py` (`ACTIVE_SCHWARTZ_SIZE`,
`ACTIVE_MFT_SIZE`) before the production scoring run.

**Sampling note.** Sub-50 sizes are drawn by random selection from the
full pool with `random_state=42`. This avoids any bias from the author's
ordering of exemplars within the JSON file. The seed is fixed so all runs
at the same size produce the same centroid.

## Optional - Topic discovery (`optional_topic_discovery.py`)

Not part of the core psychographic pipeline. Provides a complementary
view of WHAT each outlet is covering using:

- SBERT article embeddings → UMAP (15 components, cosine, 15 neighbors,
  min_dist=0.0, random_state=42) → HDBSCAN (per-outlet hyperparameters).
- TF-IDF on bigrams (English + Tagalog filler stopwords removed) for
  cluster labeling: top three bigrams per cluster become the label.
- 2D UMAP scatter plot per outlet for visualization.

**Outputs:** `outputs/topics/clusters_<outlet>.csv` and `.png`. The
psychographic profile is built directly from sentences and does not depend
on this step.

## Key design choices

- **Sentence-level scoring** (not article- or cluster-level). Long articles
  contain mixed signals; a sentence is the unit where moral framing is
  unambiguous.
- **Exemplar centroids** (not keyword lists). Captures meaning, not surface
  vocabulary. "The president ordered troops to protect the border" scores
  as Authority not Care, because the sentence's meaning is closest to
  Authority exemplars.
- **Soft scores preserved** (not winner-takes-all). Every sentence's full
  9-Schwartz + 5-MFT cosine vector is shipped in
  `outputs/profiles/sentence_scores.parquet` so researchers can re-aggregate
  with different rules (e.g., soft assignment, multi-label thresholds)
  without rerunning embeddings.
- **Median salience filter** drops neutral sentences but is symmetric across
  outlets - the absolute cutoff varies by outlet, the *retention rate*
  (50%) does not.

## Reproducibility

All stochastic steps (`random_state=42`):
- UMAP reduction
- HDBSCAN initialization
- Exemplar subsampling in the stability test and production scoring

SBERT embeddings are deterministic. Running the full pipeline twice on the
same data and reference files produces identical outputs.
