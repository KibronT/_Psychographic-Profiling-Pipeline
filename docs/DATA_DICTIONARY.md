# Data Dictionary

Every column in every output file, in the order you'll encounter them.

## `outputs/intermediate/articles.parquet`

One row per article (post-cleaning, post-min-words filter).

| Column        | Type   | Description                                                              |
|---------------|--------|--------------------------------------------------------------------------|
| `article_id`  | str    | `a000000` … `a002788` - assigned in row order across all outlets         |
| `outlet`      | str    | One of `abs`, `gma`, `rappler`                                           |
| `headline`    | str    | Normalized headline                                                      |
| `text`        | str    | Body text after footer-stripping and whitespace normalization            |
| `full_text`   | str    | `headline + ". " + text` - the unit that gets sentence-split downstream  |
| `word_count`  | int    | Words in `full_text`                                                     |
| `source_file` | str    | Source JSONL filename (for traceability)                                 |

## `outputs/intermediate/sentences.parquet`

One row per sentence.

| Column          | Type | Description                                          |
|-----------------|------|------------------------------------------------------|
| `sentence_id`   | str  | `s0000000` … assigned in row order                   |
| `article_id`    | str  | Foreign key to `articles.parquet`                    |
| `outlet`        | str  | Outlet (denormalized for convenience)                |
| `sentence_text` | str  | The sentence string                                  |

Row-aligned with `outputs/intermediate/sentence_embeddings.npy` (float32,
N × 384, L2-normalized).

## `outputs/intermediate/schwartz_sentence_scores.parquet`

| Column                  | Type  | Description                                                              |
|-------------------------|-------|--------------------------------------------------------------------------|
| `sentence_id`           | str   | Foreign key                                                              |
| `schwartz_Security`     | float | Cosine similarity to the Security centroid (range −1 to 1)               |
| `schwartz_Power`        | float | Cosine similarity to the Power centroid                                  |
| `schwartz_Achievement`  | float | …                                                                        |
| `schwartz_Benevolence`  | float |                                                                          |
| `schwartz_Universalism` | float |                                                                          |
| `schwartz_Stimulation`  | float |                                                                          |
| `schwartz_Tradition`    | float |                                                                          |
| `schwartz_Self-Direction` | float |                                                                        |
| `schwartz_Conformity`   | float |                                                                          |
| `schwartz_dominant`     | str   | Argmax: name of the value with the highest similarity                    |
| `schwartz_salience`     | float | Max similarity across the 9 values; used by the median salience filter   |

## `outputs/intermediate/mft_sentence_scores.parquet`

| Column                         | Type  | Description                                                                  |
|--------------------------------|-------|------------------------------------------------------------------------------|
| `sentence_id`                  | str   | Foreign key                                                                  |
| `mft_<foundation>`             | float | Combined similarity = max(virtue, vice) for the foundation                   |
| `mft_<foundation>_virtue`      | float | Cosine similarity to the virtue centroid for the foundation                  |
| `mft_<foundation>_vice`        | float | Cosine similarity to the vice centroid for the foundation                    |
| `mft_dominant`                 | str   | Argmax across combined similarities (care/fairness/loyalty/authority/sanctity)|
| `mft_salience`                 | float | Max combined similarity; used by the median salience filter                  |
| `mft_dominant_pole`            | str   | `virtue` or `vice` - which pole produced the dominant signal                 |

`<foundation>` ∈ {`care`, `fairness`, `loyalty`, `authority`, `sanctity`}.

## `outputs/profiles/sentence_scores.parquet`

The joined view: sentences merged with both score files. Same columns as
`sentences.parquet` + all Schwartz columns + all MFT columns. This is the
file to use if you want to re-aggregate with custom rules.

## `outputs/profiles/schwartz_profile_<outlet>.csv`

One row per Schwartz value (9 rows). Calculated on the post-salience-filter
sentence set (top 50% by Schwartz salience).

| Column                    | Type  | Description                                                  |
|---------------------------|-------|--------------------------------------------------------------|
| `value`                   | str   | One of the 9 Schwartz values                                 |
| `mean_similarity`         | float | Average cosine sim across retained sentences                 |
| `pct_dominant`            | float | % of retained sentences whose dominant value is this row     |
| `sentence_count_dominant` | int   | Raw count behind `pct_dominant`                              |

## `outputs/profiles/mft_profile_<outlet>.csv`

One row per MFT foundation (5 rows). Calculated on the post-salience-filter
sentence set (top 50% by MFT salience).

| Column                    | Type  | Description                                                              |
|---------------------------|-------|--------------------------------------------------------------------------|
| `foundation`              | str   | One of care/fairness/loyalty/authority/sanctity                          |
| `mean_similarity`         | float | Average combined similarity across retained sentences                    |
| `pct_dominant`            | float | % of retained sentences whose dominant foundation is this row            |
| `sentence_count_dominant` | int   | Raw count behind `pct_dominant`                                          |
| `pct_virtue_of_dominant`  | float | Among dominant-this-foundation sentences, % whose dominant pole is virtue|
| `pct_vice_of_dominant`    | float | Among dominant-this-foundation sentences, % whose dominant pole is vice  |

## `outputs/profiles/comparison_schwartz.csv` / `comparison_mft.csv`

Pivot of `pct_dominant`: rows are categories, columns are outlet display
names (`ABS-CBN`, `GMA`, `Rappler`). This is the headline comparison
table.

## `outputs/diagnostics/salience_distribution.csv`

| Column                       | Type  | Description                                                         |
|------------------------------|-------|---------------------------------------------------------------------|
| `outlet`                     | str   |                                                                     |
| `n_sentences`                | int   | Total sentences for this outlet                                     |
| `schwartz_salience_cutoff`   | float | Median Schwartz salience for this outlet (filter threshold)         |
| `schwartz_retained`          | int   | Sentences retained after the Schwartz salience filter               |
| `mft_salience_cutoff`        | float | Median MFT salience for this outlet                                 |
| `mft_retained`               | int   | Sentences retained after the MFT salience filter                    |

## `outputs/diagnostics/exemplar_coverage.csv`

| Column            | Type | Description                                                                    |
|-------------------|------|--------------------------------------------------------------------------------|
| `outlet`          | str  |                                                                                |
| `framework`       | str  | `schwartz` or `mft`                                                            |
| `category`        | str  | Value or foundation name                                                       |
| `n_dominant_all`  | int  | Sentences with this category as dominant - **before** any salience filter      |

Use this to spot under-represented categories - if a value's `n_dominant_all`
is near zero in all outlets, its exemplars may not be discriminating well
from neighboring values.

## `outputs/stability/stability_results.csv`

| Column                | Type   | Description                                                       |
|-----------------------|--------|-------------------------------------------------------------------|
| `framework`           | str    | `schwartz` or `mft`                                               |
| `size_a`, `size_b`    | int    | Consecutive sizes compared (e.g., 15 vs 25)                       |
| `mean_spearman`       | float  | Mean Spearman rank correlation across the 3 outlets               |
| `min_spearman`        | float  | Minimum across the 3 outlets                                      |
| `spearman_<outlet>`   | float  | Per-outlet Spearman for each of `abs`, `gma`, `rappler`           |
| `all_top3_unchanged`  | bool   | Whether the top-3 categories per outlet are identical between sizes|

## `outputs/stability/active_sizes.json`

```json
{ "ACTIVE_SCHWARTZ_SIZE": <int>, "ACTIVE_MFT_SIZE": <int> }
```

Copy these into `scripts/utils.py` before rerunning `03_score_schwartz.py`
and `04_score_mft.py`.

## `outputs/topics/clusters_<outlet>.csv`  (optional)

One row per cluster found by HDBSCAN.

| Column        | Type | Description                                          |
|---------------|------|------------------------------------------------------|
| `cluster_id`  | int  | HDBSCAN label; `-1` = noise                          |
| `n_articles`  | int  | Articles in this cluster                             |
| `tfidf_label` | str  | Top 3 TF-IDF bigrams, comma-separated                |
