# Archive

Earlier iterations of the project. None of these are required to reproduce
the current outlet-level psychographic profiles - the core pipeline lives in
`scripts/01_clean.py` through `scripts/05_build_profiles.py`. These files
are kept for reference and for researchers who want to see how the methodology
evolved.

## Contents

### `run_psychographic_profile.py`
Original end-to-end profiler. Used lexicon-based scoring (NRCLex, Empath,
VADER) plus feature aggregation per article. Superseded by the SBERT-based
sentence-level approach in the current pipeline because it (a) relied on
multiple overlapping psych lexicons, (b) aggregated only at the article level,
and (c) had no transparent reference set researchers could edit.

### `run_cluster_psychographic_mapping.py`
Cluster-centric pipeline: SBERT → UMAP → HDBSCAN → TF-IDF → Schwartz
keyword-match scoring. Each cluster received one Schwartz value (winner-
takes-all) based on keyword counts. Useful for topic discovery but mixes
WHAT-is-covered with HOW-it-is-framed at the cluster level. The new pipeline
scores Schwartz at the sentence level using exemplar centroids, which both
sharpens the signal and removes the cluster-as-a-dependency. The clustering
work itself is preserved in `scripts/optional_topic_discovery.py`.

### `schwartz_mft_analysis/run_schwartz_mft_analysis.py`
Mid-iteration script that paired Schwartz keyword classification with article-
level MFT scoring. The MFT half of this script seeded the sentence-level
exemplar approach now in `scripts/04_score_mft.py`; the Schwartz half has
been replaced by exemplar-based scoring in `scripts/03_score_schwartz.py`.

## Why keep this around

These scripts contain the keyword lists, cluster parameters, and Schwartz
value definitions that informed the current pipeline. If a researcher wants
to compare keyword-based vs exemplar-based scoring or rebuild the cluster
view with different hyperparameters, this is the reference.
