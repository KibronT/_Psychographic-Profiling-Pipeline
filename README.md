# Psychographic Profiling Pipeline

A reproducible pipeline that builds **outlet-level psychographic profiles**
from news article corpora by combining two psychological frameworks:

1. **Schwartz Basic Human Values** - WHAT each outlet emphasizes
2. **Moral Foundations Theory (MFT)** - HOW each outlet morally frames its content

Both layers are scored at the **sentence level** using SBERT and hand-authored
reference sentences (exemplars), then aggregated up to the outlet.

Built for a capstone project analyzing 2024 coverage from three Philippine
news outlets - ABS-CBN, GMA, and Rappler - but applicable to any text corpus
with minor reference-set adjustments.

## Repository layout

```
.
├── data/                              # input: per-outlet cleaned JSONL
│   ├── abs/  gma/  rappler/
├── reference/                         # exemplar sentences (researcher-editable)
│   ├── schwartz_exemplars.json        # 9 values, 50 sentences each
│   └── mft_exemplars.json             # 5 foundations, 25 virtue + 25 vice each
├── scripts/
│   ├── utils.py                       # paths + active exemplar sizes
│   ├── 01_clean.py                    # load JSONL, normalize, filter
│   ├── 02_embed_sentences.py          # sentence split + SBERT embed
│   ├── 03_score_schwartz.py           # cosine vs Schwartz centroids
│   ├── 04_score_mft.py                # cosine vs MFT centroids
│   ├── 05_build_profiles.py           # aggregate + CSVs + figures
│   ├── 06_exemplar_stability_test.py  # pick smallest stable exemplar size
│   └── optional_topic_discovery.py    # UMAP+HDBSCAN+TF-IDF clusters (side artifact)
├── outputs/
│   ├── RESULTS.md                     # final results showcase (tables + figures)
│   ├── profiles/                      # CSVs (per-outlet + comparison tables)
│   ├── figures/                       # paired PNG visualizations
│   ├── diagnostics/                   # salience cutoffs + exemplar coverage
│   ├── stability/                     # stability test results
│   ├── topics/                        # optional clustering output
│   └── intermediate/                  # cached parquet + npy artifacts
├── docs/                              # METHODS, DATA_DICTIONARY, LIMITATIONS, UPDATING_REFERENCES, VALIDATION_CHECK
├── archive/                           # earlier iterations (not part of core pipeline)
├── requirements.txt
└── README.md
```

## Quick start

The pipeline ships with `ACTIVE_SCHWARTZ_SIZE = 50` and `ACTIVE_MFT_SIZE = 50`
already set in `scripts/utils.py` based on the stability test in this run.
Run in order; each script reads from the previous one's output.

```bash
pip install -r requirements.txt

python scripts/01_clean.py
python scripts/02_embed_sentences.py          # ~5-10 min on CPU
python scripts/03_score_schwartz.py
python scripts/04_score_mft.py
python scripts/05_build_profiles.py

# Optional: rerun the stability test if you change the reference pool.
python scripts/06_exemplar_stability_test.py

# Optional topic-discovery view (not part of the psychographic profile).
python scripts/optional_topic_discovery.py
```

## What you get

- **`outputs/profiles/comparison_schwartz.csv`** - 9 values × 3 outlets, the headline numbers
- **`outputs/profiles/comparison_mft.csv`** - 5 foundations × 3 outlets, the headline numbers
- **`outputs/profiles/sentence_scores.parquet`** - every sentence's full score vector (re-aggregate without rerunning)
- **`outputs/figures/`** - grouped bar + radar charts for both frameworks, plus MFT virtue/vice split
- **`outputs/stability/`** - empirical justification for the exemplar size used

Per-outlet CSVs (`schwartz_profile_<outlet>.csv`, `mft_profile_<outlet>.csv`)
sit alongside the comparison tables for outlet-by-outlet inspection.

## Theoretical grounding

- **Schwartz, S. H. (1992, 2012)** - 10 universal value types validated across 80+ countries; this pipeline uses 9 (excludes Hedonism, see [reference/schwartz_exemplars.json](reference/schwartz_exemplars.json))
- **Haidt, J. & Graham, J. (2007); Graham et al. (2013)** - 5 moral foundations
- **Kern et al. (2019, PNAS)** - group-level value extraction from text
- **Selective exposure theory** - audiences self-select into media that matches their psychological orientation, so outlet content profiles serve as a proxy for audience orientation

## Editing the reference sets

The exemplar JSONs in `reference/` are designed to be edited by researchers.
See [`docs/UPDATING_REFERENCES.md`](docs/UPDATING_REFERENCES.md) for the workflow.

## Documentation

- [`outputs/RESULTS.md`](outputs/RESULTS.md) - final results with tables and figures
- [`docs/METHODS.md`](docs/METHODS.md) - full methodology
- [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) - every output column explained
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) - known caveats and design tradeoffs
- [`docs/UPDATING_REFERENCES.md`](docs/UPDATING_REFERENCES.md) - how to add/remove exemplars or values
- [`docs/VALIDATION_CHECK.md`](docs/VALIDATION_CHECK.md) - pre-handoff audit, refinement deltas, size decisions
- [`archive/README.md`](archive/README.md) - earlier iterations of the work
