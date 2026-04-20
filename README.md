# Psychographic Profiling Pipeline

A framework for building psychographic profiles of media sources by analyzing the **values**, **moral framings**, and **psychological orientations** signaled in their published content.

Built for a capstone project analyzing three Philippine news outlets (ABS-CBN, GMA, Rappler), but designed to be applicable to any text corpus.

## What This Does

Instead of predicting individual reader personality, this pipeline characterizes the **psychological orientation that a media outlet projects and reinforces** through its editorial choices and language patterns.

Two layers of analysis:

1. **Schwartz Basic Human Values**: classifies article topic clusters into value categories (Security, Power, Achievement, Benevolence, etc.) to show **WHAT** each outlet prioritizes
2. **Moral Foundations Theory (MFT)**: scores content on 5 moral foundations (Care, Fairness, Loyalty, Authority, Sanctity) to show **HOW** each outlet morally frames what it covers

## Theoretical Grounding

- **Schwartz (1992, 2012)**: 10 universal value types validated across 80+ countries
- **Kern et al. (2019, PNAS)**: Demonstrated Schwartz values can be extracted from text and aggregated into group-level psychological profiles
- **Haidt & Graham (2007)**: Moral Foundations Theory — 5 moral foundations that shape how people construct right/wrong
- **Selective Exposure Theory**: Audiences self-select into media that matches their psychological orientation, so outlet content profiles reflect audience orientation

## Repository Structure

```
Psychographic-Profiling-Pipeline/
├── data/
│   ├── abs/              # ABS-CBN cleaned articles (JSONL)
│   ├── gma/              # GMA cleaned articles (JSONL)
│   └── rappler/          # Rappler cleaned articles (JSONL)
├── schwartz_mft_analysis/
│   └── run_schwartz_mft_analysis.py   # Schwartz + MFT combined pipeline
├── run_psychographic_profile.py        # Feature-based psychographic profiling
├── run_cluster_psychographic_mapping.py # Cluster-to-Schwartz-value mapping + MFT
├── requirements.txt
└── README.md
```

## Scripts

### `run_psychographic_profile.py`
Extracts text-level features (NRC emotions, VADER sentiment, MFD moral foundations, Empath categories, linguistic complexity, pronoun ratios) and maps them to psychological constructs (Security Orientation, Authority Orientation, Justice/Reform, Communal Orientation, Threat Sensitivity, etc.). Produces radar charts and per-outlet narrative profiles.

### `run_cluster_psychographic_mapping.py`
Clusters articles using SBERT embeddings + UMAP + HDBSCAN, classifies each cluster into a Schwartz value category, then scores each cluster on MFT foundations. Produces heatmaps and value distribution charts.

### `schwartz_mft_analysis/run_schwartz_mft_analysis.py`
The combined Schwartz + MFT pipeline. Clusters articles, classifies into Schwartz values, then runs MFT scoring **within each value** to show how different outlets morally frame the same values differently. Produces per-outlet psychographic profiles, radar charts, and comparative visualizations.

## Data Format

Each data file is a JSONL (JSON Lines) file where each line is a JSON object with:
- `headline`: article headline
- `text`: article body text
- Other metadata fields vary by source

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## How to Run

Each script is self-contained and can be run directly:

```bash
# Feature-based psychographic profiling
python run_psychographic_profile.py

# Cluster-to-Schwartz mapping with MFT
python run_cluster_psychographic_mapping.py

# Combined Schwartz + MFT analysis
python schwartz_mft_analysis/run_schwartz_mft_analysis.py
```

Outputs (charts, CSVs, profile narratives) are saved to `psychographic_profiles/` or `schwartz_mft_analysis/` directories.

## Key Findings

Each outlet has a distinct psychographic profile:

| Outlet | Top Schwartz Values | Psychographic Signal |
|--------|-------------------|---------------------|
| **ABS-CBN** | Power (30%) > Benevolence (28%) > Security (20%) | Institutional dependence, moral protectiveness, threat vigilance |
| **GMA** | Benevolence (29%) > Security (28%) > Power (14%) | Institutional trust, managed threat awareness |
| **Rappler** | Achievement (26%) > Benevolence (18%) > Power (17%) | Aspiration, collective pride, critical awareness |

MFT analysis reveals that outlets frame the **same values through different moral lenses** — e.g., both ABS-CBN and GMA cover Security, but ABS-CBN frames it through Care (human suffering) while GMA frames it through Authority (institutional response).

## References

- Schwartz, S. H. (1992). Universals in the content and structure of values.
- Schwartz, S. H. (2012). An Overview of the Schwartz Theory of Basic Values.
- Kern, M. L., et al. (2019). Social media-predicted personality traits and values can help match people to their ideal jobs. *PNAS*.
- Haidt, J., & Graham, J. (2007). When morality opposes justice: Conservatives have moral intuitions that liberals may not recognize.
- Graham, J., et al. (2013). Moral Foundations Theory: The pragmatic validity of moral pluralism.
