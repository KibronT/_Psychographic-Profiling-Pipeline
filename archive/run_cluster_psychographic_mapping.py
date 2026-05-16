"""
Cluster → Psychographic Construct Mapping
==========================================
This script takes the existing HDBSCAN topic clusters and does two things:

1. RELABELS clusters from raw topic labels (e.g., "Drug War", "Coast Guard")
   into Schwartz Value categories + psychographic construct tags, showing
   what each cluster reveals about the outlet's orientation

2. SCORES each cluster on psychographic constructs using MFD, Empath, and
   emotion features -- so we can see which constructs each outlet's content
   clusters map to

This bridges: Clustering (what topics exist) → Psychographic Profiling (what
those topics signal about reader orientation)

Theoretical grounding:
  - Schwartz Basic Human Values: topic clusters map to value priorities
    (Kern et al., 2019 used this for occupation-level profiling)
  - Moral Foundations Theory: within-cluster moral language reveals framing
  - Agenda-setting theory: what an outlet covers = what it signals as important
"""

import json
import glob
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
import hdbscan
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "psychographic_profiles")
os.makedirs(OUT_DIR, exist_ok=True)

SOURCES = {
    "abs": {
        "name": "ABS-CBN",
        "data_dir": os.path.join(CLEANED_DIR, "abs"),
        "hdbscan_min_cluster_size": 5,
        "hdbscan_min_samples": 2,
        "hdbscan_method": "leaf",
    },
    "gma": {
        "name": "GMA",
        "data_dir": os.path.join(CLEANED_DIR, "gma"),
        "hdbscan_min_cluster_size": 8,
        "hdbscan_min_samples": 3,
        "hdbscan_method": "leaf",
    },
    "rappler": {
        "name": "Rappler",
        "data_dir": os.path.join(CLEANED_DIR, "rappler"),
        "hdbscan_min_cluster_size": 8,
        "hdbscan_min_samples": 3,
        "hdbscan_method": "leaf",
    },
}

vader = SentimentIntensityAnalyzer()
empath_lexicon = Empath()

NRC_EMOTIONS = ["anger", "anticipation", "disgust", "fear",
                "joy", "sadness", "surprise", "trust", "positive", "negative"]

# UMAP settings (match existing pipeline)
UMAP_N_COMPONENTS = 15
UMAP_METRIC = "cosine"
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0

# Words to filter from cluster labels
STOP_LABEL_WORDS = {
    "said", "says", "also", "would", "could", "one", "two", "new",
    "year", "like", "told", "added", "according", "based", "get",
    "may", "last", "first", "made", "back", "still", "just", "even",
    "gma", "news", "online", "rappler", "abs", "cbn", "philippine",
    "philippines", "filipino", "na", "sa", "ng", "ang", "mga",
    "niya", "para", "ni", "si", "ako", "ko", "po",
}

# ══════════════════════════════════════════════════════════════
# SCHWARTZ VALUE MAPPING
# ══════════════════════════════════════════════════════════════
# Keywords that signal each Schwartz value when they appear in
# cluster topic labels or article content. Used to classify
# what value each topic cluster maps to.

SCHWARTZ_VALUE_KEYWORDS = {
    "Security": [
        "coast guard", "police", "military", "drug war", "drug test",
        "illegal drug", "crime", "kill", "murder", "gun ban", "gun",
        "detention", "arrest", "bomb", "terrorism", "terrorist",
        "weapon", "war ", "attack", "threat", "violence", "violent",
        "safety", "security", "guard", "rescue", "disaster", "storm",
        "typhoon", "tropical storm", "earthquake", "volcano", "fire",
        "flood", "landslide", "evacuat", "casualt", "death", "dead",
        "offshore gaming", "smuggl", "kidnap", "hostage", "rebel",
        "insurgent", "islamic state",
    ],
    "Power": [
        "president", "senate", "congress", "governor", "mayor",
        "vice president", "administr", "government", "official",
        "budget", "national budget", "billion", "income tax",
        "election", "campaign", "political", "marcos", "duterte",
        "party", "coalition", "appointment", "cabinet",
    ],
    "Universalism": [
        "justice", "rights", "reform", "equality", "fair", "freedom",
        "fact check", "protest", "rally", "activist", "environment",
        "climate", "indigenous", "minority", "sexual minorities",
        "migrant worker", "refugee", "human rights", "gaza",
        "peace", "nobel peace", "cafeteria catholic",
    ],
    "Achievement": [
        "olympic", "championship", "university", "examination",
        "licensure", "grand slam", "medal", "award", "miss universe",
        "binibining", "contest", "win", "winner", "champion",
        "athlete", "sport", "basketball", "boxing",
        "income", "net income", "employed", "job",
    ],
    "Benevolence": [
        "health", "hospital", "department health", "welfare",
        "education", "department education", "school", "teacher",
        "family", "children", "child", "community", "volunteer",
        "help", "aid", "relief", "donation", "charity",
        "church", "quiapo church", "holy week", "prayer",
        "bipolar disorder", "mental health",
    ],
    "Stimulation": [
        "entertainment", "film", "movie", "music", "concert",
        "art fair", "festival", "celebrity", "actor", "actress",
        "live jam", "eat bulaga", "fast talk", "dantes",
        "chocolate", "recipe", "food", "restaurant",
        "travel", "tourism", "mall",
    ],
    "Tradition": [
        "church", "religion", "faith", "holy", "easter",
        "christmas", "fiesta", "tradition", "heritage",
        "cultural", "patron saint", "quiboloy",
    ],
    "Self-Direction": [
        "technology", "digital", "internet", "sim", "online",
        "startup", "innovation", "science", "research",
        "charter change", "reform",
    ],
    "Conformity": [
        "law", "regulation", "mandate", "compliance",
        "ordinance", "land transportation", "birth certificate",
        "school days",
    ],
}

# ══════════════════════════════════════════════════════════════
# MFD (same as psychographic profile script)
# ══════════════════════════════════════════════════════════════
MFD = {
    "care.virtue": [
        "safe", "peace", "compassion", "empathy", "sympath", "care", "caring",
        "protect", "shield", "shelter", "secur", "benefit", "defen", "guard",
        "preserve", "nurtur", "comfort", "gentle", "kind", "tender", "welfare",
    ],
    "care.vice": [
        "harm", "suffer", "war ", "wars", "warl", "cruel", "brutal", "abuse",
        "damag", "ruin", "ravage", "crush", "attack", "annihilat", "destroy",
        "hurt", "kill", "violen", "victim", "exploit", "wound", "torture",
    ],
    "fairness.virtue": [
        "fair", "just", "justice", "equal", "equit", "reciproc", "impartial",
        "egalitar", "rights", "libert", "freedom", "honest", "lawful",
        "transparen", "accountab", "reform", "democra",
    ],
    "fairness.vice": [
        "unfair", "unjust", "injust", "bigot", "discrim", "inequit", "bias",
        "unscrupul", "exploit", "corrupt", "cheat", "fraud", "illegal",
        "unlaw", "nepotis", "oppres",
    ],
    "loyalty.virtue": [
        "loyal", "patriot", "fidelity", "allegian", "unite", "communal",
        "nation", "group", "collect", "together", "solidar", "devot",
        "fellow", "homeland", "countrymen", "filipino", "bayan",
    ],
    "loyalty.vice": [
        "betray", "treason", "disloyal", "traitor", "desert", "defect",
        "foreign", "enem", "alienat", "separati", "splinter",
    ],
    "authority.virtue": [
        "obey", "obedien", "duty", "law", "lawful", "legal", "order",
        "respect", "author", "tradition", "hierarch", "complian", "defer",
        "reveren", "permit", "rank", "leader", "official", "govern",
        "president", "administr", "regulat", "mandate",
    ],
    "authority.vice": [
        "subver", "disobey", "defy", "defian", "rebel", "dissent", "chaos",
        "anarch", "disorder", "disrespect", "riot", "protest", "agitat",
        "resist", "violat",
    ],
    "sanctity.virtue": [
        "church", "devout", "faith", "god", "holy", "pure", "purity",
        "sacred", "spirit", "wholesome", "innocent", "moral", "pious",
        "bless", "righteous", "worship", "pray", "religio",
    ],
    "sanctity.vice": [
        "sin", "profan", "gross", "repuls", "sick", "disgust", "pervert",
        "wicked", "indecen", "degrad", "defile", "immoral", "obscen",
        "filth", "drug", "addict",
    ],
}


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def load_source_data(data_dir):
    records = []
    files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    for fpath in sorted(files):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        records.append(rec)
                except json.JSONDecodeError:
                    pass
    return records


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*\|?\s*GMA News Online\s*$", "", text)
    return text


def get_article_text(rec):
    parts = []
    headline = clean_text(rec.get("headline", ""))
    body = clean_text(rec.get("text", ""))
    if headline:
        parts.append(headline)
    if body:
        parts.append(body)
    return " ".join(parts)


def get_nrc_scores(text):
    if not text:
        return {e: 0.0 for e in NRC_EMOTIONS}
    emotion_obj = NRCLex("init")
    emotion_obj.load_raw_text(text)
    raw = emotion_obj.raw_emotion_scores
    total = sum(raw.values()) if raw else 0
    return {e: (raw.get(e, 0) / total if total > 0 else 0.0) for e in NRC_EMOTIONS}


def get_mfd_scores(text):
    if not text:
        return {k: 0.0 for k in MFD}
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)
    if word_count == 0:
        return {k: 0.0 for k in MFD}
    scores = {}
    for foundation, stems in MFD.items():
        count = sum(1 for w in words for stem in stems if w.startswith(stem) or stem in w)
        scores[foundation] = count / word_count
    return scores


def classify_cluster_value(topic_label, article_texts):
    """
    Classify a cluster into a Schwartz value category based on its
    topic label AND article content.
    """
    label_lower = topic_label.lower()
    combined_text = " ".join(article_texts).lower()

    best_value = "Uncategorized"
    best_score = 0

    for value, keywords in SCHWARTZ_VALUE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            # Check topic label (high weight)
            if kw in label_lower:
                score += 10
            # Check article content (lower weight, but adds up)
            score += combined_text.count(kw) * 0.001
        if score > best_score:
            best_score = score
            best_value = value

    return best_value


def compute_cluster_constructs(cluster_features):
    """Compute psychographic constructs for a cluster's averaged features."""
    f = cluster_features

    constructs = {}

    constructs["Security Orientation"] = np.mean([
        f.get("nrc_fear", 0) * 5,
        f.get("mfd_care.vice", 0) * 50,
        f.get("empath_crime", 0) * 50,
        f.get("empath_violence", 0) * 50,
    ])

    constructs["Authority Orientation"] = np.mean([
        f.get("mfd_authority.virtue", 0) * 50,
        f.get("empath_government", 0) * 50,
        f.get("empath_law", 0) * 50,
        f.get("empath_power", 0) * 50,
    ])

    constructs["Justice/Reform"] = np.mean([
        f.get("mfd_fairness.virtue", 0) * 50,
        f.get("mfd_fairness.vice", 0) * 50,
        f.get("empath_justice", 0) * 50,
    ])

    constructs["Communal Orientation"] = np.mean([
        f.get("mfd_care.virtue", 0) * 50,
        f.get("mfd_loyalty.virtue", 0) * 50,
        f.get("empath_help", 0) * 50,
        f.get("empath_family", 0) * 50,
        f.get("empath_community", 0) * 50,
    ])

    constructs["Threat Sensitivity"] = np.mean([
        f.get("nrc_fear", 0) * 5,
        f.get("nrc_anger", 0) * 5,
        f.get("mfd_care.vice", 0) * 50,
        f.get("vader_neg", 0) * 5,
        f.get("empath_suffering", 0) * 50,
    ])

    constructs["Hope/Opportunity"] = np.mean([
        f.get("nrc_joy", 0) * 5,
        f.get("nrc_trust", 0) * 5,
        f.get("vader_pos", 0) * 5,
        f.get("empath_optimism", 0) * 50,
    ])

    return constructs


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("CLUSTER → PSYCHOGRAPHIC CONSTRUCT MAPPING")
    print("Bridging: Topic Clusters → Schwartz Values → Psychographic Profiles")
    print("=" * 70)

    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    all_cluster_rows = []
    source_value_distributions = {}

    for key, cfg in SOURCES.items():
        source_name = cfg["name"]
        print(f"\n{'─' * 70}")
        print(f"Processing: {source_name}")
        print(f"{'─' * 70}")

        # Load articles
        records = load_source_data(cfg["data_dir"])
        texts = [get_article_text(r) for r in records]
        texts = [t for t in texts if t and len(t.split()) >= 20]
        print(f"  {len(texts)} articles loaded")

        # Embed + cluster (reproduce existing pipeline)
        print("  Computing SBERT embeddings...")
        embeddings = sbert.encode(texts, show_progress_bar=False)

        print("  UMAP reduction...")
        reducer = umap.UMAP(
            n_components=UMAP_N_COMPONENTS,
            metric=UMAP_METRIC,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            random_state=42,
        )
        reduced = reducer.fit_transform(embeddings)

        print("  HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg["hdbscan_min_cluster_size"],
            min_samples=cfg["hdbscan_min_samples"],
            cluster_selection_method=cfg["hdbscan_method"],
        )
        labels = clusterer.fit_predict(reduced)

        unique_labels = sorted(set(labels))
        n_clusters = len([l for l in unique_labels if l >= 0])
        n_noise = sum(1 for l in labels if l == -1)
        print(f"  {n_clusters} clusters, {n_noise} noise articles")

        # Generate topic labels via TF-IDF
        tfidf = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(2, 2)
        )
        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = tfidf.get_feature_names_out()

        # Process each cluster
        print(f"\n  {'Cluster':<30} {'Articles':>8}  {'Schwartz Value':<18} {'Top Construct'}")
        print(f"  {'─' * 90}")

        value_counts = {}

        for cid in unique_labels:
            if cid == -1:
                continue

            mask = (labels == cid)
            cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
            cluster_indices = [i for i in range(len(texts)) if mask[i]]

            if len(cluster_texts) < 3:
                continue

            # Topic label
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
            top_idx = cluster_tfidf.argsort()[-5:][::-1]
            bigrams = [feature_names[i] for i in top_idx]
            # Filter stop words
            filtered = []
            for bg in bigrams:
                words = bg.split()
                if not any(w in STOP_LABEL_WORDS for w in words):
                    filtered.append(bg)
            topic_label = filtered[0].title() if filtered else bigrams[0].title()

            # Schwartz value classification
            schwartz_value = classify_cluster_value(topic_label, cluster_texts[:10])

            value_counts[schwartz_value] = value_counts.get(schwartz_value, 0) + len(cluster_texts)

            # Feature extraction per cluster
            cluster_features = []
            for text in cluster_texts:
                feats = {}
                nrc = get_nrc_scores(text)
                feats.update({f"nrc_{k}": v for k, v in nrc.items()})
                vs = vader.polarity_scores(text)
                feats.update({f"vader_{k}": v for k, v in vs.items()})
                mfd = get_mfd_scores(text)
                feats.update({f"mfd_{k}": v for k, v in mfd.items()})
                emp = empath_lexicon.analyze(text, normalize=True) or {}
                for cat in ["crime", "violence", "war", "government", "law", "power",
                            "justice", "help", "family", "community", "suffering",
                            "optimism", "death", "leader", "military", "police",
                            "order", "protest", "liberty", "sympathy"]:
                    feats[f"empath_{cat}"] = emp.get(cat, 0.0)
                cluster_features.append(feats)

            cf_df = pd.DataFrame(cluster_features)
            cf_means = cf_df.mean().to_dict()

            # Compute constructs for this cluster
            constructs = compute_cluster_constructs(cf_means)
            top_construct = max(constructs, key=constructs.get)

            print(f"  {topic_label:<30} {len(cluster_texts):>8}  {schwartz_value:<18} {top_construct}")

            # Collect row
            row = {
                "source": source_name,
                "cluster_id": cid,
                "topic_label": topic_label,
                "schwartz_value": schwartz_value,
                "article_count": len(cluster_texts),
                "top_construct": top_construct,
            }
            row.update({f"construct_{k}": v for k, v in constructs.items()})
            # Add key emotions
            for e in ["anger", "fear", "joy", "trust", "sadness"]:
                row[f"nrc_{e}"] = cf_means.get(f"nrc_{e}", 0)
            row["vader_compound"] = cf_means.get("vader_compound", 0)
            all_cluster_rows.append(row)

        source_value_distributions[source_name] = value_counts

        # Print value distribution
        total_clustered = sum(value_counts.values())
        print(f"\n  SCHWARTZ VALUE DISTRIBUTION for {source_name}:")
        for value in sorted(value_counts, key=value_counts.get, reverse=True):
            count = value_counts[value]
            pct = count / total_clustered * 100
            bar = "█" * int(pct / 2)
            print(f"    {value:<20} {count:>5} articles ({pct:5.1f}%)  {bar}")

    # ── Save cluster-level data ──
    cluster_df = pd.DataFrame(all_cluster_rows)
    csv_path = os.path.join(OUT_DIR, "cluster_psychographic_mapping.csv")
    cluster_df.to_csv(csv_path, index=False)
    print(f"\n  Cluster mapping saved: {csv_path}")

    # ── Value Distribution Comparison Chart ──
    print(f"\n{'─' * 70}")
    print("Generating visualizations...")

    all_values = sorted(set(
        v for vc in source_value_distributions.values() for v in vc
    ))
    colors_map = {"ABS-CBN": "#e74c3c", "GMA": "#3498db", "Rappler": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(all_values))
    width = 0.25

    for i, (source, vc) in enumerate(source_value_distributions.items()):
        total = sum(vc.values())
        vals = [vc.get(v, 0) / total * 100 for v in all_values]
        color = colors_map.get(source, "#333")
        ax.bar(x + i * width, vals, width, label=source, color=color, alpha=0.85)

    ax.set_xlabel("Schwartz Value Category", fontsize=12)
    ax.set_ylabel("% of Clustered Articles", fontsize=12)
    ax.set_title("Issue Priorities as Schwartz Values\nWhat Each Outlet Emphasizes",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_values, fontsize=10, rotation=30, ha="right")
    ax.legend(fontsize=11)
    plt.tight_layout()
    chart_path = os.path.join(OUT_DIR, "schwartz_value_distribution.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"  Value distribution chart saved: {chart_path}")

    # ── Cluster construct heatmap per source ──
    construct_cols = [c for c in cluster_df.columns if c.startswith("construct_")]

    for source_name in cluster_df["source"].unique():
        sdf = cluster_df[cluster_df["source"] == source_name].copy()
        sdf = sdf.sort_values("article_count", ascending=False).head(15)

        if len(sdf) < 3:
            continue

        fig, ax = plt.subplots(figsize=(12, max(6, len(sdf) * 0.5)))
        data = sdf[construct_cols].values
        col_labels = [c.replace("construct_", "") for c in construct_cols]
        row_labels = [f"{r['topic_label']} [{r['schwartz_value']}]"
                      for _, r in sdf.iterrows()]

        im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(f"{source_name}: Cluster Psychographic Constructs\n(Top 15 clusters by size)",
                     fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Construct Score")
        plt.tight_layout()

        hmap_path = os.path.join(OUT_DIR, f"{source_name.lower().replace('-','')}_cluster_heatmap.png")
        plt.savefig(hmap_path, dpi=200)
        plt.close()
        print(f"  Heatmap saved: {hmap_path}")

    # ── Final comparative summary ──
    print(f"\n{'=' * 70}")
    print("COMPARATIVE CLUSTER-LEVEL FINDINGS")
    print(f"{'=' * 70}")

    for source, vc in source_value_distributions.items():
        total = sum(vc.values())
        top3 = sorted(vc.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join(f"{v} ({c/total*100:.0f}%)" for v, c in top3)
        print(f"\n  {source}: Top values = {top3_str}")

    print(f"\n{'=' * 70}")
    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
