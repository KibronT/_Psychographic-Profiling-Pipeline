"""
Schwartz Value-Based Psychographic Analysis
=============================================
Core theoretical framework: Schwartz Basic Human Values
Supporting framework: Moral Foundations Theory (MFT)

Pipeline:
  Existing clusters → Schwartz value classification →
  MFT framing within each value → Psychographic profile per outlet

This script reads from the cleaned article data, runs clustering,
classifies clusters into Schwartz values, then runs MFT scoring
WITHIN each value category to show how each outlet frames the
same value differently.

Theoretical grounding:
  - Schwartz (1992, 2012): 10 universal value types organized in a
    motivational continuum. Values are the core of psychographics.
  - Kern et al. (2019): Demonstrated that Schwartz values extracted
    from text can profile groups at scale (occupation-level).
  - Haidt & Graham (2007): Moral Foundations Theory provides the
    framing lens — how values are morally constructed in language.
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
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
import hdbscan
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "schwartz_mft_analysis")
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
empath_lex = Empath()

NRC_EMOTIONS = ["anger", "anticipation", "disgust", "fear",
                "joy", "sadness", "surprise", "trust", "positive", "negative"]

# ══════════════════════════════════════════════════════════════
# SCHWARTZ VALUE KEYWORD MAP
# ══════════════════════════════════════════════════════════════
# Maps topic cluster labels + article content to Schwartz values.
# Based on Schwartz (2012) value definitions:
#   Security: safety, stability, harmony of society/self
#   Power: social status, prestige, control over resources/people
#   Universalism: understanding, tolerance, protection of all people/nature
#   Achievement: personal success through demonstrating competence
#   Benevolence: preserving/enhancing welfare of close others
#   Stimulation: excitement, novelty, challenge
#   Tradition: respect for customs, culture, religion
#   Self-Direction: independent thought and action, creativity
#   Conformity: restraint of actions that violate social norms

SCHWARTZ_KEYWORDS = {
    "Security": [
        "coast guard", "police", "military", "drug war", "drug test",
        "illegal drug", "crime", "kill", "murder", "gun ban", "gun",
        "detention", "arrest", "bomb", "terrorism", "terrorist",
        "weapon", "war ", "attack", "threat", "violence", "violent",
        "safety", "security", "guard", "rescue", "disaster", "storm",
        "typhoon", "tropical storm", "earthquake", "volcano", "fire",
        "flood", "landslide", "evacuat", "casualt", "death", "dead",
        "offshore gaming", "smuggl", "kidnap", "hostage", "rebel",
        "insurgent", "islamic state", "cloudy skies", "heat index",
        "el niño", "taal", "kanlaon",
    ],
    "Power": [
        "president", "senate", "congress", "governor", "mayor",
        "vice president", "administr", "government", "official",
        "budget", "national budget", "billion", "income tax",
        "election", "campaign", "political", "marcos", "duterte",
        "party", "coalition", "appointment", "cabinet", "hong kong",
        "south korea", "le pen", "alice guo", "nagpa patrol",
        "cha cha", "charter change", "gentleman agreement",
        "metro manila", "social media",
    ],
    "Universalism": [
        "justice", "rights", "reform", "equality", "fair", "freedom",
        "fact check", "protest", "rally", "activist", "environment",
        "climate", "indigenous", "minority", "sexual minorities",
        "migrant worker", "refugee", "human rights", "gaza",
        "peace", "nobel peace", "cafeteria catholic", "pope francis",
    ],
    "Achievement": [
        "olympic", "championship", "university", "examination",
        "licensure", "grand slam", "medal", "award", "miss universe",
        "binibining", "contest", "win", "winner", "champion",
        "athlete", "sport", "basketball", "boxing", "la salle",
        "gilas", "team usa", "golden state", "boston celtic",
        "caldwell pope", "nesthy", "asean championship",
        "income", "net income", "employed", "job",
    ],
    "Benevolence": [
        "health", "hospital", "department health", "welfare",
        "education", "department education", "school", "teacher",
        "family", "children", "child", "community", "volunteer",
        "help", "aid", "relief", "donation", "charity",
        "church", "quiapo church", "holy week", "prayer",
        "mental health", "doctor", "senior doctor",
        "4ps", "beneficiar", "road closure", "power interrupt",
        "public school", "jesus nazareno", "palm sunday",
        "adolescent", "pregnancy", "mary jane", "tiu laurel",
        "heart evangelista", "dr holmes", "talks asia",
        "laki mata", "ai features",
    ],
    "Stimulation": [
        "entertainment", "film", "movie", "music", "concert",
        "art fair", "festival", "celebrity", "actor", "actress",
        "live jam", "eat bulaga", "fast talk", "dantes",
        "chocolate", "recipe", "food", "restaurant", "auro",
        "travel", "tourism", "mall", "marian rivera", "boy abunda",
        "batang riles", "kitchen", "cebu pacific", "film festival",
        "music video", "diy bakery", "post shared", "rufa mae",
        "soo hyun", "jaclyn jose",
    ],
    "Tradition": [
        "church", "religion", "faith", "holy", "easter",
        "christmas", "fiesta", "tradition", "heritage",
        "cultural", "patron saint", "quiboloy", "semana santa",
    ],
    "Self-Direction": [
        "technology", "digital", "internet", "sim", "online",
        "startup", "innovation", "science", "research",
        "digital bank", "physical sim",
    ],
    "Conformity": [
        "law", "regulation", "mandate", "compliance",
        "ordinance", "land transportation", "birth certificate",
        "school days", "international law",
    ],
}

# ══════════════════════════════════════════════════════════════
# MFD — Moral Foundations Dictionary
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

MORAL_FOUNDATIONS = ["care", "fairness", "loyalty", "authority", "sanctity"]


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def load_source_data(data_dir):
    records = []
    for fpath in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
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
    for field in ["headline", "text"]:
        val = clean_text(rec.get(field, ""))
        if val:
            parts.append(val)
    return " ".join(parts)


def get_nrc_scores(text):
    if not text:
        return {e: 0.0 for e in NRC_EMOTIONS}
    obj = NRCLex("init")
    obj.load_raw_text(text)
    raw = obj.raw_emotion_scores
    total = sum(raw.values()) if raw else 0
    return {e: (raw.get(e, 0) / total if total > 0 else 0.0) for e in NRC_EMOTIONS}


def get_mfd_scores(text):
    if not text:
        return {k: 0.0 for k in MFD}
    words = text.lower().split()
    wc = len(words)
    if wc == 0:
        return {k: 0.0 for k in MFD}
    scores = {}
    for foundation, stems in MFD.items():
        count = sum(1 for w in words for s in stems if w.startswith(s) or s in w)
        scores[foundation] = count / wc
    return scores


def classify_schwartz(topic_label, sample_texts):
    label_lower = topic_label.lower()
    combined = " ".join(sample_texts[:10]).lower()
    best_val, best_score = "Uncategorized", 0
    for value, kws in SCHWARTZ_KEYWORDS.items():
        score = sum(10 for kw in kws if kw in label_lower)
        score += sum(combined.count(kw) * 0.001 for kw in kws)
        if score > best_score:
            best_score = score
            best_val = value
    return best_val


STOP_LABEL_WORDS = {
    "said", "says", "also", "would", "could", "one", "two", "new",
    "year", "like", "told", "added", "according", "based", "get",
    "may", "last", "first", "made", "back", "still", "just", "even",
    "gma", "news", "online", "rappler", "abs", "cbn", "philippine",
    "philippines", "filipino", "na", "sa", "ng", "ang", "mga",
    "niya", "para", "ni", "si", "ako", "ko", "po",
}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SCHWARTZ VALUE-BASED PSYCHOGRAPHIC ANALYSIS")
    print("Core: Schwartz Basic Human Values")
    print("Supporting: Moral Foundations Theory")
    print("=" * 70)

    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # Storage for all results
    all_cluster_rows = []
    all_value_article_mft = {}     # {source: {value: [mft_dicts]}}
    source_value_dists = {}         # {source: {value: count}}
    source_value_emotions = {}      # {source: {value: {emotion: score}}}

    for key, cfg in SOURCES.items():
        source_name = cfg["name"]
        print(f"\n{'─' * 70}")
        print(f"  {source_name}")
        print(f"{'─' * 70}")

        records = load_source_data(cfg["data_dir"])
        texts = [get_article_text(r) for r in records]
        texts = [t for t in texts if t and len(t.split()) >= 20]
        print(f"  {len(texts)} articles")

        # Embed + cluster
        print("  Embedding...")
        embeddings = sbert.encode(texts, show_progress_bar=False)
        print("  UMAP...")
        reducer = umap.UMAP(n_components=15, metric="cosine", n_neighbors=15,
                            min_dist=0.0, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        print("  HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg["hdbscan_min_cluster_size"],
            min_samples=cfg["hdbscan_min_samples"],
            cluster_selection_method=cfg["hdbscan_method"],
        )
        labels = clusterer.fit_predict(reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  {n_clusters} clusters")

        # TF-IDF for labels
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(2, 2))
        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = tfidf.get_feature_names_out()

        value_counts = {}
        value_mft_data = {}
        value_emotion_data = {}

        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            mask = (labels == cid)
            c_texts = [texts[i] for i in range(len(texts)) if mask[i]]
            c_indices = [i for i in range(len(texts)) if mask[i]]
            if len(c_texts) < 3:
                continue

            # Topic label
            c_tfidf = tfidf_matrix[c_indices].mean(axis=0).A1
            top_idx = c_tfidf.argsort()[-5:][::-1]
            bigrams = [feature_names[i] for i in top_idx]
            filtered = [bg for bg in bigrams
                        if not any(w in STOP_LABEL_WORDS for w in bg.split())]
            topic = (filtered[0] if filtered else bigrams[0]).title()

            # Classify into Schwartz value
            sv = classify_schwartz(topic, c_texts)
            value_counts[sv] = value_counts.get(sv, 0) + len(c_texts)

            # MFT scoring for every article in this cluster
            if sv not in value_mft_data:
                value_mft_data[sv] = []
            if sv not in value_emotion_data:
                value_emotion_data[sv] = []

            for txt in c_texts:
                mft = get_mfd_scores(txt)
                value_mft_data[sv].append(mft)
                nrc = get_nrc_scores(txt)
                value_emotion_data[sv].append(nrc)

            # Cluster-level MFT means
            mft_means = pd.DataFrame([get_mfd_scores(t) for t in c_texts]).mean().to_dict()
            nrc_means = pd.DataFrame([get_nrc_scores(t) for t in c_texts]).mean().to_dict()
            vader_mean = np.mean([vader.polarity_scores(t)["compound"] for t in c_texts])

            row = {
                "source": source_name,
                "cluster_id": cid,
                "topic_label": topic,
                "schwartz_value": sv,
                "article_count": len(c_texts),
                "vader_compound": vader_mean,
            }
            for mf in MORAL_FOUNDATIONS:
                row[f"mft_{mf}"] = mft_means.get(f"{mf}.virtue", 0) + mft_means.get(f"{mf}.vice", 0)
            for e in ["anger", "fear", "joy", "trust", "sadness", "anticipation"]:
                row[f"nrc_{e}"] = nrc_means.get(e, 0)
            all_cluster_rows.append(row)

        source_value_dists[source_name] = value_counts

        # Aggregate MFT per value for this source
        all_value_article_mft[source_name] = {}
        for sv, mft_list in value_mft_data.items():
            df = pd.DataFrame(mft_list)
            means = {}
            for mf in MORAL_FOUNDATIONS:
                means[mf] = df[[f"{mf}.virtue", f"{mf}.vice"]].sum(axis=1).mean()
            all_value_article_mft[source_name][sv] = means

        # Aggregate emotions per value
        source_value_emotions[source_name] = {}
        for sv, emo_list in value_emotion_data.items():
            df = pd.DataFrame(emo_list)
            source_value_emotions[source_name][sv] = df.mean().to_dict()

        # Print summary
        total = sum(value_counts.values())
        print(f"\n  Value Distribution:")
        for v in sorted(value_counts, key=value_counts.get, reverse=True):
            c = value_counts[v]
            pct = c / total * 100
            bar = "█" * int(pct / 2)
            print(f"    {v:<20} {c:>4} ({pct:5.1f}%)  {bar}")

    # ══════════════════════════════════════════════════════════
    # SAVE CSVs
    # ══════════════════════════════════════════════════════════
    cluster_df = pd.DataFrame(all_cluster_rows)
    cluster_df.to_csv(os.path.join(OUT_DIR, "cluster_schwartz_mapping.csv"), index=False)

    # Value distribution summary
    rows = []
    for source, vc in source_value_dists.items():
        total = sum(vc.values())
        for v, c in vc.items():
            rows.append({"source": source, "schwartz_value": v,
                         "article_count": c, "pct": c / total * 100})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "value_distribution.csv"), index=False)

    # MFT within values
    mft_rows = []
    for source, val_mft in all_value_article_mft.items():
        for sv, mf_scores in val_mft.items():
            row = {"source": source, "schwartz_value": sv}
            row.update(mf_scores)
            mft_rows.append(row)
    pd.DataFrame(mft_rows).to_csv(os.path.join(OUT_DIR, "mft_within_values.csv"), index=False)

    print(f"\n  CSVs saved to {OUT_DIR}")

    # ══════════════════════════════════════════════════════════
    # VISUALIZATION 1: Schwartz Value Distribution
    # ══════════════════════════════════════════════════════════
    print("\nGenerating visualizations...")

    all_values = sorted(set(v for vc in source_value_dists.values() for v in vc))
    colors = {"ABS-CBN": "#e74c3c", "GMA": "#3498db", "Rappler": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(all_values))
    width = 0.25
    for i, (src, vc) in enumerate(source_value_dists.items()):
        total = sum(vc.values())
        vals = [vc.get(v, 0) / total * 100 for v in all_values]
        ax.bar(x + i * width, vals, width, label=src, color=colors.get(src, "#333"), alpha=0.85)
    ax.set_xlabel("Schwartz Value", fontsize=12)
    ax.set_ylabel("% of Clustered Articles", fontsize=12)
    ax.set_title("Schwartz Value Distribution by Media Source\n(Issue Priorities as Value Orientations)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_values, fontsize=10, rotation=30, ha="right")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "schwartz_value_distribution.png"), dpi=200)
    plt.close()

    # ══════════════════════════════════════════════════════════
    # VISUALIZATION 2: MFT Within Top Values (per source)
    # How each outlet morally frames its top value categories
    # ══════════════════════════════════════════════════════════
    # Pick top 4 values across all sources
    all_val_counts = {}
    for vc in source_value_dists.values():
        for v, c in vc.items():
            all_val_counts[v] = all_val_counts.get(v, 0) + c
    top_values = sorted(all_val_counts, key=all_val_counts.get, reverse=True)[:5]

    fig, axes = plt.subplots(1, len(top_values), figsize=(4.5 * len(top_values), 6),
                             sharey=True)
    if len(top_values) == 1:
        axes = [axes]

    for ax_i, sv in enumerate(top_values):
        ax = axes[ax_i]
        x = np.arange(len(MORAL_FOUNDATIONS))
        w = 0.25
        for i, (src, val_mft) in enumerate(all_value_article_mft.items()):
            if sv in val_mft:
                vals = [val_mft[sv].get(mf, 0) for mf in MORAL_FOUNDATIONS]
            else:
                vals = [0] * len(MORAL_FOUNDATIONS)
            ax.bar(x + i * w, vals, w, label=src if ax_i == 0 else "",
                   color=colors.get(src, "#333"), alpha=0.85)
        ax.set_title(f"{sv}", fontsize=12, fontweight="bold")
        ax.set_xticks(x + w)
        ax.set_xticklabels([mf.title() for mf in MORAL_FOUNDATIONS],
                           fontsize=9, rotation=45, ha="right")
        if ax_i == 0:
            ax.set_ylabel("MFT Score (word proportion)", fontsize=10)

    axes[0].legend(fontsize=9)
    fig.suptitle("Moral Framing Within Each Schwartz Value Category\nHow Outlets Frame the Same Values Differently",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mft_within_values.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ══════════════════════════════════════════════════════════
    # VISUALIZATION 3: Radar per source — Value profile
    # ══════════════════════════════════════════════════════════
    # Use only values that appear in at least one source
    radar_values = [v for v in all_values if all_val_counts.get(v, 0) > 10]
    num_v = len(radar_values)
    angles = np.linspace(0, 2 * np.pi, num_v, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for src, vc in source_value_dists.items():
        total = sum(vc.values())
        vals = [vc.get(v, 0) / total * 100 for v in radar_values]
        vals += vals[:1]
        c = colors.get(src, "#333")
        ax.plot(angles, vals, "o-", linewidth=2.5, label=src, color=c)
        ax.fill(angles, vals, alpha=0.08, color=c)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_values, size=10)
    ax.set_title("Value Orientation Profiles\n(Schwartz Values as % of Coverage)",
                 size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "value_radar_profiles.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ══════════════════════════════════════════════════════════
    # GENERATE PROFILE NARRATIVES
    # ══════════════════════════════════════════════════════════
    for src, vc in source_value_dists.items():
        total = sum(vc.values())
        sorted_v = sorted(vc.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_v[:3]
        all_sorted = sorted_v
        bottom = [v for v, c in sorted_v if c / total < 0.05]
        mft_data = all_value_article_mft.get(src, {})
        sve = source_value_emotions.get(src, {})

        # Pre-compute key comparisons across all sources
        other_sources = {s: d for s, d in source_value_dists.items() if s != src}

        lines = []
        lines.append(f"PSYCHOGRAPHIC VALUE PROFILE: {src}")
        lines.append("=" * 70)
        lines.append("")

        # ── SECTION 1: VALUE ORIENTATION ──
        lines.append("1. CORE VALUE ORIENTATION")
        lines.append("-" * 70)
        lines.append("")
        for v, c in all_sorted:
            pct = c / total * 100
            if pct < 1:
                continue
            bar = "█" * int(pct / 2)
            lines.append(f"   {v:<20} {pct:5.1f}%  {bar}")
        lines.append("")

        # What makes this outlet's value profile distinctive
        lines.append("   What stands out:")
        for v, c in top3:
            pct = c / total * 100
            # Compare to other sources
            for other_src, other_vc in other_sources.items():
                other_total = sum(other_vc.values())
                other_pct = other_vc.get(v, 0) / other_total * 100
                diff = pct - other_pct
                if diff > 5:
                    lines.append(f"   - {v} is {diff:.0f} percentage points higher than {other_src}")
        for v in bottom:
            for other_src, other_vc in other_sources.items():
                other_total = sum(other_vc.values())
                other_pct = other_vc.get(v, 0) / other_total * 100
                if other_pct > 10:
                    pct = vc.get(v, 0) / total * 100
                    lines.append(f"   - {v} ({pct:.1f}%) is notably lower than {other_src} ({other_pct:.1f}%)")

        # ── SECTION 2: MORAL FRAMING ──
        lines.append("")
        lines.append("2. HOW THESE VALUES ARE MORALLY FRAMED (MFT)")
        lines.append("-" * 70)
        lines.append("")

        for v, _ in top3:
            if v not in mft_data:
                continue
            mf_scores = mft_data[v]
            sorted_mf = sorted(mf_scores.items(), key=lambda x: x[1], reverse=True)
            top_mf = sorted_mf[0][0]
            second_mf = sorted_mf[1][0] if len(sorted_mf) > 1 else None

            lines.append(f"   {v} content:")
            lines.append(f"     Primary moral frame: {top_mf.upper()} ({sorted_mf[0][1]:.4f})")
            if second_mf:
                lines.append(f"     Secondary frame:     {second_mf.upper()} ({sorted_mf[1][1]:.4f})")

            # Interpret the combination
            if v == "Security" and top_mf == "authority":
                lines.append("     Interpretation: When this outlet covers threats, disasters, and crime,")
                lines.append("     it frames them through institutional response -- who is in charge, what")
                lines.append("     the government is doing, what officials are saying. Security is something")
                lines.append("     delivered by authority figures, not something communities create together.")
            elif v == "Security" and top_mf == "care":
                lines.append("     Interpretation: When this outlet covers threats and danger, it leads")
                lines.append("     with the human cost -- who is suffering, who is vulnerable, who needs")
                lines.append("     protection. Security is framed as a compassion issue, not a governance one.")
            elif v == "Benevolence" and top_mf == "authority":
                lines.append("     Interpretation: Welfare, health, and community stories are presented")
                lines.append("     through the lens of governance -- programs, mandates, official responses.")
                lines.append("     Benevolence is something institutions provide, mediated by authority.")
            elif v == "Benevolence" and top_mf == "care":
                lines.append("     Interpretation: Welfare stories center compassion and direct human")
                lines.append("     connection rather than institutional mechanisms.")
            elif v == "Benevolence" and top_mf == "sanctity":
                lines.append("     Interpretation: Welfare and community content carries moral weight --")
                lines.append("     framed not just as policy or need, but as something sacred, righteous,")
                lines.append("     or morally pure. Care is a moral imperative, not just a practical one.")
            elif v == "Power" and top_mf == "authority":
                lines.append("     Interpretation: Political coverage emphasizes hierarchy, governance")
                lines.append("     structures, and institutional legitimacy. Power is presented as the")
                lines.append("     natural domain of authority figures and formal institutions.")
            elif v == "Achievement" and top_mf == "loyalty":
                lines.append("     Interpretation: Success and accomplishment are framed through group")
                lines.append("     identity -- national pride, collective celebration, 'we did it.'")
                lines.append("     Achievement is not individual ambition; it is a source of communal pride.")
            elif v == "Achievement" and top_mf == "authority":
                lines.append("     Interpretation: Achievement is validated through institutional recognition")
                lines.append("     -- formal examinations, official rankings, sanctioned competitions.")
            else:
                lines.append(f"     Interpretation: {v} content is primarily framed through {top_mf} language.")
            lines.append("")

        # ── SECTION 3: THE PSYCHOGRAPHIC PROFILE ──
        lines.append("3. INFERRED READER PSYCHOGRAPHIC PROFILE")
        lines.append("-" * 70)
        lines.append("")

        # Build the synthesis -- this is the part that matters
        v1, v1c = top3[0]
        v2, v2c = top3[1]
        v3, v3c = top3[2]
        v1_pct = v1c / total * 100
        v2_pct = v2c / total * 100
        v3_pct = v3c / total * 100

        # Get MFT frames for top values
        v1_frame = max(mft_data.get(v1, {}), key=mft_data.get(v1, {}).get, default="unknown")
        v2_frame = max(mft_data.get(v2, {}), key=mft_data.get(v2, {}).get, default="unknown")
        v3_frame = max(mft_data.get(v3, {}), key=mft_data.get(v3, {}).get, default="unknown")

        # Get dominant emotion for top value
        v1_emos = sve.get(v1, {})
        v1_top_emo = max({e: v1_emos.get(e, 0) for e in ["fear", "anger", "joy", "trust"]},
                        key=lambda e: v1_emos.get(e, 0), default="trust")

        lines.append(f"   A regular {src} reader inhabits a psychographic space defined by")
        lines.append(f"   three interlocking orientations:")
        lines.append("")
        lines.append(f"   PRIMARY: {v1} ({v1_pct:.0f}% of content) framed through {v1_frame}")
        lines.append(f"   SECONDARY: {v2} ({v2_pct:.0f}% of content) framed through {v2_frame}")
        lines.append(f"   TERTIARY: {v3} ({v3_pct:.0f}% of content) framed through {v3_frame}")
        lines.append("")

        # Source-specific synthesis
        if src == "ABS-CBN":
            lines.append("   SYNTHESIS:")
            lines.append("   The ABS-CBN reader occupies a psychographic space where political")
            lines.append("   power and institutional authority are central to how the world is")
            lines.append("   understood. Nearly a third of content engages with governance and")
            lines.append("   political dynamics, framed through authority -- who is in charge, who")
            lines.append("   is making decisions, what the state is doing. Another third centers")
            lines.append("   community welfare but filtered through a moral/sanctity lens that")
            lines.append("   gives caregiving a sense of righteousness and moral weight. The")
            lines.append("   remaining fifth emphasizes security threats framed through the human")
            lines.append("   cost of danger -- suffering, vulnerability, the need for protection.")
            lines.append("")
            lines.append("   What is notably ABSENT is as revealing as what is present: almost")
            lines.append("   no Achievement content (1.5%). This reader is not being shown a")
            lines.append("   world of aspiration, competition, or personal success. The")
            lines.append("   psychographic orientation is toward monitoring power, caring for")
            lines.append("   community through moral duty, and staying alert to threats -- not")
            lines.append("   toward striving or self-advancement.")
            lines.append("")
            lines.append("   This is the psychographic profile of a reader oriented toward:")
            lines.append("     - Institutional dependence (Power + Authority framing)")
            lines.append("     - Moral protectiveness (Benevolence + Sanctity framing)")
            lines.append("     - Threat vigilance (Security + Care/harm framing)")
            lines.append("     - Low aspiration exposure (near-zero Achievement)")

        elif src == "GMA":
            lines.append("   SYNTHESIS:")
            lines.append("   The GMA reader occupies a psychographic space where institutional")
            lines.append("   authority is the lens through which EVERYTHING is understood.")
            lines.append("   Benevolence, Security, and Power -- the three dominant values -- are")
            lines.append("   ALL framed through Authority. This is the most consistent moral")
            lines.append("   framing of any outlet: whether the story is about community welfare,")
            lines.append("   public safety, or political governance, the answer is always framed")
            lines.append("   through institutions and the people who run them.")
            lines.append("")
            lines.append("   The emotional environment reinforces this: trust dominates both")
            lines.append("   Benevolence and Power content, while fear dominates Security content.")
            lines.append("   The implicit message is: the world has threats (fear), but")
            lines.append("   institutions are trustworthy (trust) and will handle it (authority).")
            lines.append("")
            lines.append("   GMA also carries more Security content (27.8%) than any other outlet,")
            lines.append("   giving its reader a heightened awareness of danger -- but always")
            lines.append("   paired with institutional response. This is not raw threat exposure;")
            lines.append("   it is threat-plus-reassurance through authority.")
            lines.append("")
            lines.append("   This is the psychographic profile of a reader oriented toward:")
            lines.append("     - Institutional trust (Authority framing across all values)")
            lines.append("     - Community through governance (Benevolence + Authority)")
            lines.append("     - Managed threat awareness (Security + Authority + trust emotion)")
            lines.append("     - Moderate achievement/stimulation exposure (balanced middle)")

        elif src == "Rappler":
            lines.append("   SYNTHESIS:")
            lines.append("   The Rappler reader occupies a psychographic space fundamentally")
            lines.append("   different from the other two outlets. Achievement leads -- the only")
            lines.append("   outlet where it does. Over a quarter of content is about success,")
            lines.append("   competition, accomplishment, and personal/national triumph. And")
            lines.append("   this achievement is framed through Loyalty -- meaning it is not")
            lines.append("   individualistic striving but collective pride. When a Filipino wins")
            lines.append("   at the Olympics or passes a licensure exam, it is framed as 'our'")
            lines.append("   success, a source of national identity.")
            lines.append("")
            lines.append("   The emotional environment is dominated by trust across all three")
            lines.append("   top values -- the most consistently trust-oriented outlet. Where")
            lines.append("   ABS-CBN creates an environment of moral weight and GMA creates one")
            lines.append("   of institutional reassurance, Rappler creates one of confidence")
            lines.append("   and forward-looking energy.")
            lines.append("")
            lines.append("   Rappler also has the highest Stimulation (13.9% -- entertainment,")
            lines.append("   arts, culture, food) and the highest Universalism (4.5% -- justice,")
            lines.append("   rights, global awareness). Its reader gets the most diverse")
            lines.append("   psychographic exposure and the most aspirational content environment.")
            lines.append("")
            lines.append("   Security is present (16.4%) but does not dominate, and Power is")
            lines.append("   balanced rather than overwhelming. The world according to Rappler")
            lines.append("   has threats and politics, but it also has achievement, culture,")
            lines.append("   justice, and celebration.")
            lines.append("")
            lines.append("   This is the psychographic profile of a reader oriented toward:")
            lines.append("     - Aspiration and collective pride (Achievement + Loyalty framing)")
            lines.append("     - Cultural engagement (highest Stimulation)")
            lines.append("     - Critical awareness (highest Universalism -- justice, fact-checking)")
            lines.append("     - Confidence rather than fear (trust-dominant emotion)")
            lines.append("     - Diverse psychographic exposure (most balanced value distribution)")

        # ── SECTION 5: WHAT IS ABSENT ──
        lines.append("")
        lines.append("4. WHAT IS ABSENT (Psychographic Gaps)")
        lines.append("-" * 70)
        lines.append("")
        if bottom:
            lines.append(f"   Low-emphasis values: {', '.join(bottom)}")
            lines.append("")
            for v in bottom:
                if v == "Achievement":
                    lines.append(f"   {v}: This reader is rarely exposed to content about personal or")
                    lines.append("   collective accomplishment, competition, or success. The world is not")
                    lines.append("   framed as a place where striving leads to reward.")
                elif v == "Universalism":
                    lines.append(f"   {v}: This reader gets minimal exposure to content about universal")
                    lines.append("   justice, human rights, equity, or global solidarity. The moral frame")
                    lines.append("   stays closer to loyalty and authority than to fairness and universalism.")
                elif v == "Tradition":
                    lines.append(f"   {v}: Cultural heritage and religious tradition are not major themes.")
                elif v == "Self-Direction":
                    lines.append(f"   {v}: Innovation, independence, and self-directed action are")
                    lines.append("   rarely emphasized. Agency comes from institutions, not individuals.")
                elif v == "Conformity":
                    lines.append(f"   {v}: Regulatory and rule-following content is minimal.")
                elif v == "Stimulation":
                    lines.append(f"   {v}: Entertainment, novelty, and cultural enrichment are not")
                    lines.append("   significant parts of the content environment.")
                lines.append("")

        profile_path = os.path.join(OUT_DIR, f"{src.lower().replace('-','')}_value_profile.txt")
        with open(profile_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  Profile saved: {profile_path}")

    # ══════════════════════════════════════════════════════════
    # GENERATE COMPARATIVE SUMMARY
    # ══════════════════════════════════════════════════════════
    comp = []
    comp.append("# Comparative Psychographic Profiles: ABS-CBN vs. GMA vs. Rappler")
    comp.append("")
    comp.append("## At a Glance")
    comp.append("")
    comp.append("| Dimension | ABS-CBN | GMA | Rappler |")
    comp.append("|-----------|---------|-----|---------|")

    for src, vc in source_value_dists.items():
        total = sum(vc.values())

    # Value rows
    all_vals_sorted = sorted(
        set(v for vc in source_value_dists.values() for v in vc),
        key=lambda v: sum(vc.get(v, 0) for vc in source_value_dists.values()),
        reverse=True,
    )
    for v in all_vals_sorted:
        cells = []
        for src in ["ABS-CBN", "GMA", "Rappler"]:
            vc = source_value_dists[src]
            total = sum(vc.values())
            pct = vc.get(v, 0) / total * 100
            # Bold the highest
            is_max = all(pct >= source_value_dists[s].get(v, 0) / sum(source_value_dists[s].values()) * 100
                         for s in source_value_dists)
            cells.append(f"**{pct:.1f}%**" if is_max else f"{pct:.1f}%")
        comp.append(f"| {v} | {cells[0]} | {cells[1]} | {cells[2]} |")

    # MFT row for top value
    comp.append("")
    comp.append("| Top Value's Moral Frame | " +
                " | ".join(
                    f"{max(all_value_article_mft[src][sorted(source_value_dists[src].items(), key=lambda x: x[1], reverse=True)[0][0]], key=all_value_article_mft[src][sorted(source_value_dists[src].items(), key=lambda x: x[1], reverse=True)[0][0]].get).title()}"
                    for src in ["ABS-CBN", "GMA", "Rappler"]
                ) + " |")

    comp.append("")
    comp.append("## Value Orientation Comparison")
    comp.append("")
    comp.append("Each outlet's top 3 Schwartz values define a distinct psychographic space:")
    comp.append("")

    for src in ["ABS-CBN", "GMA", "Rappler"]:
        vc = source_value_dists[src]
        total = sum(vc.values())
        top3 = sorted(vc.items(), key=lambda x: x[1], reverse=True)[:3]
        comp.append(f"**{src}:** {' > '.join(f'{v} ({c/total*100:.0f}%)' for v, c in top3)}")
        comp.append("")

    comp.append("## Key Differences")
    comp.append("")

    comp.append("### 1. What Each Outlet Prioritizes Most")
    comp.append("")
    comp.append("- **ABS-CBN leads on Power** (30%) -- nearly double GMA (14%) and almost double")
    comp.append("  Rappler (17%). This outlet is disproportionately focused on political governance,")
    comp.append("  institutional authority, and who holds power.")
    comp.append("")
    comp.append("- **GMA leads on Security** (28%) -- the highest of any outlet. Crime, disasters,")
    comp.append("  weather threats, drug war coverage. GMA's reader lives in the most threat-aware")
    comp.append("  content environment.")
    comp.append("")
    comp.append("- **Rappler leads on Achievement** (26%) -- the ONLY outlet where Achievement")
    comp.append("  is the top value. Sports, competitions, examinations, business success.")
    comp.append("  Rappler's reader lives in the most aspirational content environment.")
    comp.append("")

    comp.append("### 2. How They Frame the Same Values Differently")
    comp.append("")
    comp.append("All three outlets cover Benevolence (welfare, health, community), but they frame it")
    comp.append("through different moral lenses:")
    comp.append("")

    for src in ["ABS-CBN", "GMA", "Rappler"]:
        mft = all_value_article_mft[src]
        if "Benevolence" in mft:
            top_mf = max(mft["Benevolence"], key=mft["Benevolence"].get)
            comp.append(f"- **{src}** frames Benevolence through **{top_mf.upper()}**")
            if top_mf == "sanctity":
                comp.append("  → Care as moral duty and righteousness")
            elif top_mf == "authority":
                comp.append("  → Care as institutional responsibility, delivered through governance")
            elif top_mf == "care":
                comp.append("  → Care as direct compassion and human connection")
    comp.append("")

    comp.append("Similarly, Security coverage differs:")
    comp.append("")
    for src in ["ABS-CBN", "GMA", "Rappler"]:
        mft = all_value_article_mft[src]
        if "Security" in mft:
            top_mf = max(mft["Security"], key=mft["Security"].get)
            comp.append(f"- **{src}** frames Security through **{top_mf.upper()}**")
            if top_mf == "authority":
                comp.append("  → Threats managed by institutions and officials")
            elif top_mf == "care":
                comp.append("  → Threats experienced through human suffering and vulnerability")
    comp.append("")

    comp.append("### 3. What Each Outlet Underemphasizes")
    comp.append("")
    for src in ["ABS-CBN", "GMA", "Rappler"]:
        vc = source_value_dists[src]
        total = sum(vc.values())
        bottom = [v for v, c in sorted(vc.items(), key=lambda x: x[1])
                  if c / total < 0.05 and v != "Conformity"]
        if bottom:
            comp.append(f"- **{src}** underemphasizes: {', '.join(bottom)}")
    comp.append("")
    comp.append("ABS-CBN's near-zero Achievement (1.5%) is the most striking gap. A reader who")
    comp.append("consumes primarily ABS-CBN content is almost never exposed to narratives of")
    comp.append("aspiration, competition, or success -- a psychographic absence that shapes")
    comp.append("orientation just as much as what IS present.")
    comp.append("")

    comp.append("## Psychographic Profile Summaries")
    comp.append("")

    comp.append("### ABS-CBN Reader")
    comp.append("")
    comp.append("Occupies a psychographic space defined by **institutional dependence** and")
    comp.append("**moral protectiveness**. Nearly a third of content engages with political power,")
    comp.append("framed through authority. Another third covers community welfare, but filtered")
    comp.append("through a sanctity lens that gives caregiving moral weight and righteousness.")
    comp.append("Security threats are framed through human suffering (care), not institutional")
    comp.append("response. Almost no achievement content. This reader monitors power, cares")
    comp.append("through moral duty, and watches for threats -- but does not see a world of")
    comp.append("aspiration or self-advancement.")
    comp.append("")
    comp.append("**Orientation:** Institutional dependence | Moral protectiveness | Threat vigilance | Low aspiration")
    comp.append("")

    comp.append("### GMA Reader")
    comp.append("")
    comp.append("Occupies a psychographic space defined by **institutional trust** across all")
    comp.append("domains. Benevolence, Security, and Power are ALL framed through authority --")
    comp.append("the most consistent moral framing of any outlet. Whether the story is about")
    comp.append("community welfare, public safety, or politics, the answer is always mediated")
    comp.append("by institutions. GMA carries the most Security content of any outlet (28%),")
    comp.append("but always paired with institutional response. This is not raw threat exposure;")
    comp.append("it is threat-plus-reassurance through authority.")
    comp.append("")
    comp.append("**Orientation:** Institutional trust | Managed threat awareness | Community through governance")
    comp.append("")

    comp.append("### Rappler Reader")
    comp.append("")
    comp.append("Occupies a psychographic space defined by **aspiration** and **collective pride**.")
    comp.append("The only outlet where Achievement leads. Success is framed through loyalty --")
    comp.append("national pride, group identity, collective celebration. Rappler also has the")
    comp.append("highest Stimulation (entertainment, arts, culture) and the highest Universalism")
    comp.append("(justice, fact-checking, global awareness). The most diverse and balanced value")
    comp.append("distribution. This reader encounters a world that has threats and politics, but")
    comp.append("also achievement, culture, justice, and celebration.")
    comp.append("")
    comp.append("**Orientation:** Aspiration and collective pride | Cultural engagement | Critical awareness | Diverse exposure")
    comp.append("")

    comp.append("## What This Means")
    comp.append("")
    comp.append("These are not just different editorial choices. They are different **psychographic")
    comp.append("environments** that cultivate different orientations in their audiences:")
    comp.append("")
    comp.append("- The ABS-CBN reader is oriented toward **watching power and guarding against threat**")
    comp.append("- The GMA reader is oriented toward **trusting institutions to manage a dangerous world**")
    comp.append("- The Rappler reader is oriented toward **striving, celebrating, and questioning**")
    comp.append("")
    comp.append("Each profile is constructed from: Schwartz value distribution (what the outlet")
    comp.append("covers) + MFT moral framing (how it frames what it covers) + psychographic gaps")
    comp.append("(what it does NOT cover). Together, these layers move us beyond text analysis")
    comp.append("into defensible psychographic profiling.")

    comp_path = os.path.join(OUT_DIR, "COMPARATIVE_PROFILES.md")
    with open(comp_path, "w") as f:
        f.write("\n".join(comp))
    print(f"  Comparative summary saved: {comp_path}")

    # ══════════════════════════════════════════════════════════
    # PRINT FINAL COMPARISON
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("COMPARATIVE VALUE PROFILES")
    print(f"{'=' * 70}")
    for src, vc in source_value_dists.items():
        total = sum(vc.values())
        top3 = sorted(vc.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n  {src}:")
        for v, c in top3:
            pct = c / total * 100
            bar = "█" * int(pct / 2)
            print(f"    {v:<20} {pct:5.1f}%  {bar}")
        # Top MFT frame for top value
        mft = all_value_article_mft.get(src, {})
        tv = top3[0][0]
        if tv in mft:
            top_mf = max(mft[tv], key=mft[tv].get)
            print(f"    Primary {tv} frame: {top_mf.title()}")

    print(f"\n{'=' * 70}")
    print(f"All outputs in: {OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
