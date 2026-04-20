"""
Psychographic Profiling Pipeline
================================
Pipeline: Text → Features → Psychological Constructs → Psychographic Profile

This script moves beyond raw text analysis (emotions, sentiment, topics) and
introduces a CONSTRUCT LAYER that maps measurable text features to psychological
orientations:

  Features extracted:
    1. NRC Emotion Lexicon — 8 emotions + pos/neg
    2. NRC VAD — Valence, Arousal, Dominance (word-level continuous scores)
    3. VADER — sentiment intensity
    4. Moral Foundations Dictionary — Care, Fairness, Loyalty, Authority, Sanctity
    5. Empath — 200 psychologically meaningful lexical categories
    6. Linguistic complexity — readability, sentence length, vocabulary level
    7. Pronoun analysis — collective (we/our) vs. individual (I/my) framing

  Constructs derived (each mapped FROM features):
    - Security Orientation: threat sensitivity + safety concern + crime/security topics
    - Authority Orientation: dominance + authority moral framing + institutional language
    - Justice/Reform Orientation: fairness framing + rights language + reform topics
    - Communal Orientation: care framing + benevolence + collective pronouns
    - Threat Sensitivity: fear + anger + harm framing + negative valence
    - Cognitive Style: complexity + analytical language + information density

  Output:
    - Per-source construct scores (the psychographic profile)
    - Radar chart comparing all three outlets
    - Detailed CSV of all features and constructs
    - Profile narrative summary
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
from collections import Counter

from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
import textstat

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "psychographic_profiles")
os.makedirs(OUT_DIR, exist_ok=True)

SOURCES = {
    "abs": {"name": "ABS-CBN", "data_dir": os.path.join(CLEANED_DIR, "abs")},
    "gma": {"name": "GMA",     "data_dir": os.path.join(CLEANED_DIR, "gma")},
    "rappler": {"name": "Rappler", "data_dir": os.path.join(CLEANED_DIR, "rappler")},
}

NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
    "positive", "negative",
]

vader = SentimentIntensityAnalyzer()
lexicon = Empath()

# ══════════════════════════════════════════════════════════════
# MORAL FOUNDATIONS DICTIONARY (MFD 2.0 stems)
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

# Empath categories relevant to psychographic constructs
EMPATH_PSYCH_CATEGORIES = [
    # Security / Threat
    "violence", "crime", "war", "weapon", "death", "terrorism", "fear",
    "suffering", "pain", "aggression",
    # Authority / Institutions
    "government", "law", "politics", "leader", "military", "police",
    "power", "order", "dominance",
    # Justice / Reform
    "justice", "dispute", "protest", "liberty",
    # Community / Care
    "help", "sympathy", "trust", "family", "friends", "community",
    "children", "love", "warmth", "giving",
    # Economy / Achievement
    "business", "money", "work", "achievement", "economics",
    # Culture / Tradition
    "religion", "worship", "celebration", "art",
    # Wellbeing
    "health", "healing", "optimism", "cheerfulness",
    "sadness", "anger", "disgust", "nervousness",
]

# Pronoun sets
COLLECTIVE_PRONOUNS = {"we", "us", "our", "ours", "ourselves"}
INDIVIDUAL_PRONOUNS = {"i", "me", "my", "mine", "myself"}


# ══════════════════════════════════════════════════════════════
# DATA LOADING
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
    # Remove common outlet suffixes
    text = re.sub(r"\s*\|?\s*GMA News Online\s*$", "", text)
    return text


def get_article_text(rec):
    """Combine headline + body for full article text."""
    parts = []
    headline = clean_text(rec.get("headline", ""))
    body = clean_text(rec.get("text", ""))
    if headline:
        parts.append(headline)
    if body:
        parts.append(body)
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def get_nrc_scores(text):
    if not text:
        return {e: 0.0 for e in NRC_EMOTIONS}
    emotion_obj = NRCLex("init")
    emotion_obj.load_raw_text(text)
    raw = emotion_obj.raw_emotion_scores
    total = sum(raw.values()) if raw else 0
    return {e: (raw.get(e, 0) / total if total > 0 else 0.0) for e in NRC_EMOTIONS}


def get_vader_scores(text):
    if not text:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}
    return vader.polarity_scores(text)


def get_mfd_scores(text):
    """Score text against Moral Foundations Dictionary using stem matching."""
    if not text:
        return {k: 0.0 for k in MFD}
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)
    if word_count == 0:
        return {k: 0.0 for k in MFD}

    scores = {}
    for foundation, stems in MFD.items():
        count = 0
        for stem in stems:
            count += sum(1 for w in words if w.startswith(stem) or stem in w)
        scores[foundation] = count / word_count
    return scores


def get_empath_scores(text):
    """Get Empath category scores for psychographically relevant categories."""
    if not text:
        return {c: 0.0 for c in EMPATH_PSYCH_CATEGORIES}
    all_scores = lexicon.analyze(text, normalize=True) or {}
    return {c: all_scores.get(c, 0.0) for c in EMPATH_PSYCH_CATEGORIES}


def get_linguistic_complexity(text):
    """Compute readability and complexity features."""
    if not text or len(text.split()) < 10:
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "avg_sentence_length": 0.0,
            "avg_word_length": 0.0,
            "lexical_diversity": 0.0,
        }
    words = text.split()
    unique_words = set(w.lower() for w in words)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "avg_word_length": np.mean([len(w) for w in words]),
        "lexical_diversity": len(unique_words) / len(words) if words else 0.0,
    }


def get_pronoun_ratios(text):
    """Measure collective vs. individual pronoun usage."""
    if not text:
        return {"collective_pronoun_ratio": 0.0, "individual_pronoun_ratio": 0.0,
                "collective_vs_individual": 0.0}
    words = text.lower().split()
    word_count = len(words)
    if word_count == 0:
        return {"collective_pronoun_ratio": 0.0, "individual_pronoun_ratio": 0.0,
                "collective_vs_individual": 0.0}

    coll = sum(1 for w in words if w in COLLECTIVE_PRONOUNS)
    indv = sum(1 for w in words if w in INDIVIDUAL_PRONOUNS)

    coll_ratio = coll / word_count
    indv_ratio = indv / word_count
    # Positive = more collective, negative = more individual
    cvi = (coll - indv) / max(coll + indv, 1)

    return {
        "collective_pronoun_ratio": coll_ratio,
        "individual_pronoun_ratio": indv_ratio,
        "collective_vs_individual": cvi,
    }


def extract_all_features(text):
    """Run all feature extractors on a single article."""
    features = {}
    features.update({f"nrc_{k}": v for k, v in get_nrc_scores(text).items()})
    features.update({f"vader_{k}": v for k, v in get_vader_scores(text).items()})
    features.update({f"mfd_{k}": v for k, v in get_mfd_scores(text).items()})
    features.update({f"empath_{k}": v for k, v in get_empath_scores(text).items()})
    features.update(get_linguistic_complexity(text))
    features.update(get_pronoun_ratios(text))
    return features


# ══════════════════════════════════════════════════════════════
# CONSTRUCT MAPPING
# ══════════════════════════════════════════════════════════════
# This is the key layer: features → psychological constructs
# Each construct is a weighted combination of features that
# together indicate a measurable psychological orientation.

def compute_constructs(feature_means):
    """
    Map aggregated feature means to psychological constructs.
    Each construct draws from multiple feature sources for robustness.

    Returns dict of construct_name → score (0-1 normalized at the end).
    """
    constructs = {}

    # ── SECURITY ORIENTATION ──
    # High = outlet emphasizes safety, threat, protection, crime
    # Sources: NRC fear, MFD care (protection side), Empath crime/violence/war
    constructs["Security Orientation"] = np.mean([
        feature_means.get("nrc_fear", 0) * 5,           # fear language (scaled)
        feature_means.get("mfd_care.virtue", 0) * 50,   # protect/safe/guard language
        feature_means.get("empath_crime", 0) * 50,       # crime topic prevalence
        feature_means.get("empath_violence", 0) * 50,    # violence topic
        feature_means.get("empath_war", 0) * 50,         # war topic
        feature_means.get("empath_police", 0) * 50,      # law enforcement
    ])

    # ── AUTHORITY ORIENTATION ──
    # High = outlet defers to authority, emphasizes order, institutions, leadership
    # Sources: MFD authority, Empath government/law/military/power
    constructs["Authority Orientation"] = np.mean([
        feature_means.get("mfd_authority.virtue", 0) * 50,
        feature_means.get("empath_government", 0) * 50,
        feature_means.get("empath_law", 0) * 50,
        feature_means.get("empath_military", 0) * 50,
        feature_means.get("empath_power", 0) * 50,
        feature_means.get("empath_leader", 0) * 50,
        feature_means.get("empath_order", 0) * 50,
    ])

    # ── JUSTICE / REFORM ORIENTATION ──
    # High = outlet emphasizes fairness, rights, reform, accountability
    # Sources: MFD fairness, Empath justice/protest/liberty
    constructs["Justice/Reform Orientation"] = np.mean([
        feature_means.get("mfd_fairness.virtue", 0) * 50,
        feature_means.get("mfd_fairness.vice", 0) * 50,  # calling out injustice is also reform-oriented
        feature_means.get("empath_justice", 0) * 50,
        feature_means.get("empath_protest", 0) * 50,
        feature_means.get("empath_liberty", 0) * 50,
    ])

    # ── COMMUNAL ORIENTATION ──
    # High = outlet emphasizes community, helping, togetherness, family
    # Sources: MFD care + loyalty, Empath community/family/help, collective pronouns
    constructs["Communal Orientation"] = np.mean([
        feature_means.get("mfd_care.virtue", 0) * 50,
        feature_means.get("mfd_loyalty.virtue", 0) * 50,
        feature_means.get("empath_help", 0) * 50,
        feature_means.get("empath_family", 0) * 50,
        feature_means.get("empath_community", 0) * 50,
        feature_means.get("empath_sympathy", 0) * 50,
        feature_means.get("collective_pronoun_ratio", 0) * 200,
    ])

    # ── THREAT SENSITIVITY ──
    # High = outlet frames the world as dangerous, emphasizes loss/risk/harm
    # Sources: NRC fear + anger + disgust, MFD harm, VADER negative, Empath suffering
    constructs["Threat Sensitivity"] = np.mean([
        feature_means.get("nrc_fear", 0) * 5,
        feature_means.get("nrc_anger", 0) * 5,
        feature_means.get("nrc_disgust", 0) * 5,
        feature_means.get("mfd_care.vice", 0) * 50,     # harm/suffering language
        feature_means.get("vader_neg", 0) * 5,
        feature_means.get("empath_suffering", 0) * 50,
        feature_means.get("empath_death", 0) * 50,
    ])

    # ── COGNITIVE STYLE ──
    # High = outlet uses complex, analytical, information-dense language
    # Sources: readability (inverted), sentence length, lexical diversity
    fk_grade = feature_means.get("flesch_kincaid_grade", 0)
    constructs["Cognitive Complexity"] = np.mean([
        min(fk_grade / 16.0, 1.0),                              # grade level normalized
        min(feature_means.get("avg_sentence_length", 0) / 40, 1.0),
        feature_means.get("lexical_diversity", 0),
    ])

    # ── HOPE / OPPORTUNITY FRAMING ──
    # High = outlet frames positively, emphasizes opportunity and progress
    # Sources: NRC joy + anticipation + trust, VADER positive, Empath optimism
    constructs["Hope/Opportunity Framing"] = np.mean([
        feature_means.get("nrc_joy", 0) * 5,
        feature_means.get("nrc_anticipation", 0) * 5,
        feature_means.get("nrc_trust", 0) * 5,
        feature_means.get("vader_pos", 0) * 5,
        feature_means.get("empath_optimism", 0) * 50,
        feature_means.get("empath_cheerfulness", 0) * 50,
    ])

    # ── TRADITION / CULTURAL ORIENTATION ──
    # High = outlet emphasizes culture, religion, heritage, tradition
    # Sources: MFD sanctity + loyalty, Empath religion/worship/celebration
    constructs["Tradition/Cultural Orientation"] = np.mean([
        feature_means.get("mfd_sanctity.virtue", 0) * 50,
        feature_means.get("mfd_loyalty.virtue", 0) * 50,
        feature_means.get("empath_religion", 0) * 50,
        feature_means.get("empath_worship", 0) * 50,
        feature_means.get("empath_celebration", 0) * 50,
    ])

    return constructs


# ══════════════════════════════════════════════════════════════
# PROFILE GENERATION
# ══════════════════════════════════════════════════════════════

def generate_profile_narrative(source_name, constructs, feature_means):
    """Generate a human-readable psychographic profile statement."""
    # Sort constructs by strength
    sorted_c = sorted(constructs.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_c[:3]
    bottom2 = sorted_c[-2:]

    # Determine dominant emotions
    emotions = {e: feature_means.get(f"nrc_{e}", 0)
                for e in ["anger", "fear", "joy", "sadness", "trust", "anticipation", "disgust", "surprise"]}
    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]

    # Moral foundations
    mf_scores = {}
    for foundation in ["care", "fairness", "loyalty", "authority", "sanctity"]:
        virtue = feature_means.get(f"mfd_{foundation}.virtue", 0)
        vice = feature_means.get(f"mfd_{foundation}.vice", 0)
        mf_scores[foundation] = virtue + vice
    top_moral = sorted(mf_scores.items(), key=lambda x: x[1], reverse=True)[:2]

    lines = []
    lines.append(f"PSYCHOGRAPHIC PROFILE: {source_name}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("PROFILE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"A reader drawn to {source_name}'s content is likely oriented toward")
    lines.append(f"a psychographic profile characterized by:")
    lines.append("")
    for name, score in top3:
        lines.append(f"  - Strong {name} (score: {score:.3f})")
    lines.append("")
    lines.append(f"With relatively lower emphasis on:")
    for name, score in bottom2:
        lines.append(f"  - {name} (score: {score:.3f})")

    lines.append("")
    lines.append("EMOTIONAL ORIENTATION")
    lines.append("-" * 40)
    lines.append(f"Dominant emotions in content: {', '.join(f'{e} ({s:.3f})' for e, s in top_emotions)}")
    lines.append(f"Sentiment valence: {feature_means.get('vader_compound', 0):.3f} "
                 f"({'positive-leaning' if feature_means.get('vader_compound', 0) > 0.05 else 'negative-leaning' if feature_means.get('vader_compound', 0) < -0.05 else 'neutral'})")

    lines.append("")
    lines.append("MORAL FRAMING")
    lines.append("-" * 40)
    lines.append(f"Primary moral foundations: {', '.join(f'{f.title()} ({s:.4f})' for f, s in top_moral)}")

    lines.append("")
    lines.append("COGNITIVE STYLE")
    lines.append("-" * 40)
    fk = feature_means.get("flesch_kincaid_grade", 0)
    fre = feature_means.get("flesch_reading_ease", 0)
    lines.append(f"Reading level: Grade {fk:.1f} (Flesch-Kincaid)")
    lines.append(f"Readability: {fre:.1f} (Flesch Reading Ease)")
    lines.append(f"Avg sentence length: {feature_means.get('avg_sentence_length', 0):.1f} words")
    lines.append(f"Lexical diversity: {feature_means.get('lexical_diversity', 0):.3f}")

    lines.append("")
    lines.append("FRAMING ORIENTATION")
    lines.append("-" * 40)
    cvi = feature_means.get("collective_vs_individual", 0)
    lines.append(f"Collective vs. individual framing: {cvi:.3f} "
                 f"({'collective-leaning' if cvi > 0 else 'individual-leaning'})")

    threat = constructs.get("Threat Sensitivity", 0)
    hope = constructs.get("Hope/Opportunity Framing", 0)
    if threat > hope:
        lines.append(f"Framing tendency: threat-oriented (threat={threat:.3f} > hope={hope:.3f})")
    else:
        lines.append(f"Framing tendency: opportunity-oriented (hope={hope:.3f} > threat={threat:.3f})")

    lines.append("")
    lines.append("PSYCHOGRAPHIC INTERPRETATION")
    lines.append("-" * 40)
    lines.append(f"A regular {source_name} reader is exposed to content that signals:")

    if constructs.get("Security Orientation", 0) > constructs.get("Justice/Reform Orientation", 0):
        lines.append("  - Security and stability as primary concerns")
    else:
        lines.append("  - Justice, accountability, and reform as primary concerns")

    if constructs.get("Authority Orientation", 0) > 0.3:
        lines.append("  - Respect for institutional authority and governance structures")

    if constructs.get("Communal Orientation", 0) > 0.3:
        lines.append("  - Community welfare and collective responsibility")

    if constructs.get("Threat Sensitivity", 0) > constructs.get("Hope/Opportunity Framing", 0):
        lines.append("  - A world framed through threat and risk (threat-sensitive orientation)")
    else:
        lines.append("  - A world framed through opportunity and progress (hope-oriented)")

    if constructs.get("Tradition/Cultural Orientation", 0) > 0.2:
        lines.append("  - Cultural identity and traditional values")

    if constructs.get("Cognitive Complexity", 0) > 0.5:
        lines.append("  - Analytical, information-dense content (higher cognitive engagement)")
    else:
        lines.append("  - Accessible, direct communication style")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════

def create_radar_chart(all_constructs, output_path):
    """Create a radar chart comparing psychographic profiles across sources."""
    construct_names = list(next(iter(all_constructs.values())).keys())
    num_constructs = len(construct_names)

    angles = np.linspace(0, 2 * np.pi, num_constructs, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = {"ABS-CBN": "#e74c3c", "GMA": "#3498db", "Rappler": "#2ecc71"}

    for source_name, constructs in all_constructs.items():
        values = [constructs[c] for c in construct_names]
        values += values[:1]
        color = colors.get(source_name, "#333333")
        ax.plot(angles, values, "o-", linewidth=2, label=source_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(construct_names, size=9, wrap=True)
    ax.set_ylim(0, max(max(c.values()) for c in all_constructs.values()) * 1.15)
    ax.set_title("Psychographic Profile Comparison\nPhilippine Media Sources",
                 size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Radar chart saved: {output_path}")


def create_moral_foundations_chart(all_mf_scores, output_path):
    """Bar chart comparing moral foundations across sources."""
    foundations = ["care", "fairness", "loyalty", "authority", "sanctity"]
    x = np.arange(len(foundations))
    width = 0.25
    colors = {"ABS-CBN": "#e74c3c", "GMA": "#3498db", "Rappler": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (source, scores) in enumerate(all_mf_scores.items()):
        vals = [scores[f] for f in foundations]
        color = colors.get(source, "#333333")
        ax.bar(x + i * width, vals, width, label=source, color=color, alpha=0.85)

    ax.set_xlabel("Moral Foundation", fontsize=12)
    ax.set_ylabel("Score (word proportion)", fontsize=12)
    ax.set_title("Moral Foundations Across Philippine Media Sources", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f.title() for f in foundations], fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Moral foundations chart saved: {output_path}")


def create_emotion_comparison_chart(all_emotion_scores, output_path):
    """Grouped bar chart of NRC emotions across sources."""
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    x = np.arange(len(emotions))
    width = 0.25
    colors = {"ABS-CBN": "#e74c3c", "GMA": "#3498db", "Rappler": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (source, scores) in enumerate(all_emotion_scores.items()):
        vals = [scores[e] for e in emotions]
        color = colors.get(source, "#333333")
        ax.bar(x + i * width, vals, width, label=source, color=color, alpha=0.85)

    ax.set_xlabel("Emotion", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title("Emotional Orientation Across Philippine Media Sources", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([e.title() for e in emotions], fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Emotion comparison chart saved: {output_path}")


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PSYCHOGRAPHIC PROFILING PIPELINE")
    print("Text → Features → Constructs → Profile")
    print("=" * 60)

    all_constructs = {}
    all_feature_means = {}
    all_mf_scores = {}
    all_emotion_scores = {}
    all_article_features = []

    for key, cfg in SOURCES.items():
        source_name = cfg["name"]
        print(f"\n{'─' * 60}")
        print(f"Processing: {source_name}")
        print(f"{'─' * 60}")

        # Load data
        records = load_source_data(cfg["data_dir"])
        print(f"  Loaded {len(records)} articles")

        # Extract features for every article
        print("  Extracting features (NRC, VADER, MFD, Empath, complexity, pronouns)...")
        article_features = []
        for i, rec in enumerate(records):
            text = get_article_text(rec)
            if not text or len(text.split()) < 20:
                continue
            feats = extract_all_features(text)
            feats["source"] = source_name
            article_features.append(feats)
            if (i + 1) % 200 == 0:
                print(f"    ... {i + 1}/{len(records)} articles processed")

        print(f"  {len(article_features)} articles with sufficient text")
        all_article_features.extend(article_features)

        # Aggregate: mean of all features across articles for this source
        df = pd.DataFrame(article_features)
        numeric_cols = [c for c in df.columns if c != "source"]
        feature_means = df[numeric_cols].mean().to_dict()
        all_feature_means[source_name] = feature_means

        # Store emotion scores for visualization
        all_emotion_scores[source_name] = {
            e: feature_means[f"nrc_{e}"] for e in
            ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
        }

        # Store moral foundation scores for visualization
        mf = {}
        for foundation in ["care", "fairness", "loyalty", "authority", "sanctity"]:
            mf[foundation] = (feature_means.get(f"mfd_{foundation}.virtue", 0) +
                              feature_means.get(f"mfd_{foundation}.vice", 0))
        all_mf_scores[source_name] = mf

        # Map features → constructs
        constructs = compute_constructs(feature_means)
        all_constructs[source_name] = constructs

        print(f"\n  CONSTRUCT SCORES for {source_name}:")
        for name, score in sorted(constructs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 40)
            print(f"    {name:<32} {score:.4f}  {bar}")

        # Generate and save profile narrative
        profile = generate_profile_narrative(source_name, constructs, feature_means)
        profile_path = os.path.join(OUT_DIR, f"{key}_psychographic_profile.txt")
        with open(profile_path, "w") as f:
            f.write(profile)
        print(f"\n  Profile saved: {profile_path}")

    # ── Save all article-level features ──
    print(f"\n{'─' * 60}")
    print("Saving outputs...")
    all_df = pd.DataFrame(all_article_features)
    all_df.to_csv(os.path.join(OUT_DIR, "all_article_features.csv"), index=False)
    print(f"  Article features CSV: {os.path.join(OUT_DIR, 'all_article_features.csv')}")

    # ── Save construct comparison ──
    construct_df = pd.DataFrame(all_constructs).T
    construct_df.index.name = "source"
    construct_df.to_csv(os.path.join(OUT_DIR, "construct_comparison.csv"))
    print(f"  Construct comparison: {os.path.join(OUT_DIR, 'construct_comparison.csv')}")

    # ── Save source-level feature means ──
    means_df = pd.DataFrame(all_feature_means).T
    means_df.index.name = "source"
    means_df.to_csv(os.path.join(OUT_DIR, "source_feature_means.csv"))
    print(f"  Feature means: {os.path.join(OUT_DIR, 'source_feature_means.csv')}")

    # ── Visualizations ──
    print(f"\n{'─' * 60}")
    print("Generating visualizations...")
    create_radar_chart(all_constructs, os.path.join(OUT_DIR, "psychographic_radar.png"))
    create_moral_foundations_chart(all_mf_scores, os.path.join(OUT_DIR, "moral_foundations_comparison.png"))
    create_emotion_comparison_chart(all_emotion_scores, os.path.join(OUT_DIR, "emotion_orientation_comparison.png"))

    # ── Print comparative summary ──
    print(f"\n{'=' * 60}")
    print("COMPARATIVE PSYCHOGRAPHIC SUMMARY")
    print(f"{'=' * 60}")
    for construct in list(next(iter(all_constructs.values())).keys()):
        print(f"\n  {construct}:")
        scores = {s: c[construct] for s, c in all_constructs.items()}
        for source in sorted(scores, key=scores.get, reverse=True):
            bar = "█" * int(scores[source] * 40)
            print(f"    {source:<12} {scores[source]:.4f}  {bar}")

    print(f"\n{'=' * 60}")
    print("Done. All outputs in:", OUT_DIR)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
