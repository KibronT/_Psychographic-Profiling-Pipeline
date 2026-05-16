"""Shared paths, constants, and small helpers for the pipeline scripts."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REFERENCE_DIR = ROOT / "reference"
OUTPUTS = ROOT / "outputs"
INTERMEDIATE = OUTPUTS / "intermediate"
PROFILES = OUTPUTS / "profiles"
FIGURES = OUTPUTS / "figures"
DIAGNOSTICS = OUTPUTS / "diagnostics"
STABILITY = OUTPUTS / "stability"

OUTLETS = ["abs", "gma", "rappler"]
OUTLET_DISPLAY = {"abs": "ABS-CBN", "gma": "GMA", "rappler": "Rappler"}

SBERT_MODEL = "all-MiniLM-L6-v2"
RANDOM_STATE = 42
MIN_WORDS = 20

# Set by the stability test, used by the production pipeline.
# 50 = use the entire reference pool. The stability test writes the
# chosen size here once Spearman >= 0.95 is reached between consecutive
# sizes, or 50 if no size converges.
ACTIVE_SCHWARTZ_SIZE = 50  # Stability test: did not converge at Spearman >= 0.95; using full pool. See outputs/stability/README.md
ACTIVE_MFT_SIZE = 50       # After the loyalty/sanctity refinement the post-refinement stability test dropped to mean Spearman 0.900
                           # at 35->50, below the 0.95 threshold. Per the fallback rule we use the full 50-exemplar pool.
                           # Top-3 dominant categories per outlet are unchanged between sizes. See docs/VALIDATION_CHECK.md § 6.


def ensure_dirs():
    for d in [INTERMEDIATE, PROFILES, FIGURES, DIAGNOSTICS, STABILITY]:
        d.mkdir(parents=True, exist_ok=True)


def load_schwartz_exemplars():
    with open(REFERENCE_DIR / "schwartz_exemplars.json") as f:
        return json.load(f)


def load_mft_exemplars():
    with open(REFERENCE_DIR / "mft_exemplars.json") as f:
        return json.load(f)
