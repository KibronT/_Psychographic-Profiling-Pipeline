"""
01_clean.py - Load outlet JSONL files, normalize text, filter short articles.

Input:  data/{outlet}/*.jsonl  (one JSON object per line, fields: headline, text, ...)
Output: outputs/intermediate/articles.parquet
            columns: article_id, outlet, headline, text, full_text, word_count

The 'full_text' column concatenates headline + body and is what downstream
scripts embed. Articles with fewer than MIN_WORDS words are dropped.
"""

import json
import re
from pathlib import Path

import pandas as pd

from utils import DATA_DIR, INTERMEDIATE, OUTLETS, MIN_WORDS, ensure_dirs


OUTLET_FOOTERS = [
    "GMA News Online",
    "GMA Integrated News",
    "Rappler.com",
    "ABS-CBN News",
    "—Rappler.com",
    "— Rappler.com",
]


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_footers(s: str) -> str:
    for footer in OUTLET_FOOTERS:
        if s.endswith(footer):
            s = s[: -len(footer)].rstrip(" -—–")
    return s


def load_outlet(outlet: str) -> pd.DataFrame:
    records = []
    outlet_dir = DATA_DIR / outlet
    for fpath in sorted(outlet_dir.glob("*.jsonl")):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                headline = normalize_whitespace(str(rec.get("headline", "")))
                text = strip_footers(normalize_whitespace(str(rec.get("text", ""))))
                full_text = (headline + ". " + text).strip(". ").strip()
                word_count = len(full_text.split())
                records.append(
                    {
                        "outlet": outlet,
                        "headline": headline,
                        "text": text,
                        "full_text": full_text,
                        "word_count": word_count,
                        "source_file": fpath.name,
                    }
                )
    return pd.DataFrame(records)


def main():
    ensure_dirs()
    frames = []
    for outlet in OUTLETS:
        df = load_outlet(outlet)
        before = len(df)
        df = df[df["word_count"] >= MIN_WORDS].reset_index(drop=True)
        after = len(df)
        print(f"{outlet}: {after} articles (dropped {before - after} below {MIN_WORDS} words)")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "article_id", out.index.map(lambda i: f"a{i:06d}"))
    out_path = INTERMEDIATE / "articles.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nWrote {len(out)} articles to {out_path}")


if __name__ == "__main__":
    main()
