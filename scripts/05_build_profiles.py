"""
05_build_profiles.py - Aggregate sentence scores into outlet-level psychographic profiles.

Inputs:  outputs/intermediate/sentences.parquet
         outputs/intermediate/schwartz_sentence_scores.parquet
         outputs/intermediate/mft_sentence_scores.parquet
Outputs: outputs/profiles/
            schwartz_profile_<outlet>.csv  (per-value mean sim + % dominant + sentence count)
            mft_profile_<outlet>.csv       (per-foundation, with virtue/vice split)
            comparison_schwartz.csv        (9 values x 3 outlets)
            comparison_mft.csv             (5 foundations x 3 outlets)
            sentence_scores.parquet        (full sentence-level scores for re-aggregation)
         outputs/figures/
            schwartz_comparison.png        (grouped bar)
            mft_comparison.png             (grouped bar)
            mft_virtue_vice.png            (stacked bars per outlet)
            outlet_radar_schwartz.png      (radar overlay)
            outlet_radar_mft.png           (radar overlay)
         outputs/diagnostics/
            salience_distribution.csv      (filter cutoffs, retention counts)
            exemplar_coverage.csv          (sentence count per dominant category)

Salience filter: keep only sentences with salience above the median (per framework,
per outlet). Drops neutral/factual sentences that don't morally frame anything.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    DIAGNOSTICS,
    FIGURES,
    INTERMEDIATE,
    OUTLET_DISPLAY,
    OUTLETS,
    PROFILES,
    ensure_dirs,
    load_mft_exemplars,
    load_schwartz_exemplars,
)


def per_outlet_profile_schwartz(df: pd.DataFrame, value_names: list[str]) -> pd.DataFrame:
    rows = []
    n_total = len(df)
    dominant_counts = df["schwartz_dominant"].value_counts()
    for v in value_names:
        mean_sim = df[f"schwartz_{v}"].mean()
        n_dom = int(dominant_counts.get(v, 0))
        pct_dom = (n_dom / n_total * 100) if n_total else 0.0
        rows.append(
            {
                "value": v,
                "mean_similarity": mean_sim,
                "pct_dominant": pct_dom,
                "sentence_count_dominant": n_dom,
            }
        )
    return pd.DataFrame(rows)


def per_outlet_profile_mft(df: pd.DataFrame, foundation_names: list[str]) -> pd.DataFrame:
    rows = []
    n_total = len(df)
    dominant_counts = df["mft_dominant"].value_counts()
    for f in foundation_names:
        mean_sim = df[f"mft_{f}"].mean()
        n_dom = int(dominant_counts.get(f, 0))
        pct_dom = (n_dom / n_total * 100) if n_total else 0.0
        sub = df[df["mft_dominant"] == f]
        n_virtue = int((sub["mft_dominant_pole"] == "virtue").sum())
        n_vice = int((sub["mft_dominant_pole"] == "vice").sum())
        rows.append(
            {
                "foundation": f,
                "mean_similarity": mean_sim,
                "pct_dominant": pct_dom,
                "sentence_count_dominant": n_dom,
                "pct_virtue_of_dominant": (n_virtue / n_dom * 100) if n_dom else 0.0,
                "pct_vice_of_dominant": (n_vice / n_dom * 100) if n_dom else 0.0,
            }
        )
    return pd.DataFrame(rows)


def grouped_bar(df_long: pd.DataFrame, x_col: str, value_col: str, group_col: str, title: str, ylabel: str, out_path: Path):
    pivot = df_long.pivot(index=x_col, columns=group_col, values=value_col)
    pivot = pivot[OUTLETS]  # column order
    pivot.columns = [OUTLET_DISPLAY[c] for c in pivot.columns]
    ax = pivot.plot(kind="bar", figsize=(11, 5.5), width=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.legend(title="Outlet", loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def radar_chart(df_long: pd.DataFrame, x_col: str, value_col: str, group_col: str, title: str, out_path: Path):
    categories = df_long[x_col].unique().tolist()
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.get_cmap("tab10")
    for i, outlet in enumerate(OUTLETS):
        vals = df_long[df_long[group_col] == outlet].set_index(x_col).loc[categories, value_col].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=OUTLET_DISPLAY[outlet], color=colors(i), linewidth=2)
        ax.fill(angles, vals, color=colors(i), alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title(title, fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def virtue_vice_chart(mft_long: pd.DataFrame, out_path: Path):
    foundations = sorted(mft_long["foundation"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, outlet in zip(axes, OUTLETS):
        sub = mft_long[mft_long["outlet"] == outlet].set_index("foundation").loc[foundations]
        virtue_pct = sub["pct_dominant"] * sub["pct_virtue_of_dominant"] / 100
        vice_pct = sub["pct_dominant"] * sub["pct_vice_of_dominant"] / 100
        ax.bar(foundations, virtue_pct, label="Virtue", color="#3a7d44")
        ax.bar(foundations, vice_pct, bottom=virtue_pct, label="Vice", color="#bc4749")
        ax.set_title(OUTLET_DISPLAY[outlet], fontsize=12)
        ax.set_ylabel("% of salient sentences" if outlet == OUTLETS[0] else "")
        ax.tick_params(axis="x", rotation=30)
    axes[-1].legend(loc="upper right")
    fig.suptitle("MFT framing: virtue vs vice split, by outlet", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ensure_dirs()
    sentences = pd.read_parquet(INTERMEDIATE / "sentences.parquet")
    schwartz = pd.read_parquet(INTERMEDIATE / "schwartz_sentence_scores.parquet")
    mft = pd.read_parquet(INTERMEDIATE / "mft_sentence_scores.parquet")

    df = sentences.merge(schwartz, on="sentence_id").merge(mft, on="sentence_id")
    df.to_parquet(PROFILES / "sentence_scores.parquet", index=False)
    print(f"Merged {len(df)} sentence scores")

    value_names = list(load_schwartz_exemplars()["values"].keys())
    foundation_names = list(load_mft_exemplars()["foundations"].keys())

    # Per-outlet salience cutoff (median), filter, then profile
    schwartz_long = []
    mft_long = []
    diag_rows = []
    for outlet in OUTLETS:
        sub = df[df["outlet"] == outlet]
        s_cut = sub["schwartz_salience"].median()
        m_cut = sub["mft_salience"].median()
        s_filt = sub[sub["schwartz_salience"] >= s_cut]
        m_filt = sub[sub["mft_salience"] >= m_cut]
        diag_rows.append({
            "outlet": outlet,
            "n_sentences": len(sub),
            "schwartz_salience_cutoff": s_cut,
            "schwartz_retained": len(s_filt),
            "mft_salience_cutoff": m_cut,
            "mft_retained": len(m_filt),
        })

        schwartz_profile = per_outlet_profile_schwartz(s_filt, value_names)
        schwartz_profile.to_csv(PROFILES / f"schwartz_profile_{outlet}.csv", index=False)
        schwartz_profile["outlet"] = outlet
        schwartz_long.append(schwartz_profile)

        mft_profile = per_outlet_profile_mft(m_filt, foundation_names)
        mft_profile.to_csv(PROFILES / f"mft_profile_{outlet}.csv", index=False)
        mft_profile["outlet"] = outlet
        mft_long.append(mft_profile)

    schwartz_all = pd.concat(schwartz_long, ignore_index=True)
    mft_all = pd.concat(mft_long, ignore_index=True)

    # Comparison CSVs (% dominant is the headline number)
    s_comp = schwartz_all.pivot(index="value", columns="outlet", values="pct_dominant")[OUTLETS]
    s_comp.columns = [OUTLET_DISPLAY[c] for c in s_comp.columns]
    s_comp.to_csv(PROFILES / "comparison_schwartz.csv")

    m_comp = mft_all.pivot(index="foundation", columns="outlet", values="pct_dominant")[OUTLETS]
    m_comp.columns = [OUTLET_DISPLAY[c] for c in m_comp.columns]
    m_comp.to_csv(PROFILES / "comparison_mft.csv")

    pd.DataFrame(diag_rows).to_csv(DIAGNOSTICS / "salience_distribution.csv", index=False)

    # Exemplar coverage
    coverage = []
    for outlet in OUTLETS:
        sub = df[df["outlet"] == outlet]
        for v in value_names:
            coverage.append({"outlet": outlet, "framework": "schwartz", "category": v,
                             "n_dominant_all": int((sub["schwartz_dominant"] == v).sum())})
        for f in foundation_names:
            coverage.append({"outlet": outlet, "framework": "mft", "category": f,
                             "n_dominant_all": int((sub["mft_dominant"] == f).sum())})
    pd.DataFrame(coverage).to_csv(DIAGNOSTICS / "exemplar_coverage.csv", index=False)

    # Figures
    grouped_bar(schwartz_all, "value", "pct_dominant", "outlet",
                "Schwartz value distribution (% of salient sentences per outlet)",
                "% of salient sentences", FIGURES / "schwartz_comparison.png")
    grouped_bar(mft_all, "foundation", "pct_dominant", "outlet",
                "MFT framing distribution (% of salient sentences per outlet)",
                "% of salient sentences", FIGURES / "mft_comparison.png")
    radar_chart(schwartz_all, "value", "pct_dominant", "outlet",
                "Schwartz profile by outlet", FIGURES / "outlet_radar_schwartz.png")
    radar_chart(mft_all, "foundation", "pct_dominant", "outlet",
                "MFT profile by outlet", FIGURES / "outlet_radar_mft.png")
    virtue_vice_chart(mft_all, FIGURES / "mft_virtue_vice.png")

    print(f"\nWrote profiles to {PROFILES}")
    print(f"Wrote figures to {FIGURES}")
    print(f"Wrote diagnostics to {DIAGNOSTICS}")


if __name__ == "__main__":
    main()
