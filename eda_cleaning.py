"""Clean startup funding data and generate reusable analysis visuals."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from startup_funding.config import DATA_DIR, IMAGES_DIR
from startup_funding.data import dataset_profile, load_and_clean_data, write_data_artifacts


IMAGES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.max_open_warning": 0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def fmt_billions(value: float, _position: int | None = None) -> str:
    if value >= 1e9:
        return f"${value / 1e9:.1f}B"
    if value >= 1e6:
        return f"${value / 1e6:.0f}M"
    if value >= 1e3:
        return f"${value / 1e3:.0f}K"
    return f"${value:.0f}"


def generate_cover_image(df: pd.DataFrame) -> None:
    """Generate a README banner from the current cleaned dataset profile."""
    profile = dataset_profile(df)
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("#101820")
    ax.set_facecolor("#101820")

    ax.text(
        0.5,
        3.8,
        "STARTUP FUNDING ANALYZER",
        fontsize=32,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )
    ax.text(
        0.5,
        3.0,
        "Clean data pipeline | Funding prediction | Similar-startup benchmarking",
        fontsize=15,
        color="#65D6AD",
        ha="center",
        va="center",
    )
    ax.text(
        0.5,
        2.25,
        f"{profile['records']:,} startups | {profile['industries']} industries | "
        f"{profile['locations']} locations | {fmt_billions(profile['total_funding_usd'])} tracked funding",
        fontsize=12,
        color="#DDE7F0",
        ha="center",
        va="center",
    )
    ax.text(
        0.5,
        1.45,
        "A production-ready data science project for startup funding exploration and readiness scoring.",
        fontsize=11,
        color="#AAB7C4",
        ha="center",
        va="center",
    )
    ax.text(0.5, 0.45, "Azubi Africa Data Science LP1", fontsize=9, color="#7C8A99", ha="center", va="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.savefig(IMAGES_DIR / "cover.png", dpi=160, bbox_inches="tight", pad_inches=0.2, facecolor="#101820")
    plt.close(fig)


def perform_eda(df: pd.DataFrame) -> None:
    """Generate EDA visualizations from cleaned data."""
    yearly = df.groupby("Funding Year")["Amount in ($)"].sum().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(yearly.index, yearly.values, alpha=0.18, color="#2563EB")
    ax.plot(yearly.index, yearly.values, marker="o", linewidth=2.5, color="#2563EB")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_billions))
    ax.set_title("Funding Capital Tracked by Year", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Funding year")
    ax.set_ylabel("Total funding")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "funding_trend.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    top_locations = df.groupby("Head Quarter")["Amount in ($)"].sum().sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(top_locations)))
    ax.barh(top_locations.index, top_locations.values, color=colors)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_billions))
    ax.set_title("Top Startup Hubs by Funding", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Total funding")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "top_locations_funding.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    amounts = df["Amount in ($)"] / 1e6
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(amounts, bins=45, color="#0F766E", edgecolor="white", alpha=0.85)
    ax.axvline(amounts.median(), color="#DC2626", linestyle="--", linewidth=2, label=f"Median ${amounts.median():.1f}M")
    ax.axvline(amounts.mean(), color="#7C3AED", linestyle="--", linewidth=2, label=f"Mean ${amounts.mean():.1f}M")
    ax.set_title("Funding Round Distribution", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Funding amount ($M)")
    ax.set_ylabel("Startup count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "funding_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    pre = df[df["Funding Year"].between(2018, 2019)]["Amount in ($)"] / 1e6
    during = df[df["Funding Year"].between(2020, 2021)]["Amount in ($)"] / 1e6
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["2018-2019", "2020-2021"]
    box = ax.boxplot([pre, during], labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], ["#2563EB", "#DC2626"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    ax.set_title("Funding Before and During the Pandemic Window", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Funding amount ($M)")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "pandemic_impact.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    founded_counts = df["Year Founded"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(founded_counts.index, 0, founded_counts.values, alpha=0.25, color="#16A34A")
    ax.plot(founded_counts.index, founded_counts.values, marker="s", linewidth=2.4, color="#16A34A")
    ax.set_title("Startups by Founded Year", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Founded year")
    ax.set_ylabel("Startup count")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "startups_per_year.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    industry_counts = df[df["Industry In"] != "Unknown"]["Industry In"].value_counts().head(10)
    if len(industry_counts) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        wedges, _, autotexts = ax.pie(
            industry_counts.values,
            labels=None,
            autopct="%1.1f%%",
            startangle=90,
            colors=plt.cm.tab20(np.linspace(0, 1, len(industry_counts))),
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        for text in autotexts:
            text.set_fontsize(8)
        ax.legend(
            wedges,
            [f"{label} ({value})" for label, value in zip(industry_counts.index, industry_counts.values)],
            title="Industry",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=9,
        )
        ax.set_title("Industry Mix in the Cleaned Dataset", fontsize=15, fontweight="bold", pad=12)
        fig.tight_layout()
        fig.savefig(IMAGES_DIR / "industry_pie.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def save_clean_data(df: pd.DataFrame) -> dict[str, object]:
    """Persist the cleaned dataset and derived summaries."""
    return write_data_artifacts(df, DATA_DIR)


def main() -> None:
    data = load_and_clean_data(DATA_DIR)
    save_clean_data(data)
    generate_cover_image(data)
    perform_eda(data)
    print(f"Prepared {len(data):,} cleaned rows and refreshed visual artifacts.")


if __name__ == "__main__":
    main()
