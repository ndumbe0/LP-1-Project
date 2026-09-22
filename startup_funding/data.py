"""Data loading, cleaning, validation, and derived artifact helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import CANONICAL_COLUMNS, DATA_DIR, RANDOM_STATE, RAW_DATASETS, REFERENCE_YEAR


_HEADER_RE = re.compile(r"[^a-z0-9]+")
_AMOUNT_RE = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_DANGEROUS_CSV_PREFIXES = ("=", "+", "-", "@", "\t", "\n", "\r", "|")

COLUMN_ALIASES = {
    "company": "CompanyName",
    "companybrand": "CompanyName",
    "companyname": "CompanyName",
    "startup": "CompanyName",
    "startupname": "CompanyName",
    "brand": "CompanyName",
    "founded": "Year Founded",
    "foundedyear": "Year Founded",
    "yearfounded": "Year Founded",
    "headquarter": "Head Quarter",
    "headquarters": "Head Quarter",
    "location": "Head Quarter",
    "city": "Head Quarter",
    "sector": "Industry In",
    "industry": "Industry In",
    "industryin": "Industry In",
    "about": "AboutCompany",
    "aboutcompany": "AboutCompany",
    "description": "AboutCompany",
    "whatitdoes": "AboutCompany",
    "founder": "Founders",
    "founders": "Founders",
    "investor": "Investor",
    "investors": "Investor",
    "amount": "Amount in ($)",
    "amountdollar": "Amount in ($)",
    "amountusd": "Amount in ($)",
    "amountin": "Amount in ($)",
    "amountinusd": "Amount in ($)",
    "amountinrs": "Amount in ($)",
    "fundingamount": "Amount in ($)",
    "stage": "Funding Round/Series",
    "round": "Funding Round/Series",
    "series": "Funding Round/Series",
    "roundseries": "Funding Round/Series",
    "fundinground": "Funding Round/Series",
    "fundingroundseries": "Funding Round/Series",
    "fundingyear": "Funding Year",
    "sourceyear": "Funding Year",
}

LOCATION_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "new delhi": "New Delhi",
    "delhi": "New Delhi",
    "rajastan": "Rajasthan",
    "jaipur": "Jaipur",
}


def normalize_header(value: object) -> str:
    """Normalize a source column name to a stable lookup key."""
    text = str(value).replace("\ufeff", "").strip().lower()
    return _HEADER_RE.sub("", text)


def sanitize_csv_cell(value: object, *, default: str = "Unknown") -> str:
    """Return display-safe text and neutralize spreadsheet formulas."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default

    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown"}:
        return default
    if text[0] in _DANGEROUS_CSV_PREFIXES:
        return "'" + text
    return text


def parse_amount_to_usd(value: object) -> float:
    """Parse the project amount field into a USD numeric value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null", "undisclosed", "unknown", "-"}:
        return np.nan

    multiplier = 1.0
    if any(token in text for token in ("billion", "bn")):
        multiplier = 1_000_000_000.0
    elif re.search(r"\b(million|mn)\b", text):
        multiplier = 1_000_000.0
    elif re.search(r"\b(lakh|lakhs)\b", text):
        multiplier = 100_000.0
    elif re.search(r"\b(crore|crores|cr)\b", text):
        multiplier = 10_000_000.0

    compact = text.replace(",", "").replace("$", "").replace("usd", "")
    compact = compact.replace("₹", "").replace("rs.", "").replace("inr", "")
    match = _AMOUNT_RE.search(compact)
    if not match:
        return np.nan
    return float(match.group(1)) * multiplier


def normalize_location(value: object) -> str:
    """Use the city-level location where a long address is provided."""
    text = sanitize_csv_cell(value)
    if text == "Unknown":
        return text
    city = text.split(",")[0].strip()
    return LOCATION_ALIASES.get(city.lower(), city)


def rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Map the different source schemas into one canonical schema."""
    result = pd.DataFrame(index=df.index)
    for source_col in df.columns:
        key = normalize_header(source_col)
        if not key or key.startswith("unnamed"):
            continue
        target_col = COLUMN_ALIASES.get(key, str(source_col).replace("\ufeff", "").strip())
        series = df[source_col].replace(r"^\s*$", np.nan, regex=True)
        if target_col in result:
            result[target_col] = result[target_col].combine_first(series)
        else:
            result[target_col] = series
    return result


def clean_startup_dataframe(
    df: pd.DataFrame,
    *,
    source_year: int | None = None,
    require_amount: bool = True,
    reference_year: int = REFERENCE_YEAR,
) -> pd.DataFrame:
    """Clean one startup funding dataframe without assuming a single source schema."""
    clean = rename_to_canonical(df)

    if source_year is not None:
        clean["Funding Year"] = clean.get("Funding Year", source_year)
        clean["Funding Year"] = clean["Funding Year"].fillna(source_year)
    elif "Funding Year" not in clean:
        clean["Funding Year"] = np.nan

    for col in CANONICAL_COLUMNS:
        if col not in clean:
            clean[col] = np.nan

    clean["Amount in ($)"] = clean["Amount in ($)"].apply(parse_amount_to_usd)
    clean["Funding Year"] = pd.to_numeric(clean["Funding Year"], errors="coerce")
    clean["Year Founded"] = pd.to_numeric(clean["Year Founded"], errors="coerce")

    invalid_founded = clean["Year Founded"].gt(reference_year + 1) | clean["Year Founded"].lt(1900)
    clean.loc[invalid_founded, "Year Founded"] = np.nan
    clean["Founded Imputed"] = clean["Year Founded"].isna()
    clean["Year Founded"] = clean["Year Founded"].fillna(clean["Funding Year"])

    for col in ["CompanyName", "Industry In", "AboutCompany", "Founders", "Investor", "Funding Round/Series"]:
        clean[col] = clean[col].apply(sanitize_csv_cell)
    clean["Head Quarter"] = clean["Head Quarter"].apply(normalize_location)

    clean = clean.dropna(subset=["Year Founded"])
    if require_amount:
        clean = clean.dropna(subset=["Amount in ($)"])
        clean = clean[clean["Amount in ($)"] > 0]

    clean["Year Founded"] = clean["Year Founded"].round().astype(int)
    clean["Funding Year"] = clean["Funding Year"].fillna(clean["Year Founded"]).round().astype(int)
    clean["Founded Imputed"] = clean["Founded Imputed"].astype(bool)

    ordered = clean[CANONICAL_COLUMNS].copy()
    ordered = ordered.drop_duplicates().reset_index(drop=True)
    return ordered


def load_raw_datasets(data_dir: Path = DATA_DIR) -> list[pd.DataFrame]:
    """Load all tracked raw funding datasets with their source funding year."""
    frames: list[pd.DataFrame] = []
    for filename, funding_year in RAW_DATASETS.items():
        path = data_dir / filename
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frames.append(clean_startup_dataframe(frame, source_year=funding_year))
    return frames


def load_and_clean_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load raw datasets and return one validated startup funding table."""
    frames = load_raw_datasets(data_dir)
    clean = pd.concat(frames, ignore_index=True, sort=False)
    clean = clean.drop_duplicates(
        subset=["CompanyName", "Year Founded", "Funding Year", "Investor", "Amount in ($)"],
        keep="first",
    ).reset_index(drop=True)
    return clean


def read_clean_data(path: Path | None = None) -> pd.DataFrame:
    """Read the cleaned project dataset."""
    target = path or DATA_DIR / "startup_funding_clean.csv"
    return pd.read_csv(target)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV with stable LF line endings for cross-platform diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def _group_sum_mean(df: pd.DataFrame, group_cols: str | Iterable[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)["Amount in ($)"]
        .agg(deals="count", total_funding="sum", average_funding="mean", median_funding="median")
        .reset_index()
        .sort_values("total_funding", ascending=False)
    )
    return grouped


def write_data_artifacts(df: pd.DataFrame, data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Write cleaned, split, and summary CSV artifacts used by the app and README."""
    artifacts = {
        "clean": data_dir / "startup_funding_clean.csv",
        "training": data_dir / "trainingdata.csv",
        "testing": data_dir / "testingdata.csv",
        "funding_by_location": data_dir / "funding_by_location.csv",
        "average_funding_by_industry": data_dir / "average_funding_by_industry.csv",
        "funding_trends": data_dir / "funding_trends.csv",
        "yearly_funding": data_dir / "yearly_funding.csv",
        "funding_summary": data_dir / "funding_summary.csv",
        "funding_summary_2018_2020": data_dir / "funding_summary_2018_2020.csv",
        "industry_funding_changes": data_dir / "industry_funding_changes.csv",
    }

    write_csv(df, artifacts["clean"])

    training = df.sample(frac=0.8, random_state=RANDOM_STATE)
    testing = df.drop(training.index)
    write_csv(training.sort_index(), artifacts["training"])
    write_csv(testing.sort_index(), artifacts["testing"])

    write_csv(_group_sum_mean(df, "Head Quarter"), artifacts["funding_by_location"])
    write_csv(_group_sum_mean(df, "Industry In"), artifacts["average_funding_by_industry"])

    yearly = _group_sum_mean(df, "Funding Year").sort_values("Funding Year")
    write_csv(yearly, artifacts["funding_trends"])
    write_csv(yearly, artifacts["yearly_funding"])

    summary = pd.DataFrame([dataset_profile(df)])
    write_csv(summary, artifacts["funding_summary"])

    period = df[df["Funding Year"].between(2018, 2020)]
    write_csv(_group_sum_mean(period, "Funding Year").sort_values("Funding Year"), artifacts["funding_summary_2018_2020"])

    industry_year = (
        df.pivot_table(
            index="Industry In",
            columns="Funding Year",
            values="Amount in ($)",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .sort_values("Industry In")
    )
    industry_year.columns = [str(col) for col in industry_year.columns]
    write_csv(industry_year, artifacts["industry_funding_changes"])
    return artifacts


def dataset_profile(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Return high-level dataset quality and funding metrics."""
    amount = df["Amount in ($)"]
    founded_range = f"{int(df['Year Founded'].min())}-{int(df['Year Founded'].max())}"
    funding_range = f"{int(df['Funding Year'].min())}-{int(df['Funding Year'].max())}"
    return {
        "records": int(len(df)),
        "total_funding_usd": float(amount.sum()),
        "average_funding_usd": float(amount.mean()),
        "median_funding_usd": float(amount.median()),
        "industries": int(df["Industry In"].nunique()),
        "locations": int(df["Head Quarter"].nunique()),
        "founded_year_range": founded_range,
        "funding_year_range": funding_range,
        "founded_year_imputed_pct": round(float(df["Founded Imputed"].mean() * 100), 2),
    }
