"""Prediction helpers used by the Streamlit app and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MODELS_DIR, REFERENCE_YEAR
from .data import clean_startup_dataframe
from .model_io import load_bundle


MODEL_FILES = {
    "funding": "funding_pipeline.pkl",
    "success": "success_pipeline.pkl",
    "industry": "industry_pipeline.pkl",
}


def load_model_bundles(models_dir: Path = MODELS_DIR) -> dict[str, dict[str, Any]]:
    """Load all available production model bundles."""
    bundles: dict[str, dict[str, Any]] = {}
    for name, filename in MODEL_FILES.items():
        path = models_dir / filename
        if path.exists():
            bundles[name] = load_bundle(path)
    return bundles


def startup_input_frame(
    *,
    company_name: str,
    year_founded: int,
    funding_year: int = REFERENCE_YEAR,
    head_quarter: str,
    industry: str,
    about_company: str,
    funding_round: str,
    founders: str = "Unknown",
    investor: str = "Unknown",
) -> pd.DataFrame:
    """Create a one-row canonical dataframe for prediction."""
    return pd.DataFrame(
        [
            {
                "CompanyName": company_name,
                "Year Founded": year_founded,
                "Funding Year": funding_year,
                "Founded Imputed": False,
                "Head Quarter": head_quarter,
                "Industry In": industry,
                "AboutCompany": about_company,
                "Founders": founders,
                "Investor": investor,
                "Amount in ($)": np.nan,
                "Funding Round/Series": funding_round,
            }
        ]
    )


def clean_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize user-supplied rows without requiring known funding amount."""
    return clean_startup_dataframe(df, require_amount=False)


def predict_funding(bundle: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    predictions = bundle["model"].predict(clean_prediction_frame(df))
    return np.clip(predictions.astype(float), 0, None)


def predict_success(bundle: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    model = bundle["model"]
    clean = clean_prediction_frame(df)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(clean)[:, 1]
    raw = model.decision_function(clean)
    return 1 / (1 + np.exp(-raw))


def classify_industry(bundle: dict[str, Any], descriptions: list[str] | pd.Series) -> np.ndarray:
    return bundle["model"].predict(pd.Series(descriptions).fillna("").astype(str))


def find_similar_startups(df: pd.DataFrame, startup: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Rank historical startups by simple market similarity."""
    if df.empty:
        return df
    candidate = clean_prediction_frame(startup).iloc[0]
    scored = df.copy()
    scored["_score"] = 0.0
    scored["_score"] += (scored["Industry In"].str.lower() == str(candidate["Industry In"]).lower()) * 4
    scored["_score"] += (scored["Head Quarter"].str.lower() == str(candidate["Head Quarter"]).lower()) * 2
    scored["_score"] += (
        scored["Funding Round/Series"].str.lower() == str(candidate["Funding Round/Series"]).lower()
    ) * 2
    year_distance = (scored["Year Founded"] - int(candidate["Year Founded"])).abs()
    scored["_score"] += (1 / (1 + year_distance)).astype(float)
    return (
        scored.sort_values(["_score", "Amount in ($)"], ascending=[False, False])
        .drop(columns=["_score"])
        .head(top_n)
    )
