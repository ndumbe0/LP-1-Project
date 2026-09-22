"""Feature engineering used consistently by training and inference."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, REFERENCE_YEAR
from .data import normalize_location, sanitize_csv_cell


def build_feature_frame(df: pd.DataFrame, *, reference_year: int = REFERENCE_YEAR) -> pd.DataFrame:
    """Create model features from canonical startup records."""
    records = df.copy()

    for col in ["Year Founded", "Funding Year", "Industry In", "Head Quarter", "Funding Round/Series", "AboutCompany"]:
        if col not in records:
            records[col] = None

    records["Year Founded"] = pd.to_numeric(records["Year Founded"], errors="coerce")
    records["Funding Year"] = pd.to_numeric(records["Funding Year"], errors="coerce").fillna(reference_year)
    records["Year Founded"] = records["Year Founded"].fillna(records["Funding Year"])
    records["Company Age"] = (reference_year - records["Year Founded"]).clip(lower=0, upper=120)
    records["Funding Lag"] = (records["Funding Year"] - records["Year Founded"]).clip(lower=0, upper=80)
    records["Description Length"] = records["AboutCompany"].fillna("").astype(str).str.len().clip(upper=2000)

    for col in CATEGORICAL_FEATURES:
        if col == "Head Quarter":
            records[col] = records[col].apply(normalize_location)
        else:
            records[col] = records[col].apply(sanitize_csv_cell)

    return records[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


class StartupFeatureBuilder(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer for startup funding feature engineering."""

    def __init__(self, reference_year: int = REFERENCE_YEAR):
        self.reference_year = reference_year

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return build_feature_frame(X, reference_year=self.reference_year)
