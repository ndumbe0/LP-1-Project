"""Model training routines for funding, readiness, and industry prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, MODELS_DIR, NUMERIC_FEATURES, RANDOM_STATE
from .features import StartupFeatureBuilder
from .model_io import save_bundle


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def _startup_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", StartupFeatureBuilder()),
            ("preprocessor", _preprocessor()),
            ("model", estimator),
        ]
    )


def _clean_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["Amount in ($)"] = pd.to_numeric(clean["Amount in ($)"], errors="coerce")
    clean = clean.dropna(subset=["Amount in ($)", "Year Founded"])
    clean = clean[clean["Amount in ($)"] > 0]
    if len(clean) < 50:
        raise ValueError("At least 50 funded startup rows are required for reliable training.")
    return clean.reset_index(drop=True)


def _funding_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_funding_model(df: pd.DataFrame, output_dir: Path = MODELS_DIR) -> dict[str, Any]:
    """Train and persist the best funding amount regression pipeline."""
    clean = _clean_target_frame(df)
    X = clean.drop(columns=["Amount in ($)"])
    y = clean["Amount in ($)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    candidates = {
        "Ridge Log-Target": _startup_pipeline(
            TransformedTargetRegressor(regressor=Ridge(alpha=10.0), func=np.log1p, inverse_func=np.expm1)
        ),
        "Random Forest": _startup_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = np.clip(model.predict(X_test), 0, None)
        metrics = _funding_metrics(y_test, predictions)
        metrics["CV_R2"] = float(cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean())
        results[name] = {key: round(value, 4) for key, value in metrics.items()}
        fitted[name] = model

    best_name = min(results, key=lambda name: results[name]["RMSE"])
    save_bundle(
        {
            "kind": "funding_regression",
            "model": fitted[best_name],
            "best_model": best_name,
            "metrics": results[best_name],
            "all_metrics": results,
        },
        output_dir / "funding_pipeline.pkl",
    )
    return {"best_model": best_name, "metrics": results[best_name], "candidates": results}


def _success_labels(amounts: pd.Series) -> pd.Series:
    threshold = amounts.median()
    return (amounts >= threshold).astype(int)


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(set(y_true)) > 1:
        metrics["ROC_AUC"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def train_success_model(df: pd.DataFrame, output_dir: Path = MODELS_DIR) -> dict[str, Any]:
    """Train and persist a funding-readiness classification model."""
    clean = _clean_target_frame(df)
    X = clean.drop(columns=["Amount in ($)"])
    y = _success_labels(clean["Amount in ($)"])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidates = {
        "Logistic Regression": _startup_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "Random Forest": _startup_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": _startup_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE)),
    }

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = _classification_metrics(y_test, predictions, probabilities)
        results[name] = {key: round(value, 4) for key, value in metrics.items()}
        fitted[name] = model

    best_name = max(results, key=lambda name: (results[name]["F1"], results[name]["Accuracy"]))
    save_bundle(
        {
            "kind": "funding_readiness",
            "model": fitted[best_name],
            "best_model": best_name,
            "metrics": results[best_name],
            "all_metrics": results,
            "positive_class": "Above median funding",
        },
        output_dir / "success_pipeline.pkl",
    )
    return {"best_model": best_name, "metrics": results[best_name], "candidates": results}


def _prepare_industry_frame(df: pd.DataFrame, min_samples: int = 5) -> pd.DataFrame:
    clean = df.dropna(subset=["AboutCompany", "Industry In"]).copy()
    clean["AboutCompany"] = clean["AboutCompany"].astype(str).str.strip()
    clean = clean[(clean["AboutCompany"] != "") & (clean["AboutCompany"] != "Unknown")]
    counts = clean["Industry In"].value_counts()
    rare = counts[counts < min_samples].index
    clean["Industry In"] = clean["Industry In"].where(~clean["Industry In"].isin(rare), "Other")
    return clean.reset_index(drop=True)


def train_industry_model(df: pd.DataFrame, output_dir: Path = MODELS_DIR) -> dict[str, Any]:
    """Train and persist a text classifier for startup industry prediction."""
    clean = _prepare_industry_frame(df)
    if len(clean) < 50 or clean["Industry In"].nunique() < 2:
        raise ValueError("Industry training requires at least 50 usable descriptions across 2 classes.")

    X = clean["AboutCompany"]
    y = clean["Industry In"]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "Accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "F1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "Classes": int(y.nunique()),
        "Training Rows": int(len(clean)),
    }

    save_bundle(
        {
            "kind": "industry_classifier",
            "model": model,
            "best_model": "TF-IDF Logistic Regression",
            "metrics": metrics,
        },
        output_dir / "industry_pipeline.pkl",
    )
    return {"best_model": "TF-IDF Logistic Regression", "metrics": metrics}


def train_all(df: pd.DataFrame, output_dir: Path = MODELS_DIR) -> dict[str, Any]:
    """Train all production model bundles and return a serializable summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "funding": train_funding_model(df, output_dir),
        "success": train_success_model(df, output_dir),
        "industry": train_industry_model(df, output_dir),
    }
    return summary


def write_training_summary(summary: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
