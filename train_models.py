"""
Model Training Pipeline for Startup Funding Prediction.
Trains 3+ models with hyperparameter tuning, evaluates, and saves best model.
"""
import pandas as pd
import numpy as np
import os
import hashlib
import logging
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def _write_model_hash(model_path):
    """Generate and save SHA256 hash file for model integrity verification."""
    sha256 = hashlib.sha256()
    with open(model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    hash_path = model_path + '.sha256'
    with open(hash_path, 'w') as f:
        f.write(sha256.hexdigest())
    logger.info(f"Hash saved to {hash_path}")


def load_clean_data():
    """Load cleaned startup funding dataset."""
    path = os.path.join(DATA_DIR, 'startup_funding_clean.csv')
    if not os.path.exists(path):
        logger.error(f"Clean data not found at {path}. Run eda_cleaning.py first.")
        return None
    df = pd.read_csv(path)
    logger.info(f"Loaded clean data: {df.shape}")
    return df


def preprocess_for_funding(df):
    """Preprocess data for funding amount regression."""
    df = df.copy()

    df['Company Age'] = datetime.now().year - df['Year Founded']
    df['Log Amount'] = np.log1p(df['Amount in ($)'])

    df.dropna(subset=['Log Amount'], inplace=True)
    df = df[df['Amount in ($)'] > 0]

    cat_cols = ['Industry In', 'Head Quarter', 'Funding Round/Series']
    for c in cat_cols:
        if c in df.columns:
            le = LabelEncoder()
            df[f'{c}_enc'] = le.fit_transform(df[c].astype(str))

    feature_cols = ['Year Founded', 'Company Age']
    for c in cat_cols:
        col = f'{c}_enc'
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols]
    y = df['Log Amount']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_cols


def preprocess_for_success(df):
    """Preprocess data for startup success classification."""
    df = df.copy()

    df['Company Age'] = datetime.now().year - df['Year Founded']
    median_amount = df['Amount in ($)'].median()
    df['Success'] = (df['Amount in ($)'] > median_amount).astype(int)

    cat_cols = ['Industry In', 'Head Quarter', 'Funding Round/Series']
    for c in cat_cols:
        if c in df.columns:
            le = LabelEncoder()
            df[f'{c}_enc'] = le.fit_transform(df[c].astype(str))

    feature_cols = ['Year Founded', 'Company Age']
    for c in cat_cols:
        col = f'{c}_enc'
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols]
    y = df['Success']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_cols


def preprocess_for_industry(df):
    """Preprocess data for industry classification from text."""
    df = df.copy()
    df = df.dropna(subset=['AboutCompany', 'Industry In'])
    df['AboutCompany'] = df['AboutCompany'].astype(str).str.lower().str.strip()

    min_samples = 5
    counts = df['Industry In'].value_counts()
    rare = counts[counts < min_samples].index
    df['Industry In'] = df['Industry In'].apply(lambda x: 'Other' if x in rare else x)

    return df['AboutCompany'], df['Industry In']


def train_funding_model(df):
    """Train regression models for funding amount prediction."""
    logger.info("=" * 60)
    logger.info("FUNDING AMOUNT PREDICTION (Regression)")
    logger.info("=" * 60)

    X, y, scaler, features = preprocess_for_funding(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge': Ridge(random_state=42)
    }

    results = {}
    best_model = None
    best_score = float('inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        logger.info(f"\n{name}:")
        logger.info(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        logger.info(f"  CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        if rmse < best_score:
            best_score = rmse
            best_model = model

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    logger.info("\n--- Tuning Random Forest ---")
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    logger.info(f"Best params: {grid.best_params_}")

    y_pred = grid.best_estimator_.predict(X_test)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_tuned = r2_score(y_test, y_pred)
    logger.info(f"Tuned RF - RMSE: {rmse_tuned:.4f}, R2: {r2_tuned:.4f}")

    if rmse_tuned < best_score:
        best_model = grid.best_estimator_

    pipeline = {
        'model': best_model,
        'scaler': scaler,
        'features': features,
        'type': 'regression'
    }
    model_path = os.path.join(MODELS_DIR, 'funding_pipeline.pkl')
    joblib.dump(pipeline, model_path)
    _write_model_hash(model_path)
    logger.info(f"Best funding model saved. R2: {max(r2_tuned, r2_score(y_test, best_model.predict(X_test))):.4f}")

    return results


def train_success_model(df):
    """Train classification models for startup success prediction."""
    logger.info("\n" + "=" * 60)
    logger.info("STARTUP SUCCESS PREDICTION (Classification)")
    logger.info("=" * 60)

    X, y, scaler, features = preprocess_for_success(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    results = {}
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        if name == 'Gradient Boosting':
            model.fit(X_train_res, y_train_res)
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        results[name] = {'Accuracy': acc, 'F1': f1, 'Precision': prec, 'Recall': rec}
        logger.info(f"\n{name}:")
        logger.info(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'class_weight': [None, 'balanced']
    }
    logger.info("\n--- Tuning Random Forest Classifier ---")
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train_res, y_train_res)
    logger.info(f"Best params: {grid.best_params_}")

    y_pred = grid.best_estimator_.predict(X_test)
    f1_tuned = f1_score(y_test, y_pred)
    acc_tuned = accuracy_score(y_test, y_pred)
    logger.info(f"Tuned RF - Accuracy: {acc_tuned:.4f}, F1: {f1_tuned:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    if f1_tuned > best_f1:
        best_model = grid.best_estimator_

    pipeline = {
        'model': best_model,
        'scaler': scaler,
        'features': features,
        'type': 'classification',
        'classes': ['Not Successful', 'Successful']
    }
    model_path = os.path.join(MODELS_DIR, 'success_pipeline.pkl')
    joblib.dump(pipeline, model_path)
    _write_model_hash(model_path)
    logger.info("Best success model saved.")

    return results


def train_industry_model(df):
    """Train text classification model for industry prediction."""
    logger.info("\n" + "=" * 60)
    logger.info("INDUSTRY CLASSIFICATION (Text Classification)")
    logger.info("=" * 60)

    X, y = preprocess_for_industry(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    param_grid = {
        'tfidf__max_features': [2000, 3000],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 20]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_train, y_train)
    logger.info(f"Best params: {grid.best_params_}")

    y_pred = grid.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    logger.info(f"Accuracy: {acc:.4f}, Weighted F1: {f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    pipeline = {
        'model': grid.best_estimator_,
        'type': 'text_classification'
    }
    model_path = os.path.join(MODELS_DIR, 'industry_pipeline.pkl')
    joblib.dump(pipeline, model_path)
    _write_model_hash(model_path)
    logger.info("Best industry model saved.")

    return {'Accuracy': acc, 'F1': f1}


def main():
    logger.info("Starting model training pipeline...")
    df = load_clean_data()
    if df is None:
        return

    funding_results = train_funding_model(df)
    success_results = train_success_model(df)
    industry_results = train_industry_model(df)

    summary = {
        'funding': {k: {sk: round(sv, 4) for sk, sv in v.items()} for k, v in funding_results.items()},
        'success': {k: {sk: round(sv, 4) for sk, sv in v.items()} for k, v in success_results.items()},
        'industry': {k: round(v, 4) for k, v in industry_results.items()}
    }
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(json.dumps(summary, indent=2))

    with open(os.path.join(BASE_DIR, 'training_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
