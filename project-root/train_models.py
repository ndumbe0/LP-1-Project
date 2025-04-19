import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import os
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = "F:\\school\\Azubi Africa\\LP1 Data Analytics Project\\LP-1-Project\\project-root\\models"
TRAINING_DATA_FILE = "F:\\school\\Azubi Africa\\LP1 Data Analytics Project\\LP-1-Project\\data\\trainingdata.csv"

def ensure_directory_exists(directory: str) -> None:
    """Create directory if it doesn't exist."""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        raise

def validate_data_types(df: pd.DataFrame, expected_types: Dict[str, Any]) -> bool:
    """Validate that DataFrame columns match expected data types."""
    for col, expected_type in expected_types.items():
        if col not in df.columns:
            logger.error(f"Column {col} not found in DataFrame")
            return False
        if not all(isinstance(x, expected_type) for x in df[col].dropna()):
            logger.error(f"Column {col} does not match expected type {expected_type}")
            return False
    return True

def evaluate_regression_model(model, X_test, y_test, model_name: str) -> Dict[str, float]:
    """Evaluate regression model and return metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
    }
    logger.info(f"\n{model_name} Regression Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    return metrics

def evaluate_classification_model(model, X_test, y_test, model_name: str) -> Dict[str, float]:
    """Evaluate classification model and return metrics."""
    y_pred = model.predict(X_test)
    
    # Handle zero_division warnings
    zero_division_param = 0  # You can set this to 1 if you want to consider them as 1.0
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=zero_division_param),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=zero_division_param),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=zero_division_param),
    }
    logger.info(f"\n{model_name} Classification Metrics:")
    logger.info(classification_report(y_test, y_pred, zero_division=zero_division_param))
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    return metrics

def train_funding_model(training_data_file: str) -> None:
    """Trains a model to predict funding amount with enhanced features."""
    try:
        logger.info("Starting funding model training")
        
        # Load and validate data
        if not os.path.exists(training_data_file):
            raise FileNotFoundError(f"Training data file not found: {training_data_file}")
        
        df = pd.read_csv(training_data_file)
        
        # Data validation
        required_columns = ['Amount in ($)', 'Founded_scaled', 'Industry In', 'Head Quarter']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in dataset")
        
        # Feature engineering
        current_year = datetime.now().year
        df['Company Age'] = current_year - df['Founded_scaled']
        df['Log Amount'] = np.log1p(df['Amount in ($)'])
        
        # Check for NaN in target variable
        if df['Log Amount'].isna().any():
            logger.warning(f"Found {df['Log Amount'].isna().sum()} NaN values in target variable. Dropping these rows.")
            df = df.dropna(subset=['Log Amount'])
            logger.info(f"Data shape after dropping NaN values: {df.shape}")
        
        # Prepare features and target
        numeric_features = ['Founded_scaled', 'Company Age']
        categorical_features = ['Industry In', 'Head Quarter']
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Define model pipeline with hyperparameter tuning
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        # Hyperparameter grid
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5]
        }
        
        # Split data
        X = df[numeric_features + categorical_features]
        y = df['Log Amount']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Train with GridSearchCV
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1,
            error_score='raise')
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_regression_model(
            grid_search.best_estimator_, X_test, y_test, "Funding Prediction")
        
        # Save model
        ensure_directory_exists(OUTPUT_DIR)
        model_path = os.path.join(OUTPUT_DIR, 'funding_model.joblib')
        joblib.dump(grid_search.best_estimator_, model_path)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Funding model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error in training funding model: {e}", exc_info=True)
        raise

def train_success_model(training_data_file: str) -> None:
    """Trains a model to predict startup success with enhanced features."""
    try:
        logger.info("Starting success model training")
        
        # Load and validate data
        if not os.path.exists(training_data_file):
            raise FileNotFoundError(f"Training data file not found: {training_data_file}")
        
        df = pd.read_csv(training_data_file)
        
        # Data validation
        required_columns = ['Amount in ($)', 'Industry In', 'Founded_scaled', 'Head Quarter']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in dataset")
        
        # Feature engineering
        current_year = datetime.now().year
        df['Company Age'] = current_year - df['Founded_scaled']
        df['Success'] = (df['Amount in ($)'] > df['Amount in ($)'].median()).astype(int)
        
        # Prepare features and target
        numeric_features = ['Founded_scaled', 'Company Age']
        categorical_features = ['Industry In', 'Head Quarter']
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Define model pipeline with hyperparameter tuning
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Hyperparameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__class_weight': [None, 'balanced']
        }
        
        # Split data
        X = df[numeric_features + categorical_features]
        y = df['Success']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train with GridSearchCV
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_classification_model(
            grid_search.best_estimator_, X_test, y_test, "Success Prediction")
        
        # Save model
        ensure_directory_exists(OUTPUT_DIR)
        model_path = os.path.join(OUTPUT_DIR, 'success_model.joblib')
        joblib.dump(grid_search.best_estimator_, model_path)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Success model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error in training success model: {e}", exc_info=True)
        raise

def train_industry_model(training_data_file: str) -> None:
    """Trains a model to classify the industry based on 'AboutCompany'."""
    try:
        logger.info("Starting industry model training")
        
        # Load and validate data
        if not os.path.exists(training_data_file):
            raise FileNotFoundError(f"Training data file not found: {training_data_file}")
        
        df = pd.read_csv(training_data_file)
        
        # Data validation
        required_columns = ['Industry In', 'AboutCompany']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in dataset")
        
        # Clean and prepare data
        df = df.dropna(subset=['Industry In', 'AboutCompany'])
        df['AboutCompany'] = df['AboutCompany'].str.lower().str.strip()
        
        # Filter out rare industry classes (those with fewer than 5 samples)
        min_samples_per_class = 5
        value_counts = df['Industry In'].value_counts()
        rare_classes = value_counts[value_counts < min_samples_per_class].index
        df = df[~df['Industry In'].isin(rare_classes)]
        
        if len(df) == 0:
            raise ValueError("No data remaining after filtering rare industry classes")
            
        logger.info(f"Training on {len(df)} samples across {len(df['Industry In'].unique())} industry classes")
        
        # Split data
        X = df['AboutCompany']
        y = df['Industry In']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Check if any class has too few samples for stratified split
        class_counts = y.value_counts()
        if any(class_counts < 5):
            logger.warning(f"Some classes have very few samples: {class_counts[class_counts < 5]}")
            logger.warning("Consider merging similar rare categories or using a different evaluation strategy")
        
        # Define model pipeline with hyperparameter tuning
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Hyperparameter grid
        param_grid = {
            'tfidf__max_features': [3000, 5000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__n_estimators': [100],
            'classifier__max_depth': [None, 10],
            'classifier__class_weight': ['balanced']  # Always use balanced for imbalanced classes
        }
        
        # Reduce cv if some classes have too few samples
        min_class_count = class_counts.min()
        cv_value = min(5, min_class_count)  # Don't use more folds than the smallest class count
        
        # Train with GridSearchCV
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv_value, 
            scoring='f1_weighted', 
            n_jobs=-1,
            error_score='raise')
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_classification_model(
            grid_search.best_estimator_, X_test, y_test, "Industry Classification")
        
        # Save model
        ensure_directory_exists(OUTPUT_DIR)
        model_path = os.path.join(OUTPUT_DIR, 'industry_model.joblib')
        joblib.dump(grid_search.best_estimator_, model_path)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Industry model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error in training industry model: {e}", exc_info=True)
        raise

def main():
    try:
        logger.info("Starting model training pipeline")
        
        # Ensure output directory exists
        ensure_directory_exists(OUTPUT_DIR)
        
        # Train models
        train_funding_model(TRAINING_DATA_FILE)
        train_success_model(TRAINING_DATA_FILE)
        train_industry_model(TRAINING_DATA_FILE)
        
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error("Model training failed", exc_info=True)
        raise

if __name__ == "__main__":
    main()