# train_models.py
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

#  Load and preprocess data for Funding/Success models ---

file_path = "F:\\school\\Azubi Africa\\LP1 Data Analytics Project\\LP-1-Project\\data\\Aba3_cleaned.csv"
df = pd.read_csv(file_path)

# Handle missing values
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
categorical_cols = ['Head_Quarter', 'Industry']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Map RoundSeries to numerical labels
round_series_mapping = {
    'Pre-seed': 0, 'Seed': 1, 'Pre-series A': 2, 'Series A': 3,
    'Series B': 4, 'Series C': 5, 'Series D': 6, 'Series E': 7,
    'Debt': 8, 'Bridge': 9
}
df['RoundSeries_Numerical'] = df['RoundSeries'].map(round_series_mapping).fillna(-1)

# Scale numerical features
scaler = StandardScaler()
df[['Founded_scaled', 'RoundSeries_scaled']] = scaler.fit_transform(df[['Founded', 'RoundSeries_Numerical']])

# Split data for Funding/Success models
feature_columns = ['Founded_scaled', 'RoundSeries_scaled', 'Head_Quarter', 'Industry']
X = df[feature_columns]
y_amount = df['Amount']  # Target for Funding model

# Split into train/test
X_train, X_test, y_train_amount, y_test_amount = train_test_split(
    X, y_amount, test_size=0.2, random_state=42
)

# Create binary target for Success model
success_threshold = 1_000_000
y_train_success = (y_train_amount > success_threshold).astype(int)
y_test_success = (y_test_amount > success_threshold).astype(int)

# --- Preprocess data for Industry model ---
df_industry = df.dropna(subset=['About_Company', 'Industry'])
X_industry = df_industry['About_Company']
y_industry = df_industry['Industry']

# Split into train/test
X_train_ind, X_test_ind, y_train_ind, y_test_ind = train_test_split(
    X_industry, y_industry, test_size=0.2, random_state=42
)

# --- Model 1: Funding Prediction ---
# Imputer and model training
funding_imputer = SimpleImputer(strategy='median')
X_train_funding = funding_imputer.fit_transform(X_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_funding, y_train_amount)

# Save artifacts
joblib.dump(funding_imputer, 'models/funding_imputer.joblib')
joblib.dump(rf_model, 'models/funding_model.joblib')

# --- Model 2: Startup Success ---
# Pipeline with imputer, scaler, and classifier
success_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])
success_pipeline.fit(X_train, y_train_success)
joblib.dump(success_pipeline, 'models/success_pipeline.joblib')

# --- Model 3: Industry Classification ---
# Pipeline with TF-IDF and classifier
industry_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])
industry_pipeline.fit(X_train_ind, y_train_ind)
joblib.dump(industry_pipeline, 'models/industry_pipeline.joblib')

print("All models trained and saved successfully!")