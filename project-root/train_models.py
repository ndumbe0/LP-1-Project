import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

def train_funding_model(training_data_file):
    """Trains a model to predict funding amount."""
    try:
        df = pd.read_csv(training_data_file)
        if 'Amount' not in df.columns or 'Founded' not in df.columns:
            raise KeyError("Required columns 'Amount' or 'Founded' not found in the dataset.")
        
        df = df.dropna(subset=['Amount', 'Founded'])
        X = df[['Founded']]  # Example feature: Founded year
        y = df['Amount']  # Target variable: Funding Amount
        X = X.fillna(X.mean()) # Fills any empty values with mean of column
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()  # Or any other regression model
        model.fit(X_train, y_train)

        joblib.dump(model, 'funding_model.joblib')
        print("Funding model trained and saved as funding_model.joblib")
    except Exception as e:
        print(f"Error in training funding model: {e}")

def train_success_model(training_data_file):
    """Trains a model to predict startup success (example)."""
    try:
        df = pd.read_csv(training_data_file)
        if 'Amount' not in df.columns or 'Industry' not in df.columns or 'Founded' not in df.columns:
            raise KeyError("Required columns 'Amount', 'Industry', or 'Founded' not found in the dataset.")
        
        df['Success'] = (df['Amount'] > df['Amount'].median()).astype(int) # Example: Success if funding is above median
        df = df.dropna(subset=['Industry', 'Founded', 'Success'])
        X = df[['Industry', 'Founded']]
        y = df['Success']
        X = pd.get_dummies(X, columns=['Industry'], drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)  # Or any other classification model
        model.fit(X_train, y_train)

        joblib.dump(model, 'success_model.joblib')
        print("Startup success model trained and saved as success_model.joblib")
    except Exception as e:
        print(f"Error in training success model: {e}")

def train_industry_model(training_data_file):
    """Trains a model to classify the industry based on 'AboutCompany'."""
    try:
        df = pd.read_csv(training_data_file)
        if 'Industry' not in df.columns or 'AboutCompany' not in df.columns:
            raise KeyError("Required columns 'Industry' or 'AboutCompany' not found in the dataset.")
        
        df = df.dropna(subset=['Industry', 'AboutCompany'])
        X = df['AboutCompany']  # Text data
        y = df['Industry']  # Target variable: Industry
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a pipeline for text vectorization and classification
        model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', RandomForestClassifier(random_state=42))  # Or any other classifier
        ])
        model.fit(X_train, y_train)

        joblib.dump(model, 'industry_model.joblib')
        print("Industry classification model trained and saved as industry_model.joblib")
    except Exception as e:
        print(f"Error in training industry model: {e}")

def main():
    # You may need to adjust the file paths
    training_data_file = "F:\\school\\Azubi Africa\\LP1 Data Analytics Project\\LP-1-Project\\data\\trainingdata.csv"

    train_funding_model(training_data_file)
    train_success_model(training_data_file)
    train_industry_model(training_data_file)

if __name__ == "__main__":
    main()