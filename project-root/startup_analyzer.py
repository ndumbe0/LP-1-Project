import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config
st.set_page_config(
    page_title="Startup Analyzer",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if 'cleaned_data' not in st.session_state:
    st.session_state['cleaned_data'] = None

# Data cleaning function
def clean_data(df):
    # Convert numeric columns
    df['Year Founded'] = pd.to_numeric(df['Year Founded'], errors='coerce')
    df['Amount in ($)'] = pd.to_numeric(df['Amount in ($)'], errors='coerce')
    
    # Drop rows with missing essential values
    df = df.dropna(subset=['Amount in ($)', 'Year Founded'])
    
    # Fill remaining missing values
    df = df.fillna('Unknown')
    
    # Standardize column names
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    df.rename(columns={
        'Company_Name': 'CompanyName',
        'Head_Quarter': 'Head Quarter',
        'About_Company': 'AboutCompany'
    }, inplace=True)
    
    # Select relevant columns
    return df[[
        'CompanyName', 'Year Founded', 'Head Quarter', 'Industry In',
        'AboutCompany', 'Founders', 'Investor', 'Amount in ($)',
        'Funding Round/Series'
    ]]

# Home Page
def home_page():
    st.title("🏠 Startup Data Upload")
    uploaded_file = st.file_uploader("Upload your startup data (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Original Data:")
            st.dataframe(df)
            
            cleaned_df = clean_data(df.copy())
            st.write("### Cleaned Data:")
            st.dataframe(cleaned_df)
            
            st.session_state['cleaned_data'] = cleaned_df
            st.success("Data uploaded and cleaned successfully!")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload a CSV file to begin")

# Funding Predictor Page
def funding_predictor_page():
    st.title("💰 Funding Predictor")
    
    if st.session_state['cleaned_data'] is not None:
        df = st.session_state['cleaned_data']
        st.write("### Data Preview:")
        st.dataframe(df.head())
        
        try:
            # Load model (in a real app, this would be a proper path)
            model = joblib.load('funding_model.joblib')
            
            # Prepare features
            X = df[['Year Founded']]
            X = X.fillna(X.mean())
            
            # Make predictions
            predictions = model.predict(X)
            df['Predicted Funding'] = predictions
            
            # Show results
            st.write("### Funding Predictions")
            st.dataframe(df[['CompanyName', 'Year Founded', 'Predicted Funding']])
            
            # Visualizations
            st.write("### Funding Distribution")
            fig = px.histogram(df, x='Predicted Funding', nbins=30)
            st.plotly_chart(fig)
            
            st.write("### Year Founded vs. Predicted Funding")
            fig = px.scatter(df, x='Year Founded', y='Predicted Funding')
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        st.info("Please upload data on the Home page first")

# Startup Success Page
def startup_success_page():
    st.title("✅ Startup Success Predictor")
    
    if st.session_state['cleaned_data'] is not None:
        df = st.session_state['cleaned_data']
        st.write("### Data Preview:")
        st.dataframe(df.head())
        
        try:
            # Load model
            model = joblib.load('success_model.joblib')
            
            # Prepare features
            X = df[['Industry In', 'Year Founded']]
            X = pd.get_dummies(X, columns=['Industry In'], drop_first=True)
            X = X.fillna(X.mean())
            
            # Make predictions
            predictions = model.predict(X)
            df['Success Probability'] = predictions
            
            # Show results
            st.write("### Success Predictions")
            st.dataframe(df[['CompanyName', 'Industry In', 'Year Founded', 'Success Probability']])
            
            # Visualization
            st.write("### Success Probability Distribution")
            fig, ax = plt.subplots()
            sns.histplot(data=df, x='Success Probability', bins=20, ax=ax)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        st.info("Please upload data on the Home page first")

# Industry Classifier Page
def industry_classifier_page():
    st.title("🏭 Industry Classifier")
    
    if st.session_state['cleaned_data'] is not None:
        df = st.session_state['cleaned_data']
        st.write("### Data Preview:")
        st.dataframe(df.head())
        
        try:
            # Load model and vectorizer
            model = joblib.load('industry_model.joblib')
            vectorizer = TfidfVectorizer()
            
            # Prepare features
            X = vectorizer.fit_transform(df['AboutCompany'])
            
            # Make predictions
            predictions = model.predict(X)
            industry_mapping = {
                0: 'Technology',
                1: 'Healthcare',
                2: 'Finance',
                3: 'Retail',
                4: 'Manufacturing'
            }
            df['Predicted Industry'] = [industry_mapping.get(p, 'Other') for p in predictions]
            
            # Show results
            st.write("### Industry Predictions")
            st.dataframe(df[['CompanyName', 'Industry In', 'Predicted Industry']])
            
            # Visualization
            st.write("### Industry Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(data=df, x='Predicted Industry', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error in classification: {e}")
    else:
        st.info("Please upload data on the Home page first")

# Main App
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Home",
        "💰 Funding Predictor",
        "✅ Startup Success",
        "🏭 Industry Classifier"
    ])

    if page == "🏠 Home":
        home_page()
    elif page == "💰 Funding Predictor":
        funding_predictor_page()
    elif page == "✅ Startup Success":
        startup_success_page()
    elif page == "🏭 Industry Classifier":
        industry_classifier_page()

if __name__ == "__main__":
    main()