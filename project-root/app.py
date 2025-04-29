import streamlit as st 
import sys
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


sys.path.append(os.path.join(os.path.dirname(__file__), "pages"))


st.set_page_config(page_title="Startup Analyzer", page_icon="📈", layout="wide")



st.set_page_config(page_title="Startup Analyzer", page_icon="📈")



st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Funding Predictor", "Startup Success", "Industry In Classifier"))

 

page_modules = {
    "Home": "1__Home",
    "Funding Predictor": "2__Funding_Predictor",
    "Startup Success": "3__Startup_Success",
    "Industry In Classifier": "4__Industry_Classifier",
}


if page == "Home":
    st.title("Welcome to the Startup Analyzer App 📈")
    st.write("""
    This app helps you analyze startup data, predict funding, assess startup success, and classify industries.
    Use the navigation bar on the left to explore the different features:
    - **Home**: Upload and clean your startup data.
    - **Funding Predictor**: Predict funding amounts based on startup data.
    - **Startup Success**: Predict the success probability of startups.
    - **Industry In Classifier**: Classify startups into industries based on their descriptions.
    """)
else:
    
    if page in page_modules:
        module_name = page_modules[page]
        module = __import__(module_name)  
        module.main()  


def predict_api(input_data):
    
    funding_model = joblib.load('models/funding_model.joblib')
    success_model = joblib.load('models/success_model.joblib')
    industry_model = joblib.load('models/industry_model.joblib')
    
    
    df = pd.DataFrame(input_data)
    
    # Funding prediction
    X_funding = df[['Year Founded']]
    X_funding = X_funding.fillna(X_funding.mean())
    funding_predictions = funding_model.predict(X_funding)
    
    # Success prediction
    X_success = df[['Industry In', 'Year Founded']]
    X_success = pd.get_dummies(X_success, columns=['Industry In'], drop_first=True)
    X_success = X_success.fillna(X_success.mean())
    success_predictions = success_model.predict(X_success)
    
    # Industry In classification
    vectorizer = TfidfVectorizer()
    X_industry = vectorizer.fit_transform(df['AboutCompany'])
    industry_predictions = industry_model.predict(X_industry)
    industry_mapping = {0: 'Industry In1', 1: 'Industry In2', 2: 'Industry In3'}
    predicted_industries = [industry_mapping.get(pred, 'Unknown') for pred in industry_predictions]
    
    # Return predictions
    return {
        'funding_predictions': funding_predictions.tolist(),
        'success_predictions': success_predictions.tolist(),
        'industry_predictions': predicted_industries
    }

# Run the app only if executed directly
if __name__ == "__main__":
    st.write("Streamlit app is running...")

# Load the selected page
if page == "Home":
    import Home
    Home.main()
elif page == "_Funding_Predictor":
    import Funding_Predictor
    Funding_Predictor.main()
elif page == "Startup Success":
    import Startup_Success
    Startup_Success.main()
elif page == "Industry In Classifier":
    import Industry_Classifier
    Industry_Classifier.main()

