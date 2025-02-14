# pages/2_💰_Funding_Predictor.py
import streamlit as st
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load('models/funding_model.joblib')
imputer = joblib.load('models/funding_imputer.joblib')

def preprocess_data(df):
    # Implement your cleaning logic from notebook
    return processed_df

st.title("💰 Funding Prediction Model")
uploaded_file = st.file_uploader("Upload startup data", 
                               type=["csv", "xlsx"])

if uploaded_file:
    with st.spinner("Processing data..."):
        df = pd.read_csv(uploaded_file)  # or read_excel for xlsx
        processed_df = preprocess_data(df)
        X = imputer.transform(processed_df)
        predictions = model.predict(X)
    
    st.success("Predictions complete!")
    st.subheader("Results")
    st.dataframe(pd.DataFrame({
        "Predicted Funding": np.expm1(predictions)
    }), use_container_width=True)