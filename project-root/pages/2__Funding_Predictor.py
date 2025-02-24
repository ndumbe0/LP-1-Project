import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent
models_dir = current_dir.parent / "models"  

model = joblib.load(models_dir / "funding_model.joblib")
imputer = joblib.load(models_dir / "funding_imputer.joblib")

def preprocess_data(df):
    st.write("Uploaded file columns:", df.columns.tolist())
    df.columns = df.columns.str.strip().str.lower()
    expected_columns = ['Founded', 'RoundSeries', 'Head_Quarter', 'Industry']
    
    if not all(col in df.columns for col in expected_columns):
        st.error(f"Uploaded data must contain the following columns: {expected_columns}")
        return None
    
    processed_df = df[expected_columns]
    return processed_df

st.title("💰 Funding Prediction Model")
uploaded_file = st.file_uploader("Upload startup data", type=["csv", "xlsx"])

if uploaded_file:
    with st.spinner("Processing data..."):
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()  # Use st.stop() instead of return
        
        processed_df = preprocess_data(df)
        
        if processed_df is not None:
            X = imputer.transform(processed_df)
            predictions = model.predict(X)
            
            st.success("Predictions complete!")
            st.subheader("Results")
            st.dataframe(pd.DataFrame({
                "Predicted Funding": np.expm1(predictions)
            }), use_container_width=True)