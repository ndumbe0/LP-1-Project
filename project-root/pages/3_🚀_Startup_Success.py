# pages/3_🚀_Startup_Success.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load pre-trained pipeline
pipeline = joblib.load('models/success_pipeline.joblib')

def preprocess_success_data(df):
    """Replicate preprocessing from notebook"""
    # Handle missing values - use same strategy as training
    df = df.fillna({
        'Founded_scaled': df['Founded_scaled'].median(),
        'RoundSeries_scaled': df['RoundSeries_scaled'].median(),
        'Head_Quarter': 'Unknown',
        'Industry': 'Unknown'
    })
    return df[['Founded_scaled', 'RoundSeries_scaled', 'Head_Quarter', 'Industry']]

st.title("🚀 Startup Success Predictor")
st.markdown("Predict whether startups will secure > $1M funding")

uploaded_file = st.file_uploader("Upload startup data", 
                                type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        with st.spinner("Analyzing data..."):
            # Preprocess
            processed_data = preprocess_success_data(df)
            
            # Predict
            predictions = pipeline.predict(processed_data)
            probabilities = pipeline.predict_proba(processed_data)[:, 1]
            
            # Format results
            results_df = pd.DataFrame({
                'Success Probability': probabilities,
                'Predicted Success': ['Yes' if x == 1 else 'No' for x in predictions]
            })
            
        st.success("Analysis complete!")
        
        # Show interactive results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Predictions Overview")
            st.dataframe(results_df, use_container_width=True)
            
        with col2:
            st.subheader("Success Distribution")
            st.bar_chart(results_df['Predicted Success'].value_counts())
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")