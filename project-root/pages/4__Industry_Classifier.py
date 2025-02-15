# pages/4_🏭_Industry_Classifier.py
import streamlit as st
import joblib
import pandas as pd
import spacy
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Load pre-trained pipeline and components
pipeline = joblib.load('models/industry_pipeline.joblib')
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Replicate notebook's text preprocessing"""
    doc = nlp(str(text))
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

st.title("🏭 Industry Classifier")
st.markdown("Classify startups into industries based on company descriptions")

input_type = st.radio("Choose input type:", 
                     ["📁 Upload File", "📝 Direct Input"])

if input_type == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload company data (CSV/Excel)", 
                                    type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            if 'About_Company' not in df.columns:
                st.error("File must contain 'About_Company' column")
            else:
                with st.spinner("Classifying industries..."):
                    # Preprocess text
                    df['Processed_Text'] = df['About_Company'].apply(preprocess_text)
                    
                    # Predict
                    predictions = pipeline.predict(df['Processed_Text'])
                    
                    # Add to dataframe
                    df['Predicted_Industry'] = predictions
                    
                st.success("Classification complete!")
                st.subheader("Results")
                st.dataframe(df[['About_Company', 'Predicted_Industry']], 
                            use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

else:
    text_input = st.text_area("Enter company description:", 
                            height=150,
                            placeholder="Paste company description here...")
    
    if text_input:
        with st.spinner("Analyzing text..."):
            processed_text = preprocess_text(text_input)
            prediction = pipeline.predict([processed_text])[0]
            probability = np.max(pipeline.decision_function([processed_text]))
            
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Industry", prediction)
        with col2:
            st.metric("Confidence Score", f"{probability:.2f}")