import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Industry Classification")

    # Check if cleaned data exists in session state
    if 'cleaned_data' in st.session_state:
        df = st.session_state['cleaned_data']
        st.write("Data from Home Page:")
        st.dataframe(df)

        # Load the model
        model = joblib.load('industry_model.joblib') # Assuming your model is named 'industry_model.joblib'

        # Preprocess the data (Adapt to your model's input)
        # Example: Using 'AboutCompany' (text data)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()  # You might need to load a pre-fitted vectorizer
        X = vectorizer.fit_transform(df['AboutCompany'])  # Transforms the 'AboutCompany' descriptions into a matrix of TF-IDF features.

        # Make predictions
        predictions = model.predict(X)
        st.write("Predictions:")
        st.write(predictions)

        # Map the numerical predictions back to Industry names
        # Assuming you have a mapping of numbers to industry names
        industry_mapping = {0: 'Industry1', 1: 'Industry2', 2: 'Industry3'} # Replace with your actual mapping
        predicted_industries = [industry_mapping.get(pred, 'Unknown') for pred in predictions] # Returns the industry name
        st.write("Predicted Industries:")
        st.write(predicted_industries)

        # Create visualizations (example)
        st.subheader("Industry Distribution")
        industry_counts = pd.Series(predicted_industries).value_counts() # Counts how many times each value appears in the Series
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=industry_counts.index, y=industry_counts.values, ax=ax) # Creates a barplot
        plt.xticks(rotation=45, ha='right') # Rotates the labels on x axis
        st.pyplot(fig) # Displays graph in Streamlit
    else:
        st.info("Please upload data on the Home page first.")

if __name__ == "__main__":
    main()
