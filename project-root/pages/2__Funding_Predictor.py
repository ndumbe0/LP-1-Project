import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Title of the page
    st.title("Funding Prediction 💰")

    # Check if cleaned_data exists in session state
    if 'cleaned_data' in st.session_state:
        df = st.session_state['cleaned_data']
        st.write("Data from Home Page:")
        st.dataframe(df)

        # Load the pre-trained model
        try:
            model = joblib.load('models/funding_model.joblib')  # Ensure this path is correct
        except FileNotFoundError:
            st.error("Model file not found. Please check the path.")
            return

        # Prepare input data for predictions
        X = df[['Year Founded']]  
        X = X.fillna(X.mean())  

        # Make predictions
        predictions = model.predict(X)
        st.write("Predictions:")
        st.write(predictions)

        # Plot Funding Amount Distribution
        st.subheader("Funding Amount in ($) Distribution")
        fig = px.histogram(df, x='Amount in ($)', nbins=30, title="Funding Amount in ($) Distribution")
        st.plotly_chart(fig)

        # Plot Year Founded vs Predicted Funding
        st.subheader("Year Founded vs. Predicted Funding")
        fig = px.scatter(df, x='Year Founded', y=predictions, title="Year Founded vs. Predicted Funding")
        st.plotly_chart(fig)

        # Plot using Matplotlib and Seaborn
        st.subheader("Funding Amount in ($) Distribution (Matplotlib)")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['Amount in ($)'], ax=ax)
        st.pyplot(fig)

        st.subheader("Year Founded vs. Predicted Funding (Matplotlib)")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(x=df['Year Founded'], y=predictions, ax=ax)
        st.pyplot(fig)

    else:
        st.info("Please upload data on the Home page first.")

if __name__ == "__main__":
    main()