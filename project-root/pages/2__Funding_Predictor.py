import streamlit as st
import pandas as pd
import joblib
<<<<<<< HEAD
import plotly.express as px
import plotly.graph_objects as go

def main():
    st.title("Funding Prediction 💰")
=======
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Funding Prediction")
>>>>>>> a652281e95dbfb0b149a8ac1a2269bc80ef39378

    # Check if cleaned data exists in session state
    if 'cleaned_data' in st.session_state:
        df = st.session_state['cleaned_data']
        st.write("Data from Home Page:")
        st.dataframe(df)

        # Load the model
<<<<<<< HEAD
        model = joblib.load('models/funding_model.joblib')

        # Preprocess the data for prediction (example)
=======
        model = joblib.load('funding_model.joblib')

        # Preprocess the data for prediction (example)
        # This needs to be adjusted based on your model's requirements
>>>>>>> a652281e95dbfb0b149a8ac1a2269bc80ef39378
        X = df[['Founded']]  # Example: Using 'Founded' year as a feature
        X = X.fillna(X.mean())  # Handle any missing values

        # Make predictions
        predictions = model.predict(X)
        st.write("Predictions:")
        st.write(predictions)

        # Create visualizations (example)
        st.subheader("Funding Amount Distribution")
<<<<<<< HEAD
        fig = px.histogram(df, x='Amount', nbins=30, title="Funding Amount Distribution")
        st.plotly_chart(fig)

        st.subheader("Founded Year vs. Predicted Funding")
        fig = px.scatter(df, x='Founded', y=predictions, title="Founded Year vs. Predicted Funding")
        st.plotly_chart(fig)
=======
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['Amount'], ax=ax)
        st.pyplot(fig)

        st.subheader("Founded Year vs. Predicted Funding")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(x=df['Founded'], y=predictions, ax=ax)
        st.pyplot(fig)
>>>>>>> a652281e95dbfb0b149a8ac1a2269bc80ef39378

    else:
        st.info("Please upload data on the Home page first.")

if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> a652281e95dbfb0b149a8ac1a2269bc80ef39378
