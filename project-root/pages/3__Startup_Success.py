import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Startup Success Prediction")

    # Check if cleaned data exists in session state
    if 'cleaned_data' in st.session_state:
        df = st.session_state['cleaned_data']
        st.write("Data from Home Page:")
        st.dataframe(df)

        # Load the model
        model = joblib.load('success_model.joblib')  # Assuming your model is named 'success_model.joblib'

        # Preprocess the data for prediction (Adapt this to your model)
        # Example:  Using 'Industry' and 'Founded'
        X = df[['Industry', 'Founded']]
        X = pd.get_dummies(X, columns=['Industry'], drop_first=True) # One-hot encode 'Industry'
        X = X.fillna(X.mean())

        # Make predictions
        predictions = model.predict(X)
        st.write("Predictions:")
        st.write(predictions)

        # Create visualizations (example)
        st.subheader("Startup Success Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(predictions, ax=ax)
        st.pyplot(fig)

        # Example: Success Rate by Industry (if applicable)
        # (Requires more complex processing based on your model's output)
        # ...

    else:
        st.info("Please upload data on the Home page first.")

if __name__ == "__main__":
    main()
