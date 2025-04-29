import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    st.title("Funding Prediction 💰")

    
    if 'cleaned_data' in st.session_state:
        df = st.session_state['cleaned_data']
        st.write("Data from Home Page:")
        st.dataframe(df)

        
        try:
            model = joblib.load('models/funding_model.joblib')  
        except FileNotFoundError:
            st.error("Model file not found. Please check the path.")
            return

        
        X = df[['Year Founded']]  
        X = X.fillna(X.mean())  

        
        predictions = model.predict(X)
        st.write("Predictions:")
        st.write(predictions)

        
        st.subheader("Funding Amount in ($) Distribution")
        fig = px.histogram(df, x='Amount in ($)', nbins=30, title="Funding Amount in ($) Distribution")
        st.plotly_chart(fig)

        
        st.subheader("Year Founded vs. Predicted Funding")
        fig = px.scatter(df, x='Year Founded', y=predictions, title="Year Founded vs. Predicted Funding")
        st.plotly_chart(fig)

       
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