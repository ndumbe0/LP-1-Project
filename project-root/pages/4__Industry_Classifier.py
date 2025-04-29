import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Industry Classification")

    
    if 'cleaned_data' in st.session_state:
        df = st.session_state['cleaned_data']
        st.write("Data from Home Page:")
        st.dataframe(df)

        
        model = joblib.load('industry_model.joblib')

        #
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()  
        X = vectorizer.fit_transform(df['AboutCompany'])  

       
        predictions = model.predict(X)
        st.write("Predictions:")
        st.write(predictions)

        
        industry_mapping = {0: 'Industry In1', 1: 'Industry In2', 2: 'Industry In3'} 
        predicted_industries = [industry_mapping.get(pred, 'Unknown') for pred in predictions] 
        st.write("Predicted Industries:")
        st.write(predicted_industries)

        
        st.subheader("Industry Distribution")
        industry_counts = pd.Series(predicted_industries).value_counts() 
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=industry_counts.index, y=industry_counts.values, ax=ax) 
        plt.xticks(rotation=45, ha='right') 
        st.pyplot(fig) 
    else:
        st.info("Please upload data on the Home page first.")

if __name__ == "__main__":
    main()
