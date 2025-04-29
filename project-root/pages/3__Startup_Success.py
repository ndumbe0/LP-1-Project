import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Startup Success Prediction")

    if 'cleaned_data' in st.session_state:
        df = st.session_state['cleaned_data']
        st.write("Data from Home Page:")
        st.dataframe(df)

        
        model = joblib.load('success_model.joblib')

        
        X = df[['Industry In', 'Year Founded']]
        X = pd.get_dummies(X, columns=['Industry In'], drop_first=True)
        X = X.fillna(X.mean())

        
        predictions = model.predict(X)
        st.write("Predictions:")
        st.write(predictions)

        
        st.subheader("Startup Success Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(predictions, ax=ax)
        st.pyplot(fig)

    else:
        st.info("Please upload data on the Home page first.")

if __name__ == "__main__":
    main()