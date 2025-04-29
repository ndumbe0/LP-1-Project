import streamlit as st
import pandas as pd
import numpy as np
import joblib  

def clean_data(df):
    # Convert 'Year Founded' to numeric, handling errors by coercion
    df['Year Founded'] = pd.to_numeric(df['Year Founded'], errors='coerce')

   
    df['Amount in ($)'] = pd.to_numeric(df['Amount in ($)'], errors='coerce')

   
    df = df.dropna(subset=['Amount in ($)', 'Year Founded'])

    
    df = df.fillna('Unknown')

    # Standardize column names (optional, but good practice)
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    df.rename(columns={'Company_Name': 'CompanyName', 'Head_Quarter': 'Head Quarter', 'About_Company': 'AboutCompany'}, inplace=True)
   # Keep only necessary columns
    df = df[['CompanyName', 'Year Founded', 'Head Quarter', 'Industry In', 'AboutCompany', 'Founders', 'Investor', 'Amount in ($)', 'Funding Round/Series']]

    return df

def main():
    st.title("Startup Data Analyzer")

    uploaded_file = st.file_uploader("Upload your startup data (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Original Data:")
            st.dataframe(df)

            
            cleaned_df = clean_data(df.copy())  
            st.write("Cleaned Data:")
            st.dataframe(cleaned_df)

            
            st.session_state['cleaned_data'] = cleaned_df

            st.success("Data uploaded and cleaned successfully! Navigate to other pages.")

        except Exception as e:
            st.error(f"Error processing the file: {e}")
    else:
        st.info("Upload a CSV file to begin.")

if __name__ == "__main__":
    main()
