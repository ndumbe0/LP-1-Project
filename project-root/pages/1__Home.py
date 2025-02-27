import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading pre-trained models

def clean_data(df):
    # Convert 'Founded' to numeric, handling errors by coercion
    df['Founded'] = pd.to_numeric(df['Founded'], errors='coerce')

    # Convert 'Amount' to numeric, handling missing values
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Drop rows where 'Amount' or 'Founded' is NaN
    df = df.dropna(subset=['Amount', 'Founded'])

    # Fill missing values in other columns with a placeholder (e.g., 'Unknown')
    df = df.fillna('Unknown')

    # Standardize column names (optional, but good practice)
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    df.rename(columns={'Company_Name': 'CompanyName', 'Head_Quarter': 'HeadQuarter', 'About_Company': 'AboutCompany'}, inplace=True)
   # Keep only necessary columns
    df = df[['CompanyName', 'Founded', 'HeadQuarter', 'Industry', 'AboutCompany', 'Founders', 'Investor', 'Amount', 'RoundSeries']]

    return df

def main():
    st.title("Startup Data Analyzer")

    uploaded_file = st.file_uploader("Upload your startup data (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Original Data:")
            st.dataframe(df)

            # Clean the data
            cleaned_df = clean_data(df.copy())  # Use a copy to avoid modifying the original DataFrame
            st.write("Cleaned Data:")
            st.dataframe(cleaned_df)

            # Store the cleaned data in Streamlit session state
            st.session_state['cleaned_data'] = cleaned_df

            st.success("Data uploaded and cleaned successfully! Navigate to other pages.")

        except Exception as e:
            st.error(f"Error processing the file: {e}")
    else:
        st.info("Upload a CSV file to begin.")

if __name__ == "__main__":
    main()
