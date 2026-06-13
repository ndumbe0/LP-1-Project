"""
EDA and Data Cleaning for Indian Startup Funding Analysis.
Loads raw data, cleans, visualizes, and saves cleaned output.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 10})


def load_and_clean_data():
    """Load raw CSV files and merge into a single clean dataset."""
    path = lambda f: os.path.join(DATA_DIR, f)

    rows_2020 = pd.read_csv(path('dbo.LP1_startup_funding2020.csv'))
    rows_2021 = pd.read_csv(path('dbo.LP1_startup_funding2021.csv'))
    rows_2019 = pd.read_csv(path('startup_funding2019.csv'))
    rows_2018 = pd.read_csv(path('startup_funding2018.csv'))

    rows_2020['source_year'] = 2020
    rows_2021['source_year'] = 2021
    rows_2019['source_year'] = 2019
    rows_2018['source_year'] = 2018

    dfs_2020_2021 = [rows_2020, rows_2021]
    for df in dfs_2020_2021:
        df.rename(columns={
            'Company_Brand': 'CompanyName',
            'Founded': 'Year Founded',
            'HeadQuarter': 'Head Quarter',
            'Sector': 'Industry In',
            'What_it_does': 'AboutCompany',
            'Amount': 'Amount in ($)',
            'Stage': 'Funding Round/Series'
        }, inplace=True)
        if 'column10' in df.columns:
            df.drop(columns=['column10'], inplace=True)

    rows_2019.rename(columns={
        'Company/Brand': 'CompanyName',
        'Founded': 'Year Founded',
        'HeadQuarter': 'Head Quarter',
        'Sector': 'Industry In',
        'What it does': 'AboutCompany',
        'Amount($)': 'Amount in ($)',
    }, inplace=True)

    rows_2018.rename(columns={
        'Company Name': 'CompanyName',
        'Industry': 'Industry In',
        'Round/Series': 'Funding Round/Series',
        'Amount': 'Amount in ($)',
        'Location': 'Head Quarter',
        'About Company': 'AboutCompany',
    }, inplace=True)

    combined = pd.concat([rows_2020, rows_2021, rows_2019, rows_2018],
                         ignore_index=True, sort=False)

    combined['Amount in ($)'] = (
        combined['Amount in ($)']
        .astype(str)
        .str.replace(r'[\$,]', '', regex=True)
        .str.strip()
    )
    combined['Amount in ($)'] = pd.to_numeric(combined['Amount in ($)'], errors='coerce')

    combined['Year Founded'] = pd.to_numeric(combined['Year Founded'], errors='coerce')

    keep_cols = ['CompanyName', 'Year Founded', 'Head Quarter', 'Industry In',
                 'AboutCompany', 'Founders', 'Investor', 'Amount in ($)',
                 'Funding Round/Series']
    keep_cols = [c for c in keep_cols if c in combined.columns]
    clean = combined[keep_cols].copy()

    clean.dropna(subset=['Amount in ($)', 'Year Founded'], inplace=True)
    clean['Year Founded'] = clean['Year Founded'].astype(int)

    for col in ['CompanyName', 'Head Quarter', 'Industry In', 'Founders', 'Investor',
                'Funding Round/Series', 'AboutCompany']:
        if col in clean.columns:
            clean[col] = clean[col].fillna('Unknown')

    clean = clean[clean['Amount in ($)'] > 0].reset_index(drop=True)
    clean = clean.drop_duplicates().reset_index(drop=True)

    return clean


def perform_eda(df):
    """Generate EDA visualizations and summary statistics."""
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nNumeric summary:\n{df.describe()}")

    top_industries = df.groupby('Industry In')['Amount in ($)'].mean().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_industries.values, y=top_industries.index, palette='viridis', ax=ax)
    ax.set_title('Top 15 Industries by Average Funding')
    ax.set_xlabel('Average Funding Amount ($)')
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'top_industries_funding.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    top_locations = df.groupby('Head Quarter')['Amount in ($)'].sum().sort_values(ascending=False).head(15)
    sns.barplot(x=top_locations.values, y=top_locations.index, palette='plasma', ax=ax)
    ax.set_title('Top 15 Locations by Total Funding')
    ax.set_xlabel('Total Funding Amount ($)')
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'top_locations_funding.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    round_funding = df.groupby('Funding Round/Series')['Amount in ($)'].median().sort_values(ascending=False)
    sns.barplot(x=round_funding.values, y=round_funding.index, palette='magma', ax=ax)
    ax.set_title('Median Funding Amount by Round/Series')
    ax.set_xlabel('Median Funding Amount ($)')
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'funding_by_round.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    yearly = df.groupby('Year Founded')['Amount in ($)'].sum()
    sns.lineplot(x=yearly.index, y=yearly.values, marker='o', ax=ax)
    ax.set_title('Total Funding Amount by Year')
    ax.set_xlabel('Year Founded')
    ax.set_ylabel('Total Funding ($)')
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'funding_by_year.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    df['Log Amount'] = np.log1p(df['Amount in ($)'])
    corr_df = df[['Year Founded', 'Log Amount']].copy()
    corr_df['Head Quarter Enc'] = pd.factorize(df['Head Quarter'])[0]
    corr_matrix = corr_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\n--- EDA Complete. Charts saved to images/ ---")


def save_clean_data(df):
    """Save cleaned dataset to CSV."""
    out_path = os.path.join(DATA_DIR, 'startup_funding_clean.csv')
    df.to_csv(out_path, index=False)
    print(f"Cleaned data saved to: {out_path}")
    return out_path


if __name__ == '__main__':
    data = load_and_clean_data()
    perform_eda(data)
    save_clean_data(data)
