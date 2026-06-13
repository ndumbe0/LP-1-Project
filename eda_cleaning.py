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

plt.rcParams.update({
    'figure.max_open_warning': 0,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

import matplotlib.ticker as mticker


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


def fmt_billions(x, pos):
    if x >= 1e9:
        return f'${x/1e9:.1f}B'
    elif x >= 1e6:
        return f'${x/1e6:.0f}M'
    elif x >= 1e3:
        return f'${x/1e3:.0f}K'
    return f'${x:.0f}'


def generate_cover_image():
    """Generate the README banner image."""
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')

    ax.text(0.5, 3.8, 'INDIAN STARTUP FUNDING ANALYSIS',
            fontsize=32, fontweight='bold', color='white',
            ha='center', va='center')
    ax.text(0.5, 3.0, '1982 — 2021  \u2022  Machine Learning  \u2022  Data Analytics',
            fontsize=15, color='#58A6FF', ha='center', va='center')
    ax.text(0.5, 2.3, '1,060+ Startups  |  34 Industries  |  66 Locations  |  3 ML Models',
            fontsize=12, color='#8B949E', ha='center', va='center')
    ax.text(0.5, 1.6, 'A comprehensive analysis of funding trends, predictive modeling, and insights\n'
                      'into the Indian startup ecosystem.',
            fontsize=10, color='#6E7681', ha='center', va='center', style='italic')
    ax.text(0.5, 0.4, 'Azubi Africa \u2014 Data Science Cohort 7',
            fontsize=9, color='#484F58', ha='center', va='center')

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.savefig(os.path.join(IMAGES_DIR, 'cover.png'), dpi=150,
                bbox_inches='tight', pad_inches=0.2, facecolor='#0D1117')
    plt.close(fig)
    print('  [OK] cover.png')


def perform_eda(df):
    """Generate EDA visualizations and summary statistics."""
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nNumeric summary:\n{df.describe()}")

    # 1. Funding Trend by Year
    fig, ax = plt.subplots(figsize=(12, 5))
    yearly = df.groupby('Year Founded')['Amount in ($)'].sum() / 1e9
    ax.fill_between(yearly.index, yearly.values, alpha=0.2, color='#636EFA')
    ax.plot(yearly.index, yearly.values, marker='o', linewidth=2.5, color='#636EFA', markersize=5)
    ax.axvspan(2020, 2020.5, alpha=0.12, color='red', label='COVID-19 Pandemic')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:.1f}B'))
    ax.set_title('Total Funding Amount by Year', fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Total Funding ($)', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'funding_trend.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Top Locations
    fig, ax = plt.subplots(figsize=(12, 6))
    top_locs = df.groupby('Head Quarter')['Amount in ($)'].sum().sort_values(ascending=True).tail(12)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top_locs)))
    ax.barh(range(len(top_locs)), top_locs.values / 1e9, color=colors, edgecolor='white', linewidth=0.5)
    for i, (val, label) in enumerate(zip(top_locs.values, top_locs.index)):
        ax.text(val / 1e9 + 0.02, i, f'${val/1e9:.2f}B', va='center', fontsize=9, color='#333')
    ax.set_yticks(range(len(top_locs)))
    ax.set_yticklabels(top_locs.index, fontsize=10)
    ax.set_xlabel('Total Funding ($ Billions)', fontsize=11)
    ax.set_title('Top Locations by Total Funding', fontsize=15, fontweight='bold', pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:.1f}B'))
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'top_locations_funding.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. Funding Distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    amounts = df['Amount in ($)'] / 1e6
    ax.hist(amounts, bins=40, color='#636EFA', edgecolor='white', alpha=0.8)
    ax.axvline(amounts.median(), color='#EF553B', linestyle='--', linewidth=2,
               label=f'Median: ${amounts.median():.1f}M')
    ax.axvline(amounts.mean(), color='#00CC96', linestyle='--', linewidth=2,
               label=f'Mean: ${amounts.mean():.1f}M')
    ax.set_xlabel('Funding Amount ($ Millions)', fontsize=11)
    ax.set_ylabel('Number of Startups', fontsize=11)
    ax.set_title('Distribution of Funding Amounts', fontsize=15, fontweight='bold', pad=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'funding_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 4. Pandemic Impact
    pre = df[(df['Year Founded'] >= 2018) & (df['Year Founded'] <= 2019)]['Amount in ($)']
    during = df[(df['Year Founded'] >= 2020) & (df['Year Founded'] <= 2021)]['Amount in ($)']
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ['Pre-Pandemic\n(2018–2019)', 'During Pandemic\n(2020–2021)']
    data = [pre.values / 1e6, during.values / 1e6]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], ['#636EFA', '#EF553B']):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for i, d in enumerate(data):
        ax.annotate(f'Mean: ${np.mean(d):.1f}M', xy=(i + 1, np.mean(d)),
                    xytext=(i + 1.2, np.mean(d) + 3), fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.set_title('Pandemic Impact on Startup Funding', fontsize=15, fontweight='bold', pad=12)
    ax.set_ylabel('Funding Amount ($ Millions)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'pandemic_impact.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 5. Startups Per Year
    counts = df['Year Founded'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(counts.index, 0, counts.values, alpha=0.3, color='#00CC96')
    ax.plot(counts.index, counts.values, marker='s', linewidth=2.5, color='#00CC96', markersize=5)
    ax.set_title('Number of Startups Founded per Year', fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Number of Startups', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'startups_per_year.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 6. Industry Distribution Pie
    known = df[df['Industry In'] != 'Unknown']
    counts = known['Industry In'].value_counts().head(10)
    if len(counts) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=None, autopct='%1.1f%%',
            startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(counts))),
            wedgeprops=dict(edgecolor='white', linewidth=1.5))
        for t in autotexts:
            t.set_fontsize(8)
        ax.legend(wedges, [f'{l} ({v})' for l, v in zip(counts.index, counts.values)],
                  title='Industry', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.set_title('Industry Distribution of Startups', fontsize=15, fontweight='bold', pad=12)
        fig.tight_layout()
        fig.savefig(os.path.join(IMAGES_DIR, 'industry_pie.png'), dpi=150, bbox_inches='tight')
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
    generate_cover_image()
    perform_eda(data)
    save_clean_data(data)
