import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent.parent / "repo" / "data"
OUT = Path(__file__).parent / "startup_funding_unified.csv"


def split_part(series, i):
    def f(x):
        if pd.isna(x):
            return None
        parts = [p.strip() for p in str(x).split(",")]
        return parts[i] if len(parts) > i else None
    return series.map(f)


def clean(v):
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    return None if s in ("", "nan", "None", "Unknown", "undisclosed", "Undisclosed", "?", "\ufffd", "\x9d", "□") else s


PATHS = {
    2018: ("cleaned_data_startup_funding2018.csv", "latin-1"),
    2019: ("cleaned_data_startup_funding2019.csv", "utf-8"),
    2020: ("dbo.LP1_startup_funding2020.csv", "utf-8"),
    2021: ("dbo.LP1_startup_funding2021.csv", "utf-8"),
}

rows = []
for yr, (fname, enc) in PATHS.items():
    df = pd.read_csv(DATA / fname, encoding=enc).rename(columns=lambda c: c.strip())
    n = len(df)

    if yr == 2021:
        shifted = df["Round/Series"].astype(str).str.match(r"^\$?[\d,]+$")
        for i in df[shifted].index:
            df.loc[i, "Founders"] = df.loc[i, "Investor"]
            df.loc[i, "Investor"] = df.loc[i, "Amount"]
            df.loc[i, "Amount"] = str(df.loc[i, "Round/Series"]).replace("$", "").replace(",", "")
            df.loc[i, "Round/Series"] = None

    def col(name):
        return df[name] if name in df else pd.Series([None] * n)

    founded = pd.to_numeric(col("Founded"), errors="coerce")
    amount = pd.to_numeric(
        col("Amount").astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip(),
        errors="coerce")
    city_s = split_part(col("HeadQuarter"), 0)
    state_s = split_part(col("HeadQuarter"), 1)
    industry_col = col("Industry")
    IND_FIX = {"Fintech": "FinTech", "Edtech": "EdTech"}
    CITY_FIX = {"Bengaluru": "Bangalore", "Gurgaon": "Gurugram"}

    for j in range(n):
        def g(s):
            return clean(s[j])
        city = g(city_s)
        city = CITY_FIX.get(city, city) if city else city
        industry = g(industry_col)
        industry = IND_FIX.get(industry, industry) if industry else industry
        f = founded[j]
        a = amount[j]
        rows.append({
            "funding_year": yr,
            "company_name": g(col("Company_Name")),
            "founded_year": int(f) if pd.notna(f) else None,
            "headquarters_city": city,
            "headquarters_state": g(state_s),
            "industry": industry,
            "about_company": g(col("About_Company")),
            "founders": g(col("Founders")),
            "investor": g(col("Investor")),
            "amount_usd": float(a) if pd.notna(a) else None,
            "funding_round": g(col("Round/Series")),
        })

u = pd.DataFrame(rows).drop_duplicates(
    subset=["funding_year", "company_name", "amount_usd", "funding_round", "investor"])
u = u.sort_values(["funding_year", "company_name"]).reset_index(drop=True)
u.to_csv(OUT, index=False)

chk = pd.read_csv(OUT)
print("rows:", len(chk), "| null year:", chk["funding_year"].isna().sum())
print(chk.groupby("funding_year")["amount_usd"].agg(["count", "sum"]).to_string())
print("null amounts:", chk["amount_usd"].isna().sum())
