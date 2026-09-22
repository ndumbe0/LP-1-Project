import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "startup_funding_unified.csv")


def q(v, is_num=False):
    if pd.isna(v) or v is None:
        return "NULL"
    if is_num:
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)
    return "N'" + str(v).replace("'", "''") + "'"


cols = ["funding_year", "company_name", "founded_year", "headquarters_city",
        "headquarters_state", "industry", "about_company", "founders",
        "investor", "amount_usd", "funding_round"]

lines = []
A = lines.append
A("-- ============================================================")
A("-- LP-1 Startup Funding Analyzer - SQL Server database script")
A("-- Generated from repo/data cleaned CSVs (2018-2021)")
A("-- Rows: %d | Run in SSMS / sqlcmd" % len(df))
A("-- ============================================================")
A("IF DB_ID('LP1_StartupFunding') IS NULL")
A("    CREATE DATABASE LP1_StartupFunding;")
A("GO")
A("USE LP1_StartupFunding;")
A("GO")
A("")
A("IF OBJECT_ID('dbo.startup_funding', 'U') IS NOT NULL DROP TABLE dbo.startup_funding;")
A("CREATE TABLE dbo.startup_funding")
A("(")
A("    funding_id          INT IDENTITY(1,1) NOT NULL PRIMARY KEY,")
A("    funding_year        SMALLINT          NOT NULL,")
A("    company_name        NVARCHAR(200)     NULL,")
A("    founded_year        SMALLINT          NULL,")
A("    headquarters_city   NVARCHAR(100)     NULL,")
A("    headquarters_state  NVARCHAR(100)     NULL,")
A("    industry            NVARCHAR(300)     NULL,")
A("    about_company       NVARCHAR(1000)    NULL,")
A("    founders            NVARCHAR(500)     NULL,")
A("    investor            NVARCHAR(500)     NULL,")
A("    amount_usd          BIGINT            NULL,")
A("    funding_round       NVARCHAR(100)     NULL,")
A("    loaded_at           DATETIME2         NOT NULL CONSTRAINT DF_startup_funding_loaded DEFAULT SYSUTCDATETIME()")
A(");")
A("GO")
A("")

BATCH = 200
for start in range(0, len(df), BATCH):
    chunk = df.iloc[start:start + BATCH]
    A("INSERT INTO dbo.startup_funding")
    A("    (funding_year, company_name, founded_year, headquarters_city, headquarters_state,")
    A("     industry, about_company, founders, investor, amount_usd, funding_round)")
    A("VALUES")
    vals = []
    for _, r in chunk.iterrows():
        vals.append("    (" + ", ".join([
            q(r["funding_year"], True), q(r["company_name"]), q(r["founded_year"], True),
            q(r["headquarters_city"]), q(r["headquarters_state"]), q(r["industry"]),
            q(r["about_company"]), q(r["founders"]), q(r["investor"]),
            q(r["amount_usd"], True), q(r["funding_round"])]) + ")")
    A(",\n".join(vals) + ";")
    A("GO")
    A("")

A("-- Indexes for common analytical queries")
A("CREATE INDEX IX_sf_year    ON dbo.startup_funding (funding_year) INCLUDE (amount_usd);")
A("CREATE INDEX IX_sf_city    ON dbo.startup_funding (headquarters_city);")
A("CREATE INDEX IX_sf_industry ON dbo.startup_funding (industry);")
A("GO")
A("")
A("-- ============ ANALYTICAL VIEWS ============")
A("")
A("-- Headline yearly funding trend (total, deal count, average, median)")
A("CREATE OR ALTER VIEW dbo.v_yearly_funding_trend AS")
A("SELECT funding_year,")
A("       COUNT(*)                              AS deal_count,")
A("       SUM(amount_usd)                       AS total_funding_usd,")
A("       AVG(CAST(amount_usd AS FLOAT))        AS avg_deal_usd,")
A("       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount_usd)")
A("                                 AS median_deal_usd")
A("FROM dbo.startup_funding")
A("GROUP BY funding_year;")
A("GO")
A("")
A("-- Year-over-year change on total funding")
A("CREATE OR ALTER VIEW dbo.v_yoy_change AS")
A("SELECT funding_year,")
A("       total_funding_usd,")
A("       LAG(total_funding_usd) OVER (ORDER BY funding_year) AS prev_year_usd,")
A("       total_funding_usd - LAG(total_funding_usd) OVER (ORDER BY funding_year) AS yoy_diff_usd,")
A("       ROUND(100.0 * (total_funding_usd - LAG(total_funding_usd) OVER (ORDER BY funding_year))")
A("             / NULLIF(LAG(total_funding_usd) OVER (ORDER BY funding_year), 0), 1) AS yoy_pct")
A("FROM dbo.v_yearly_funding_trend;")
A("GO")
A("")
A("-- Funding by industry and year")
A("CREATE OR ALTER VIEW dbo.v_industry_funding AS")
A("SELECT industry, funding_year,")
A("       COUNT(*)                       AS deal_count,")
A("       SUM(amount_usd)                AS total_funding_usd,")
A("       AVG(CAST(amount_usd AS FLOAT)) AS avg_deal_usd")
A("FROM dbo.startup_funding")
A("WHERE industry IS NOT NULL")
A("GROUP BY industry, funding_year;")
A("GO")
A("")
A("-- Funding by headquarters city and year")
A("CREATE OR ALTER VIEW dbo.v_location_funding AS")
A("SELECT headquarters_city, headquarters_state, funding_year,")
A("       COUNT(*)        AS deal_count,")
A("       SUM(amount_usd) AS total_funding_usd")
A("FROM dbo.startup_funding")
A("WHERE headquarters_city IS NOT NULL")
A("GROUP BY headquarters_city, headquarters_state, funding_year;")
A("GO")
A("")
A("-- Funding round mix per year")
A("CREATE OR ALTER VIEW dbo.v_round_mix AS")
A("SELECT funding_round, funding_year,")
A("       COUNT(*)        AS deal_count,")
A("       SUM(amount_usd) AS total_funding_usd")
A("FROM dbo.startup_funding")
A("WHERE funding_round IS NOT NULL")
A("GROUP BY funding_round, funding_year;")
A("GO")
A("")
A("-- Top investors by total deployed capital")
A("CREATE OR ALTER VIEW dbo.v_top_investors AS")
A("SELECT TOP 50 investor,")
A("       COUNT(*)        AS deal_count,")
A("       SUM(amount_usd) AS total_funding_usd")
A("FROM dbo.startup_funding")
A("WHERE investor IS NOT NULL")
A("GROUP BY investor")
A("ORDER BY total_funding_usd DESC;")
A("GO")
A("")
A("-- Verification query")
A("SELECT funding_year, COUNT(*) AS rows_loaded, SUM(amount_usd) AS total_usd")
A("FROM dbo.startup_funding GROUP BY funding_year ORDER BY funding_year;")
A("GO")

out = HERE / "lp1_startup_funding.sql"
out.write_text("\n".join(lines), encoding="utf-8")
print("wrote", out, out.stat().st_size, "bytes,", len(df), "insert rows")
