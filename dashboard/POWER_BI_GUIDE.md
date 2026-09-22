# Power BI Dashboard — LP-1 Startup Funding Analyzer (2018–2021)

Build this in ~10 minutes with **Power BI Desktop** (free). Everything uses the
ready-made file `startup_funding_unified.csv` in this folder (1,630 funding
deals, 4 years, cleaned and normalized from your repo).

## 1. Load the data

1. Open Power BI Desktop → **Get data → Text/CSV**.
2. Pick `startup_funding_unified.csv` → **Transform Data** (not Load).
3. In Power Query set column types:
   - `funding_year`, `founded_year` → Whole Number
   - `amount_usd` → Decimal (or Whole) Number
   - everything else → Text
4. Rename headers for report labels: `company_name` → Company,
   `headquarters_city` → City, `funding_round` → Funding Round,
   `amount_usd` → Funding Amount. Double-click the query name in the Queries
   pane and rename it to `Funding` (the DAX below assumes that name).
   Close & Apply.

## 2. DAX measures (Modeling → New Measure)

```dax
Total Funding USD = SUM(Funding[Funding Amount])

Deal Count = COUNTROWS(Funding)

Avg Deal USD = AVERAGE(Funding[Funding Amount])

Median Deal USD = MEDIAN(Funding[Funding Amount])

// Previous-year total for the year currently in filter context
Funding PY =
VAR cur = MAX(Funding[funding_year])
RETURN CALCULATE([Total Funding USD],
       FILTER(ALL(Funding[funding_year]), Funding[funding_year] = cur - 1))

YoY Growth % = DIVIDE([Total Funding USD] - [Funding PY], [Funding PY])
```

## 3. Pages and visuals

### Page 1 — Funding Over Time (headline)
| Visual | Fields |
|---|---|
| **Clustered column chart** | Axis: `funding_year` • Values: `Total Funding USD` |
| **Line chart** | Axis: `funding_year` • Values: `Avg Deal USD` |
| **Card** | `Total Funding USD` |
| **Card** | `Deal Count` |
| **Multi-row card** | Fields: `funding_year` • Values: `Total Funding USD`, `YoY Growth %` |

### Page 2 — Industry Breakdown
| Visual | Fields |
|---|---|
| **Bar chart** (Top N filter: top 10 by Total Funding) | Axis: `industry` • Values: `Total Funding USD` |
| **Matrix** | Rows: `industry` • Columns: `funding_year` • Values: `Total Funding USD` + conditional formatting (color scale) → heatmap |

### Page 3 — Location & Rounds
| Visual | Fields |
|---|---|
| **Map** | Location: `City` • Size: `Total Funding USD` |
| **100% stacked bar** | Axis: `funding_year` • Legend: `Funding Round` • Values: `Deal Count` |
| **Slicer** | `funding_year` (add to all pages: Format → Sync slicers) |

Tip: `industry` values are comma-separated multi-tags (e.g.
"FinTech, Payments"). For a stricter split, in Power Query select `industry` →
**Split Column by delimiter** → Unpivot, then relate back.

## 4. Optional: use SQL Server instead of the CSV

Run `lp1_startup_funding.sql` in SSMS, then in Power BI:
**Get data → SQL Server** → server name, database `LP1_StartupFunding` →
select `startup_funding` plus the views (`v_yearly_funding_trend`,
`v_yoy_change`, `v_industry_funding`, `v_location_funding`, `v_round_mix`,
`v_top_investors`). The views do the same aggregations the DAX above does, so
you can bind visuals straight to them.

## 5. Save

File → Save As → `LP1_Funding_Dashboard.pbix`. Publish to powerbi.com if you
want to share it.
