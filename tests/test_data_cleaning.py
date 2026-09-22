import math

import pandas as pd

from startup_funding.data import clean_startup_dataframe, parse_amount_to_usd, sanitize_csv_cell


def test_raw_aliases_are_cleaned_to_canonical_columns():
    raw = pd.DataFrame(
        [
            {
                "Company_Name": "Krayonnz",
                "Founded": 2019,
                "HeadQuarter": "Bangalore, Karnataka, India",
                "Industry": "EdTech",
                "About_Company": "Learning platform",
                "Amount": "$100,000",
                "Round/Series": "Seed",
            }
        ]
    )

    clean = clean_startup_dataframe(raw, source_year=2021)

    assert clean.loc[0, "CompanyName"] == "Krayonnz"
    assert clean.loc[0, "Head Quarter"] == "Bengaluru"
    assert clean.loc[0, "Industry In"] == "EdTech"
    assert clean.loc[0, "Amount in ($)"] == 100000.0
    assert clean.loc[0, "Funding Round/Series"] == "Seed"


def test_missing_founded_year_uses_source_year_flag():
    raw = pd.DataFrame(
        [
            {
                "Company Name": "TheCollegeFever",
                "Industry": "Marketing",
                "Amount": "$250,000",
                "Location": "Bangalore",
                "About Company": "College events",
            }
        ]
    )

    clean = clean_startup_dataframe(raw, source_year=2018)

    assert clean.loc[0, "Year Founded"] == 2018
    assert clean.loc[0, "Founded Imputed"]


def test_amount_parser_handles_common_units_and_undisclosed_values():
    assert parse_amount_to_usd("$1,250,000") == 1250000.0
    assert parse_amount_to_usd("2.5 million") == 2500000.0
    assert math.isnan(parse_amount_to_usd("Undisclosed"))


def test_csv_formula_values_are_neutralized():
    assert sanitize_csv_cell("=cmd|A1").startswith("'=")
