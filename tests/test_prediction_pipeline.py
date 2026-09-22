import numpy as np
import pandas as pd

from startup_funding.prediction import find_similar_startups, predict_funding, predict_success, startup_input_frame
from startup_funding.training import train_funding_model, train_success_model
from startup_funding.model_io import load_bundle


def sample_training_data() -> pd.DataFrame:
    rows = []
    industries = ["FinTech", "HealthTech", "EdTech", "Ecommerce"]
    cities = ["Bengaluru", "Mumbai", "New Delhi", "Gurugram"]
    rounds = ["Seed", "Series A", "Series B"]
    for index in range(72):
        industry = industries[index % len(industries)]
        rows.append(
            {
                "CompanyName": f"Startup {index}",
                "Year Founded": 2015 + (index % 7),
                "Funding Year": 2019 + (index % 4),
                "Founded Imputed": False,
                "Head Quarter": cities[index % len(cities)],
                "Industry In": industry,
                "AboutCompany": f"{industry} platform for recurring revenue and analytics {index}",
                "Founders": "Founders",
                "Investor": "Investor",
                "Amount in ($)": 200000 + (index * 45000) + (industries.index(industry) * 100000),
                "Funding Round/Series": rounds[index % len(rounds)],
            }
        )
    return pd.DataFrame(rows)


def test_training_bundles_predict_unknown_categories(tmp_path):
    data = sample_training_data()
    funding_summary = train_funding_model(data, tmp_path)
    success_summary = train_success_model(data, tmp_path)

    startup = startup_input_frame(
        company_name="NewCo",
        year_founded=2022,
        funding_year=2026,
        head_quarter="Pune",
        industry="ClimateTech",
        about_company="Software for energy forecasting",
        funding_round="Pre-seed",
    )

    funding_bundle = load_bundle(tmp_path / "funding_pipeline.pkl")
    success_bundle = load_bundle(tmp_path / "success_pipeline.pkl")

    funding = predict_funding(funding_bundle, startup)
    readiness = predict_success(success_bundle, startup)

    assert np.isfinite(funding[0])
    assert 0 <= readiness[0] <= 1


def test_similar_startups_prioritizes_matching_market():
    data = sample_training_data()
    startup = startup_input_frame(
        company_name="NewCo",
        year_founded=2020,
        funding_year=2022,
        head_quarter="Bengaluru",
        industry="FinTech",
        about_company="Payments analytics",
        funding_round="Seed",
    )

    similar = find_similar_startups(data, startup, top_n=3)

    assert len(similar) == 3
    assert similar.iloc[0]["Industry In"] == "FinTech"
