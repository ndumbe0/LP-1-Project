"""Project paths and shared constants."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42
REFERENCE_YEAR = 2026

RAW_DATASETS = {
    "startup_funding2018.csv": 2018,
    "startup_funding2019.csv": 2019,
    "dbo.LP1_startup_funding2020.csv": 2020,
    "dbo.LP1_startup_funding2021.csv": 2021,
}

CANONICAL_COLUMNS = [
    "CompanyName",
    "Year Founded",
    "Funding Year",
    "Founded Imputed",
    "Head Quarter",
    "Industry In",
    "AboutCompany",
    "Founders",
    "Investor",
    "Amount in ($)",
    "Funding Round/Series",
]

CATEGORICAL_FEATURES = [
    "Industry In",
    "Head Quarter",
    "Funding Round/Series",
]

NUMERIC_FEATURES = [
    "Year Founded",
    "Funding Year",
    "Company Age",
    "Funding Lag",
    "Description Length",
]
