"""Sepsis... (see P5) — this is the LP-1 Startup Funding Analyzer API."""

import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from startup_funding.config import BASE_DIR, DATA_DIR  # noqa: E402
from startup_funding.data import load_and_clean_data  # noqa: E402
from startup_funding.prediction import (  # noqa: E402
    classify_industry,
    find_similar_startups,
    load_model_bundles,
    predict_funding,
    predict_success,
    startup_input_frame,
)

API_KEY = os.getenv("API_KEY", "")

app = FastAPI(
    title="LP-1 Startup Funding Analyzer API",
    description="Funding estimate, success probability, industry classification and similar-market lookup.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@lru_cache(maxsize=1)
def bundles():
    return load_model_bundles()


@lru_cache(maxsize=1)
def history():
    return load_and_clean_data(DATA_DIR)


def check_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class StartupInput(BaseModel):
    company_name: str = Field(..., examples=["Zippin"])
    year_founded: int = Field(..., ge=1900, le=2100)
    head_quarter: str = Field(..., examples=["Bangalore"])
    industry: str = Field(..., examples=["FinTech"])
    about_company: str = Field(..., examples=["Autonomous checkout stores"])
    funding_round: str = Field(..., examples=["Seed"])
    funding_year: Optional[int] = None
    founders: str = "Unknown"
    investor: str = "Unknown"


@app.get("/health")
def health():
    loaded = sorted(bundles())
    return {
        "status": "ok",
        "models_loaded": loaded,
        "history_rows": len(history()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict/startup")
def predict_startup(item: StartupInput, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    b = bundles()
    missing = {"funding", "success", "industry"} - set(b)
    if missing:
        raise HTTPException(status_code=503, detail=f"Models not available: {missing}")
    frame = startup_input_frame(
        company_name=item.company_name,
        year_founded=item.year_founded,
        head_quarter=item.head_quarter,
        industry=item.industry,
        about_company=item.about_company,
        funding_round=item.funding_round,
        founders=item.founders,
        investor=item.investor,
        **({"funding_year": item.funding_year} if item.funding_year else {}),
    )
    similar = find_similar_startups(history(), frame, top_n=5)
    return {
        "predicted_funding_usd": round(float(predict_funding(b["funding"], frame)[0]), 2),
        "success_probability": round(float(predict_success(b["success"], frame)[0]), 4),
        "predicted_industry": str(classify_industry(b["industry"], [item.about_company])[0]),
        "similar_startups": similar.to_dict(orient="records"),
    }


class IndustryInput(BaseModel):
    descriptions: list[str] = Field(..., min_length=1, max_length=200)


@app.post("/classify/industry")
def classify(item: IndustryInput, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    b = bundles()
    if "industry" not in b:
        raise HTTPException(status_code=503, detail="industry model not available")
    preds = classify_industry(b["industry"], item.descriptions)
    return {"predictions": [str(p) for p in preds]}
