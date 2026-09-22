"""Streamlit app for startup funding analysis and prediction."""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from startup_funding.config import BASE_DIR, DATA_DIR, IMAGES_DIR, REFERENCE_YEAR
from startup_funding.data import clean_startup_dataframe, dataset_profile, read_clean_data, sanitize_csv_cell
from startup_funding.model_io import load_bundle
from startup_funding.prediction import (
    classify_industry,
    find_similar_startups,
    predict_funding,
    predict_success,
    startup_input_frame,
)


load_dotenv()

st.set_page_config(
    page_title="Startup Funding Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODEL_PATHS = {
    "funding": BASE_DIR / "models" / "funding_pipeline.pkl",
    "success": BASE_DIR / "models" / "success_pipeline.pkl",
    "industry": BASE_DIR / "models" / "industry_pipeline.pkl",
}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return read_clean_data(DATA_DIR / "startup_funding_clean.csv")


@st.cache_data(show_spinner=False)
def load_training_results() -> dict:
    path = BASE_DIR / "training_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    models = {}
    for name, path in MODEL_PATHS.items():
        if path.exists():
            models[name] = load_bundle(path)
    return models


def currency(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def readiness_label(probability: float) -> str:
    if probability >= 0.7:
        return "Strong"
    if probability >= 0.45:
        return "Promising"
    return "Early"


def metric_row(df: pd.DataFrame) -> None:
    profile = dataset_profile(df)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Startups", f"{profile['records']:,}")
    col2.metric("Total funding", currency(profile["total_funding_usd"]))
    col3.metric("Median round", currency(profile["median_funding_usd"]))
    col4.metric("Industries", f"{profile['industries']:,}")
    col5.metric("Locations", f"{profile['locations']:,}")


def render_model_cards(results: dict) -> None:
    st.subheader("Model scorecards")
    if not results:
        st.info("Run `python train_models.py` to refresh model scorecards.")
        return

    cols = st.columns(3)
    labels = {
        "funding": "Funding amount",
        "success": "Funding readiness",
        "industry": "Industry classifier",
    }
    for idx, key in enumerate(["funding", "success", "industry"]):
        result = results.get(key, {})
        metrics = result.get("metrics", {})
        with cols[idx]:
            st.markdown(f"**{labels[key]}**")
            st.caption(result.get("best_model", "Model not trained"))
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    st.metric(metric_name, f"{value:,.4f}" if isinstance(value, float) else f"{value:,}")


def render_overview(df: pd.DataFrame) -> None:
    cover = IMAGES_DIR / "cover.png"
    if cover.exists():
        st.image(str(cover), use_container_width=True)

    st.title("Startup Funding Analyzer")
    st.write(
        "Explore Indian startup funding patterns, compare a new idea with similar funded companies, "
        "and estimate its likely funding range and readiness signal."
    )
    metric_row(df)
    render_model_cards(load_training_results())

    st.subheader("Featured analysis")
    image_files = [
        "funding_trend.png",
        "top_locations_funding.png",
        "funding_distribution.png",
        "industry_pie.png",
    ]
    cols = st.columns(2)
    for idx, filename in enumerate(image_files):
        path = IMAGES_DIR / filename
        if path.exists():
            cols[idx % 2].image(str(path), use_container_width=True)


def render_market_explorer(df: pd.DataFrame) -> None:
    st.title("Market explorer")
    industries = sorted(df["Industry In"].dropna().unique())
    locations = sorted(df["Head Quarter"].dropna().unique())

    col1, col2, col3 = st.columns([2, 2, 1])
    selected_industries = col1.multiselect("Industries", industries, default=industries[:8])
    selected_locations = col2.multiselect("Locations", locations)
    min_year, max_year = int(df["Funding Year"].min()), int(df["Funding Year"].max())
    year_range = col3.slider("Funding years", min_year, max_year, (min_year, max_year))

    filtered = df[df["Funding Year"].between(*year_range)]
    if selected_industries:
        filtered = filtered[filtered["Industry In"].isin(selected_industries)]
    if selected_locations:
        filtered = filtered[filtered["Head Quarter"].isin(selected_locations)]

    metric_row(filtered)

    trend = filtered.groupby("Funding Year", as_index=False)["Amount in ($)"].sum()
    by_industry = (
        filtered.groupby("Industry In", as_index=False)["Amount in ($)"]
        .sum()
        .sort_values("Amount in ($)", ascending=False)
        .head(15)
    )
    col1, col2 = st.columns(2)
    col1.plotly_chart(
        px.line(trend, x="Funding Year", y="Amount in ($)", markers=True, title="Funding by year"),
        use_container_width=True,
    )
    col2.plotly_chart(
        px.bar(by_industry, x="Amount in ($)", y="Industry In", orientation="h", title="Top funded industries"),
        use_container_width=True,
    )
    st.dataframe(filtered.sort_values("Amount in ($)", ascending=False).head(200), use_container_width=True)


def render_startup_predictor(df: pd.DataFrame, models: dict) -> None:
    st.title("Funding readiness predictor")
    if "funding" not in models or "success" not in models:
        st.warning("Funding and readiness models are not available. Run `python train_models.py` first.")
        return

    industries = sorted(df["Industry In"].dropna().unique())
    locations = sorted(df["Head Quarter"].dropna().unique())
    rounds = sorted(df["Funding Round/Series"].dropna().unique())

    with st.form("startup_prediction_form"):
        col1, col2, col3 = st.columns(3)
        company_name = col1.text_input("Startup name", "Sample AI Health")
        fintech_index = industries.index("FinTech") if "FinTech" in industries else 0
        industry = col2.selectbox("Industry", industries, index=fintech_index)
        location_index = locations.index("Bengaluru") if "Bengaluru" in locations else 0
        head_quarter = col3.selectbox("Head quarter", locations, index=location_index)

        col1, col2, col3 = st.columns(3)
        year_founded = col1.number_input("Year founded", min_value=1980, max_value=REFERENCE_YEAR, value=2022)
        funding_year = col2.number_input("Funding year", min_value=2018, max_value=REFERENCE_YEAR, value=REFERENCE_YEAR)
        round_index = rounds.index("Seed") if "Seed" in rounds else 0
        funding_round = col3.selectbox("Target round", rounds, index=round_index)

        about_company = st.text_area(
            "What the startup does",
            "AI-enabled operating system that helps clinics predict patient demand and manage working capital.",
            height=110,
        )
        submitted = st.form_submit_button("Score startup", use_container_width=True)

    if not submitted:
        st.info("Fill in a startup concept and score it against the historical funding patterns.")
        return

    startup = startup_input_frame(
        company_name=sanitize_csv_cell(company_name),
        year_founded=int(year_founded),
        funding_year=int(funding_year),
        head_quarter=head_quarter,
        industry=industry,
        about_company=sanitize_csv_cell(about_company),
        funding_round=funding_round,
    )
    estimated_funding = float(predict_funding(models["funding"], startup)[0])
    readiness = float(predict_success(models["success"], startup)[0])

    if "industry" in models and about_company.strip():
        inferred_industry = classify_industry(models["industry"], [about_company])[0]
    else:
        inferred_industry = "Not available"

    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated funding", currency(estimated_funding))
    col2.metric("Readiness probability", f"{readiness:.0%}", readiness_label(readiness))
    col3.metric("Description fit", str(inferred_industry))

    similar = find_similar_startups(df, startup, top_n=8)
    st.subheader("Similar funded startups")
    display_cols = ["CompanyName", "Industry In", "Head Quarter", "Funding Round/Series", "Amount in ($)", "Funding Year"]
    st.dataframe(similar[display_cols], use_container_width=True)


def render_batch_predictions(models: dict) -> None:
    st.title("Batch predictions")
    uploaded = st.file_uploader("Upload startup CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV with startup fields such as company, industry, location, founded year, stage, and description.")
        return
    raw = pd.read_csv(uploaded)
    clean = clean_startup_dataframe(raw, require_amount=False)
    if clean.empty:
        st.error("No usable rows found after cleaning. Check the founded year or funding year columns.")
        return

    if "funding" in models:
        clean["Predicted Funding ($)"] = predict_funding(models["funding"], clean)
    if "success" in models:
        clean["Funding Readiness Probability"] = predict_success(models["success"], clean)
    if "industry" in models:
        clean["Predicted Industry"] = classify_industry(models["industry"], clean["AboutCompany"])

    st.dataframe(clean, use_container_width=True)
    st.download_button(
        "Download predictions",
        data=clean.to_csv(index=False, lineterminator="\n"),
        file_name="startup_predictions.csv",
        mime="text/csv",
    )


def sanitize_prompt_input(text: str, max_len: int = 1200) -> str:
    value = sanitize_csv_cell(text, default="")
    blocked_phrases = [
        "ignore previous instructions",
        "ignore all instructions",
        "system prompt",
        "developer message",
        "you are now",
        "act as",
        "override",
    ]
    lowered = value.lower()
    for phrase in blocked_phrases:
        if phrase in lowered:
            value = value.replace(phrase, "[redacted]")
    return value[:max_len]


def render_ai_assistant(df: pd.DataFrame) -> None:
    st.title("Data assistant")
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        st.warning("Set `GOOGLE_AI_API_KEY` in a local `.env` file to enable the Gemini assistant.")
        return

    import google.generativeai as genai

    profile = dataset_profile(df)
    context = (
        f"Dataset records: {profile['records']}. "
        f"Total funding: {currency(profile['total_funding_usd'])}. "
        f"Industries: {profile['industries']}. Locations: {profile['locations']}. "
        f"Funding years: {profile['funding_year_range']}."
    )

    question = st.chat_input("Ask about the funding data")
    if not question:
        st.info("Ask a focused question about funding trends, industries, locations, or model output.")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=(
            "You are a startup funding data assistant. Answer only from the supplied dataset context, "
            "avoid investment advice, and call out uncertainty."
        ),
    )
    prompt = f"Context: {context}\n\nUser question: {sanitize_prompt_input(question)}"
    with st.spinner("Analyzing..."):
        response = model.generate_content(prompt)
    st.write(response.text)


def main() -> None:
    data = load_data()
    models = load_models()

    st.sidebar.title("Startup Funding Analyzer")
    st.sidebar.caption("Indian startup funding data, 2018-2021")
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Market explorer", "Funding predictor", "Batch predictions", "Data assistant"],
    )
    st.sidebar.divider()
    st.sidebar.write(f"Clean records: **{len(data):,}**")
    st.sidebar.write(f"Models loaded: **{len(models)}/3**")

    if page == "Overview":
        render_overview(data)
    elif page == "Market explorer":
        render_market_explorer(data)
    elif page == "Funding predictor":
        render_startup_predictor(data, models)
    elif page == "Batch predictions":
        render_batch_predictions(models)
    else:
        render_ai_assistant(data)


if __name__ == "__main__":
    main()
