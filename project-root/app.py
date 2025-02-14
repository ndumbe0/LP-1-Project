# app.py
import streamlit as st

st.set_page_config(
    page_title="Startup Analytics Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Home",
        "Funding Predictor",
        "Startup Success",
        "Industry Classifier"
    ])

    if page == "Home":
        st.switch_page("pages/1_🏠_Home.py")
    elif page == "Funding Predictor":
        st.switch_page("pages/2_💰_Funding_Predictor.py")
    elif page == "Startup Success":
        st.switch_page("pages/3_🚀_Startup_Success.py")
    elif page == "Industry Classifier":
        st.switch_page("pages/4_🏭_Industry_Classifier.py")

if __name__ == "__main__":
    main()