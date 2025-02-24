import streamlit as st

# Set the title for the app
st.set_page_config(page_title="Startup Analyzer", page_icon="📈")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Funding Predictor", "Startup Success", "Industry Classifier"))

# Load the selected page
if page == "Home":
    import Home
    Home.main()
elif page == "Funding Predictor":
    import Funding_Predictor
    Funding_Predictor.main()
elif page == "Startup Success":
    import Startup_Success
    Startup_Success.main()
elif page == "Industry Classifier":
    import Industry_Classifier
    Industry_Classifier.main()
