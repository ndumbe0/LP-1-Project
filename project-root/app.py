import streamlit as st
import socket
import sys
import webbrowser
import time
import subprocess
import os
import atexit


LOCK_FILE = "streamlit_app.lock"

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def create_lock_file():
    """Create a lock file to indicate the app is running."""
    if os.path.exists(LOCK_FILE):
        print("Another instance of the app is already running.")
        sys.exit(0)  # Exit gracefully
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))  # Write the current process ID to the lock file

def delete_lock_file():
    """Delete the lock file when the app exits."""
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Home",
        "Funding Predictor",
        "Startup Success",
        "Industry Classifier"
    ])

    ctx = st.runtime.scriptrunner.get_script_run_ctx()
    if ctx is None:
        st.error("No session context found. Please rerun the app.")
        return

    if page == "Home":
        st.switch_page("pages/1__Home.py")
    elif page == "Funding Predictor":
        st.switch_page("pages/2__Funding_Predictor.py")
    elif page == "Startup Success":
        st.switch_page("pages/3__Startup_Success.py")
    elif page == "Industry Classifier":
        st.switch_page("pages/4__Industry_Classifier.py")

if __name__ == "__main__":
    
    create_lock_file()

    
    atexit.register(delete_lock_file)

    
    if is_port_in_use(8501):
        print("Port 8501 is already in use. Please close the other instance.")
        delete_lock_file()  
        sys.exit(0)

    
    st.set_page_config(
        page_title="Startup Analytics Suite",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    
    process = subprocess.Popen(["streamlit", "run", __file__])

    
    time.sleep(3)  
    
    
    webbrowser.open("http://localhost:8501")

    # Run the main function
    main()