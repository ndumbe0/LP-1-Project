import subprocess
import time
import os


print("Running setup tasks...")



streamlit_proc = subprocess.Popen([
    "streamlit", 
    "run", 
    os.path.join(os.path.dirname(__file__), "startup_analyzer.py"),  
    "--server.port=8501"                                
])

try:
    while True:
        time.sleep(1)  
except KeyboardInterrupt:
    streamlit_proc.terminate()
    print("App closed.")