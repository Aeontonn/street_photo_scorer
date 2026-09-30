"""Hugging Face 'Gradio' Space entrypoint that just launches the Streamlit app (HF serves port 7860)."""
import os
import sys

os.execvp(sys.executable, [sys.executable, "-m", "streamlit", "run", "src/app/Score_Photos.py",
                           "--server.port=7860", "--server.address=0.0.0.0"])
