import os
import streamlit as st

# Checks environment variable first, then falls back to Streamlit secrets on the cloud
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
  try:
    api_key = st.secrets.get("GOOGLE_API_KEY")
  except Exception:
    api_key = None

if not api_key:
  st.error(
      "API Key missing! Please set GOOGLE_API_KEY in your environment or"
      " Streamlit Secrets."
  )
  st.stop()