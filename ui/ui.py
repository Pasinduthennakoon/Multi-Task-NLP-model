import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL")

st.title("Content Safety Classifier")
user_input = st.text_area("Enter text to analyze:")

if st.button("Predict"):
    if user_input:
        response = requests.post(f"{API_URL}/predict", json={"text": user_input})
        
        if response.status_code == 200:
            res = response.json()
            
            st.json(res)
        else:
            st.error(f"Error: {response.status_code}")
    else:
        st.warning("Please enter some text first!")