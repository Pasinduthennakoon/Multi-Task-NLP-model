import streamlit as st
import requests

st.title("Content Safety Classifier")
user_input = st.text_area("Enter text to analyze:")

if st.button("Predict"):
    if user_input:
        response = requests.post("http://127.0.0.1:8000/predict", json={"text": user_input})
        
        if response.status_code == 200:
            res = response.json()
            
            st.json(res)
        else:
            st.error(f"Error: {response.status_code}")
    else:
        st.warning("Please enter some text first!")