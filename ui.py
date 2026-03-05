import streamlit as st
import requests

st.title("🛡️ Content Safety Classifier")
user_input = st.text_area("Enter text to analyze:")

if st.button("Predict"):
    if user_input:
        response = requests.post("http://127.0.0.1:8000/predict", json={"text": user_input})
        
        if response.status_code == 200:
            res = response.json()
            
            # This displays the raw dictionary without any extra UI components
            st.json(res)
            
            # OR, if you want it to look like a simple code block:
            # st.code(res)
        else:
            st.error(f"Error: {response.status_code}")
    else:
        st.warning("Please enter some text first!")