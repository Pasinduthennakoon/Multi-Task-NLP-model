# Multi-Task NLP Prediction API

A web-based AI application that predicts **emotion**, **violence category**, and **hate speech class** from text input using a **multi-task deep learning model**. Built with **FastAPI**, **TensorFlow/Keras**, and **NLTK**, featuring a clean **HTML/JS frontend** and **Dockerized deployment**.

---

## Features

- Predict **Emotion**: sadness, joy, love, anger, fear, surprise
- Predict **Violence Category**: harmful_traditional_practices, physical_violence, economic_violence, emotional_violence, sexual_violence
- Predict **Hate Class**: offensive_speech, hate_speech, neither
- Interactive frontend for quick testing
- REST API with **CORS enabled**
- Containerized with **Docker** for easy deployment

---

## Project Structure


Multi-Task-NLP/
├── app.py                     # FastAPI backend server
├── prediction_pipeline.py     # Text preprocessing & model prediction
├── index.html                 # Simple frontend interface
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container instructions
├── .dockerignore              # Files to exclude from Docker build
├── .gitignore                 # Files to exclude from Git
└── model/                     # Directory for model assets
    ├── model.h5               # Trained Keras model
    └── tokenizer.pkl          # Tokenizer for preprocessing


---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/multi-task-nlp.git
cd multi-task-nlp

pip install -r requirements.txt
