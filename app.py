from fastapi import FastAPI
from prediction_pipeline import predict_content

app = FastAPI()

@app.post('/predict')
def predict_result(text):
    return predict_content(text)