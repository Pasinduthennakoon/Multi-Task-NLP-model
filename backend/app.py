from fastapi import FastAPI
from prediction_pipeline import predict_content
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

@app.post('/predict')
def predict_result(text : PredictionRequest):
    return predict_content(text.text)