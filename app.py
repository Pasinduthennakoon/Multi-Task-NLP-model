from fastapi import FastAPI
from prediction_pipeline import predict_content
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend
origins = ["*"]  # Allow all origins for simplicity (good for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str

@app.post('/predict')
def predict_result(text : PredictionRequest):
    return predict_content(text.text)