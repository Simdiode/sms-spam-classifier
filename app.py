from fastapi import FastAPI
from pydantic import BaseModel

from inference import predict_sms

app = FastAPI(title="SMS Spam Classifier")


class PredictionRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    return predict_sms(request.text)