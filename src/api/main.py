from fastapi import FastAPI
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import mlflow.sklearn
import pandas as pd

app = FastAPI()

mlflow.set_tracking_uri("file:src/mlruns")
# Load best model from MLflow

model_uri = "models:/CreditRiskModel/1"
model = mlflow.sklearn.load_model(model_uri)


@app.get("/")
def root():
    return {"message": "Credit Risk Model API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    input_df = pd.DataFrame([request.dict()])
    probability = model.predict_proba(input_df)[0][1]
    return PredictionResponse(risk_probability=round(probability, 4))
