import uvicorn
from pathlib import Path
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from yaml import safe_load
import joblib

from src.training import train_model

with open("src/config.yml", "r") as f:
    CONFIG = safe_load(f)

model_pipeline = joblib.load(CONFIG["model"]["path"])

app = FastAPI()

class ModelInput(BaseModel):
    account_status: str
    duration: int
    credit_history: str 
    purpose: str
    credit_amount: int
    savings: str
    employment: str
    installment_rate: int
    personal_status_and_sex: str
    other_debtors: str 
    residence_years: int
    property: str
    age: int
    other_installments: str
    housing: str
    number_of_credits: int
    job: str
    number_of_people_liable: int
    telephone: str
    foreign_worker: str


@app.get("/")
async def root():
    return {"message": "Quick Credit ML Model!"}


@app.get("/train")
async def train():
    print("Training model....")
    train_model()
    return {"message": "Completed model training"}


@app.post("/predict")
async def predict(model_input: ModelInput):
    data = pd.DataFrame(model_input.dict(), index=[0])
    y_pred = model_pipeline.predict(data)
    y_pred_prob = model_pipeline.predict_proba(data)
    return JSONResponse(
        {
            "message": "Prediction",
            "bad_loan": int(y_pred[0]),
            "bad_loan_prob": float(y_pred_prob[0][1])
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
