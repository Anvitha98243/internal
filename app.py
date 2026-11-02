import json
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from features import predict_one

BASE = Path(__file__).parent
model = joblib.load(BASE / "churn_model.pkl")
columns = joblib.load(BASE / "model_columns.pkl")
data_median = json.loads((BASE / "model_info.json").read_text())["data_median"]

app = FastAPI(title="Customer Churn Prediction")


class Customer(BaseModel):
    Gender: Literal["Male", "Female"]
    Senior_Citizen: Literal["Yes", "No"]
    Tenure_Months: float = Field(ge=0)
    Contract_Type: Literal["Month-to-month", "One year", "Two year"]
    Internet_Service: Literal["DSL", "Fiber", "None"]
    Payment_Method: Literal["Cash", "UPI", "Credit Card", "Bank Transfer"]
    Monthly_Charges_INR: float = Field(ge=0)
    Total_Charges_INR: float = Field(ge=0)
    Tech_Support: Literal["Yes", "No"]
    Online_Security: Literal["Yes", "No"]
    Support_Tickets_6M: float = Field(ge=0)
    Avg_Data_Usage_GB: float = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    label, prob = predict_one(model, columns, data_median, customer.model_dump())
    return {"prediction": label, "churn_probability": prob}
