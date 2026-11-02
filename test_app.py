from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

SAMPLE = {
    "Gender": "Male", "Senior_Citizen": "No", "Tenure_Months": 5,
    "Contract_Type": "Month-to-month", "Internet_Service": "Fiber",
    "Payment_Method": "UPI", "Monthly_Charges_INR": 1000, "Total_Charges_INR": 5000,
    "Tech_Support": "No", "Online_Security": "No",
    "Support_Tickets_6M": 4, "Avg_Data_Usage_GB": 120,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert r.json()["prediction"] in ["Churn", "Not Churn"]
    assert 0 <= r.json()["churn_probability"] <= 1


def test_predict_bad_input():
    bad = dict(SAMPLE, Contract_Type="Weekly")
    assert client.post("/predict", json=bad).status_code == 422
