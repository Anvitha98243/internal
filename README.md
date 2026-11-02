# Customer Churn Prediction (FastAPI + Docker + GitHub Actions)

- `churn_prediction_synthetic.ipynb` - notebook: data, preprocessing, training, testing
- `train.py` - same training as a script (`python train.py`), saves the model files
- `features.py` - feature engineering used by training and the API
- `app.py` - FastAPI app: `GET /health`, `POST /predict`
- `test_app.py` - automated tests (pytest)
- `Dockerfile` - builds the app image
- `.github/workflows/ci.yml` - install, validate, test, build and push image to GHCR

## Run locally
```
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://127.0.0.1:8000/docs

## Run the Docker image
```
docker run -p 8000:8000 ghcr.io/<your-github-username>/<repo-name>:latest
```

## Example /predict body
```
{"Gender":"Male","Senior_Citizen":"No","Tenure_Months":5,"Contract_Type":"Month-to-month",
 "Internet_Service":"Fiber","Payment_Method":"UPI","Monthly_Charges_INR":1000,
 "Total_Charges_INR":5000,"Tech_Support":"No","Online_Security":"No",
 "Support_Tickets_6M":4,"Avg_Data_Usage_GB":120}
```
