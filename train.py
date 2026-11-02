"""Train the churn model. Run:  python train.py
Reads mlops_dataset.csv and saves churn_model.pkl, model_columns.pkl, model_info.json"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import prepare

df = pd.read_csv("mlops_dataset.csv")
df["Internet_Service"] = df["Internet_Service"].fillna("None")   # "None" = no internet
df = df.drop(columns=["Customer_ID"])

num_cols = ["Tenure_Months", "Monthly_Charges_INR", "Total_Charges_INR",
            "Support_Tickets_6M", "Avg_Data_Usage_GB"]
cat_cols = ["Gender", "Senior_Citizen", "Contract_Type", "Internet_Service",
            "Payment_Method", "Tech_Support", "Online_Security"]

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df[c] < 0, c] = np.nan
df = df.dropna(subset=["Churn"])
df["Churn"] = df["Churn"].map({"Churn": 1, "Not Churn": 0})
df = df.dropna(subset=["Churn"])
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

data_median = float(df["Avg_Data_Usage_GB"].median())
y = df["Churn"]
X = prepare(df.drop(columns=["Churn"]), data_median)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

best_name, best_f1, best_model, scores = None, -1, None, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores[name] = {"accuracy": round(accuracy_score(y_test, pred), 3),
                    "f1": round(f1_score(y_test, pred), 3)}
    print(name, scores[name])
    if scores[name]["f1"] > best_f1:
        best_name, best_f1, best_model = name, scores[name]["f1"], model

joblib.dump(best_model, "churn_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")
with open("model_info.json", "w") as f:
    json.dump({"best_model": best_name, "data_median": data_median, "scores": scores}, f, indent=2)
print("Best model:", best_name, "- saved churn_model.pkl, model_columns.pkl, model_info.json")
