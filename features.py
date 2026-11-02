"""Feature engineering + encoding. Same steps as the notebook (used by train.py and app.py)."""
import pandas as pd


def add_features(df, data_median):
    df = df.copy()
    df["Tenure_Group"] = pd.cut(df["Tenure_Months"], bins=[-1, 12, 24, 48, 100],
                                labels=["0-1yr", "1-2yr", "2-4yr", "4yr+"]).astype(str)
    df["Avg_Monthly_Spend"] = df["Total_Charges_INR"] / df["Tenure_Months"].replace(0, 1)
    df["Has_Tech_Support"] = (df["Tech_Support"] == "Yes").astype(int)
    df["Has_Online_Security"] = (df["Online_Security"] == "Yes").astype(int)
    df["Has_Internet"] = (df["Internet_Service"] != "None").astype(int)
    df["High_Data_User"] = (df["Avg_Data_Usage_GB"] > data_median).astype(int)
    df["Many_Tickets"] = (df["Support_Tickets_6M"] >= 3).astype(int)
    return df


def encode(df):
    df = df.copy()
    df["Senior_Citizen"] = df["Senior_Citizen"].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["Tech_Support"] = df["Tech_Support"].map({"Yes": 1, "No": 0})
    df["Online_Security"] = df["Online_Security"].map({"Yes": 1, "No": 0})
    return pd.get_dummies(df, columns=["Contract_Type", "Internet_Service",
                                       "Payment_Method", "Tenure_Group"], dtype=int)


def prepare(df, data_median):
    return encode(add_features(df, data_median))


def predict_one(model, columns, data_median, customer):
    """customer = dict with the 12 raw input columns. Returns (label, churn_probability)."""
    row = prepare(pd.DataFrame([customer]), data_median)
    row = row.reindex(columns=columns, fill_value=0)
    prob = float(model.predict_proba(row)[0][1])
    return ("Churn" if prob >= 0.5 else "Not Churn"), round(prob, 4)
