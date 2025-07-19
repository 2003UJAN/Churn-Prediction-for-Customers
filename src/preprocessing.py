import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path="data/customer_churn.csv"):
    df = pd.read_csv(path)
    df.drop(columns=['customerID'], inplace=True)
    df.replace(" ", pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    # Convert total charges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    # Encode categorical
    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
