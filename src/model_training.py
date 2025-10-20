# Trains and saves your model.
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    data = pd.DataFrame({
        "income": [30, 50, 80, 20, 45, 70, 100, 60],
        "credit_score": [600, 650, 720, 580, 640, 700, 750, 680],
        "age": [25, 30, 40, 22, 28, 35, 45, 32],
        "loan_approved": [0, 1, 1, 0, 1, 1, 1, 1]
    })

    X = data[["income", "credit_score", "age"]]
    y = data["loan_approved"]

    model = LogisticRegression().fit(X, y)
    joblib.dump(model, "model/loan_model.pkl")
    print("✅ Model trained and saved successfully.")

if __name__ == "__main__":
    train_model()
