# Test file focuses on performance aspect of AI QA

import pandas as pd
from src.utils import load_model, evaluate_model

import os
import joblib

# Construct absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "loan_model.pkl")
model_path = os.path.abspath(model_path)

model = joblib.load(model_path)


def test_model_accuracy_threshold():
    test_data = pd.DataFrame({
        "income": [30, 80, 45, 60],
        "credit_score": [620, 710, 640, 690],
        "age": [27, 42, 33, 38],
        "loan_approved": [0, 1, 1, 1]
    })
    X_test = test_data[["income", "credit_score", "age"]]
    y_test = test_data["loan_approved"]
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy >= 0.75, f"Accuracy too low: {accuracy:.2f}"
