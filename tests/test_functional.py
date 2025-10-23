# Test file focuses on functional aspect of AI QA.

from src.utils import load_model
import pandas as pd
import numpy as np

import os
import joblib

# Construct absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "loan_model.pkl")
model_path = os.path.abspath(model_path)

model = joblib.load(model_path)


def test_prediction_output_type():
    X_sample = pd.DataFrame([[40, 650, 30]], columns=["income", "credit_score", "age"])
    y_pred = model.predict(X_sample)
    assert y_pred.dtype == np.int64
    assert y_pred[0] in [0, 1]
