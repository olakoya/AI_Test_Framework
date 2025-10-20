#PREDICTION LOGIC

import numpy as np
import pandas as pd
import joblib

model = joblib.load("loan_model.pkl")

def test_prediction_output_type():
    X_sample = pd.DataFrame([[40, 650, 30]], columns=["income", "credit_score", "age"])
    y_pred = model.predict(X_sample)
    assert y_pred.dtype == np.int64, "Prediction should be integer (0 or 1)"
    assert y_pred[0] in [0, 1], "Prediction should be either 0 or 1"
