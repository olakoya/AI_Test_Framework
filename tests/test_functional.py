# Test file focuses on functional aspect of AI QA.
from src.utils import load_model
import pandas as pd
import numpy as np

model = load_model("model/loan_model.pkl")

def test_prediction_output_type():
    X_sample = pd.DataFrame([[40, 650, 30]], columns=["income", "credit_score", "age"])
    y_pred = model.predict(X_sample)
    assert y_pred.dtype == np.int64
    assert y_pred[0] in [0, 1]
