# Test file focuses on fairness aspect of AI QA
import pandas as pd
from src.utils import load_model

model = load_model("model/loan_model.pkl")

def test_no_age_bias():
    data = pd.DataFrame({
        "income": [40, 40, 60, 60],
        "credit_score": [650, 650, 700, 700],
        "age": [25, 45, 25, 45]
    })
    preds = model.predict(data)
    avg_young = preds[:2].mean()
    avg_old = preds[2:].mean()
    bias_gap = abs(avg_young - avg_old)
    assert bias_gap < 0.3, f"Age bias detected! Gap={bias_gap:.2f}"
