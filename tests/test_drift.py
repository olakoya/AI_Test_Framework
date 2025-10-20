# Test file focuses on drift or data consistency aspect of AI QA
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def test_data_drift():
    ref_data = pd.DataFrame({
        "income": [30, 50, 80, 20, 45, 70, 100, 60],
        "credit_score": [600, 650, 720, 580, 640, 700, 750, 680],
        "age": [25, 30, 40, 22, 28, 35, 45, 32]
    })
    new_data = pd.DataFrame({
        "income": [10, 15, 20, 25, 30],
        "credit_score": [400, 420, 450, 470, 480],
        "age": [18, 20, 21, 22, 25]
    })
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=new_data)
    result = report.as_dict()
    drift_detected = result['metrics'][0]['result']['data']['drift_detected']
    assert not drift_detected, "Data drift detected!"
