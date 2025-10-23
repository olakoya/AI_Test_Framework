# tests/test_drift.py

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import ValueDrift
from evidently.report import Report



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

    # Optional column mapping
    column_mapping = ColumnMapping(
        numerical_features=["income", "credit_score", "age"]
    )

    # # Create a TestSuite using DataDriftTestPreset
    # suite = TestSuite(tests=[DataDriftTestPreset()])
    # suite.run(reference_data=ref_data, current_data=new_data, column_mapping=column_mapping)

    report = Report(metrics=[ValueDrift()])
    report.run(reference_data=ref_data, current_data=new_data)
    result = report.as_dict()

    # Convert results to dict
    result = report.as_dict()

    # Check if drift is detected
    drift_detected = not result["summary"]["all_passed"]

    assert not drift_detected, "Data drift detected!"
