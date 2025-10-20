# Contains helper functions for predictions and metric checks.
import joblib
from sklearn.metrics import accuracy_score

def load_model(model_path):
    return joblib.load(model_path)

def evaluate_model(model, X, y):
    preds = model.predict(X)
    return accuracy_score(y, preds)
