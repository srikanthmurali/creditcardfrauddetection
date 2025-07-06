import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predictor import load_model, predict
import pandas as pd

def test_model_loading():
    model = load_model("model/best_xgb.pkl")
    assert model is not None

def test_prediction_shape():
    model = load_model("model/best_xgb.pkl")
    # Correct feature order: scaled_amount, scaled_time, V1 to V28
    cols = ['scaled_amount', 'scaled_time'] + [f"V{i}" for i in range(1, 29)]
    sample = pd.DataFrame([[0.1] * 30], columns=cols)

    prediction = predict(model, sample)
    assert prediction.shape == (1,)