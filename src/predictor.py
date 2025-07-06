# src/predictor.py

import pickle
import pandas as pd

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def predict(model, X: pd.DataFrame):
    return model.predict(X)
