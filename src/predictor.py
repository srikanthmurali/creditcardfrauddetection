# src/predictor.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import pandas as pd

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def predict(model, X: pd.DataFrame):
    return model.predict(X)
