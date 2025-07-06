import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.predictor import load_model, predict

st.title("Credit Card Fraud Detection - Confusion Matrix Validator")

# Upload X_test
x_file = st.file_uploader("Upload X_test.csv (30 feature columns)", type=["csv"])
y_file = st.file_uploader("Upload y_test.csv (labels)", type=["csv"])

if x_file and y_file:
    try:
        # Load files
        X_test = pd.read_csv(x_file)
        y_test = pd.read_csv(y_file).squeeze()  # Make sure it's a Series

        # Validate input shape
        if X_test.shape[1] != 30:
            st.error("X_test must have 30 columns.")
        elif len(X_test) != len(y_test):
            st.error("X_test and y_test row count mismatch.")
        else:
            # Load model and predict
            model = load_model("model/best_xgb.pkl")
            y_pred = predict(model, X_test)

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
