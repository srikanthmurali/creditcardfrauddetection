import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.title("Credit Card Fraud Detection - Confusion Matrix Validator")

# Load model
with open("model/best_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Upload test features (X_test)
x_file = st.file_uploader("Upload X_test file (CSV with 30 features)", type=["csv"])

# Upload true labels (y_test)
y_file = st.file_uploader("Upload y_test file (CSV with single column of labels)", type=["csv"])

if x_file and y_file:
    try:
        X_test = pd.read_csv(x_file)
        y_test = pd.read_csv(y_file).squeeze()  # Converts to Series if single column

        # Validate shapes
        if X_test.shape[1] != 30:
            st.error(f"X_test must have 30 columns. Found: {X_test.shape[1]}")
        elif len(X_test) != len(y_test):
            st.error(f"Mismatch: X_test has {len(X_test)} rows, y_test has {len(y_test)}")
        else:
            # Make predictions
            y_pred = model.predict(X_test)

            # Show basic output
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            labels = ["Legit", "Fraud"]

            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            st.pyplot(fig)

            # Optional: Summary
            st.write("Raw Confusion Matrix:")
            st.dataframe(pd.DataFrame(cm, index=labels, columns=labels))

    except Exception as e:
        st.error(f"Error processing input files: {e}")
