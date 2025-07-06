# Credit Card Fraud Detection - MLOps Project

This is an end-to-end Machine Learning and MLOps-ready project to detect fraudulent credit card transactions using a trained XGBoost model. The project includes modular code, unit testing, CI pipeline, and a deployed Streamlit app that validates predictions using uploaded test files.

---

## Problem Statement

Credit card fraud is a serious issue in the financial sector, causing massive losses. Detecting fraudulent transactions quickly and accurately is essential. This project uses machine learning to build a predictive model to classify transactions as **fraudulent (1)** or **legitimate (0)** based on 30 anonymized features.

---

## Dataset Used

- Dataset: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains:
  - 284,807 transactions
  - 30 anonymized features (V1 to V28, `Amount`, `Time`)
  - Binary target: `Class` (1 = fraud, 0 = legit)

---

## Model and Evaluation

- **Model used:** XGBoost Classifier  
- **Accuracy:** ~99.93%
- **Evaluation metric:** Confusion Matrix, Precision, Recall, F1 Score

---

## Project Structure

creditcardfrauddetection/
│
├── app/ # Streamlit app
│ └── main.py
│
├── model/ # Saved ML models
│ └── best_xgb.pkl
│
├── notebooks/ # Exploratory notebooks
│ └── CreditCardFraudDetection.ipynb
│
├── src/ # Modularized code
│ └── predictor.py
│
├── tests/ # Unit tests
│ └── test_predictor.py
│
├── .github/workflow/ # GitHub Actions CI
│ └── python-ci.yml
│
├── requirements.txt # Required packages
├── README.md # Project readme
└── .gitignore


---

## Streamlit App Demo

Upload your `X_test.csv` and `y_test.csv` files to get model predictions and validate against the actual labels using a confusion matrix.

 **[Launch App Here](https://creditcardfrauddetection-mm8pdgxoarp6qdnpxbsjjf.streamlit.app/)**  
_(Replace with your Streamlit Cloud link)_

---

## How to Run Locally

```bash
# Step 1: Clone the repository
git clone https://github.com/srikanthmurali/creditcardfrauddetection.git
cd creditcardfrauddetection

# Step 2: Set up virtual environment (optional but recommended)
python3 -m venv fraud_env
source fraud_env/bin/activate  # On Mac/Linux

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the Streamlit app
streamlit run app/main.py
