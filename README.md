# Fraud Detection Project
This project focuses on detecting fraudulent transactions using machine learning techniques on a large-scale financial dataset. It aims to assist financial institutions in flagging suspicious activities and minimizing loss due to fraud.

## Project Structure
- `data/` — Contains the raw input dataset (`transactions.csv`)
- `models/` — Stores trained models and encoders (`xgb_model.joblib`, `scaler.joblib`, `type_encoder.joblib`)
- `notebook/` — Includes the complete Jupyter notebook (`fraud_detection.ipynb`) with data exploration, preprocessing, training, and evaluation
- `app.py` — Streamlit application for prediction and model explanation
- `README.md` — Project documentation

## Dataset
- Approximately 6 million transaction records in CSV format
- Fields include: transaction type, amount, origin/destination balances, and fraud flags
- Target variable: `isFraud`

## Model Details
- **Algorithm**: XGBoost Classifier
- **Data Techniques**: Label Encoding, SMOTE for class imbalance, Standard Scaling
- **Evaluation Metric**: ROC-AUC Score
- **Performance**:  
  - ROC AUC Score on test data: **0.9995**

## Implementation Workflow
1. Data loading and initial inspection
2. Cleaning and feature engineering
   - Removing irrelevant columns (`nameOrig`, `nameDest`)
   - Creating new features: `errorBalanceOrig`, `errorBalanceDest`
3. Label encoding categorical features
4. Addressing class imbalance using SMOTE
5. Splitting dataset into train/test
6. Scaling features using StandardScaler
7. Training XGBoost model
8. Evaluating performance with metrics and ROC-AUC
9. Interpreting model predictions with SHAP
10. Deploying a simple Streamlit interface

## How to Run
1. Clone or download the repository
2. Place your `transactions.csv` file inside the `data/` directory
3. Install required packages:
```bash
pip install -r requirements.txt

Run the notebook to generate models (if not already created):
jupyter notebook notebook/fraud_detection.ipynb
Launch the web app:
streamlit run app.py

Business Insights
Most frauds occur in TRANSFER and CASH_OUT transactions
Large gaps in balance inconsistencies often correlate with fraud
Real-time flagging is possible using a trained ML model with minimal features
Model interpretability (via SHAP) supports compliance and explainability for flagged transactions

Deliverables
Jupyter notebook with clean modular code and visualizations
Streamlit dashboard for single transaction predictions
Trained model files (xgb_model.joblib, scaler.joblib, type_encoder.joblib)
README documentation

Dependencies
Python 3.9+
pandas, numpy, matplotlib, seaborn
scikit-learn, xgboost, imbalanced-learn
shap, streamlit, joblib

## Author
Kartik Sharma
