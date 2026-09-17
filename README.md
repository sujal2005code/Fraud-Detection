# 🛡️ Fraud Detection ML System

An end-to-end machine learning project for detecting fraudulent financial transactions using **XGBoost**, **SMOTE**, feature engineering, and **SHAP explainability**. It also includes an interactive **Streamlit dashboard** for transaction-level fraud risk assessment.

## 🚀 Overview

```text
PaySim Dataset
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Train / Test Split
      ↓
SMOTE on Training Data
      ↓
XGBoost Training
      ↓
Model Evaluation
      ↓
SHAP Explainability
      ↓
Streamlit Dashboard
```

### Key Features

- Fraud / non-fraud classification
- Severe class-imbalance handling with SMOTE
- Transaction feature engineering
- XGBoost classification
- Accuracy, Precision, Recall and F1 evaluation
- Probability-based ROC-AUC evaluation
- SHAP explainability
- Interactive transaction risk assessment
- Transaction-level analysis
- Dedicated SHAP prediction breakdown page

---

## 📊 Dataset

This project uses the **PaySim synthetic financial transaction dataset**.

The dataset contains:

- Transaction type
- Transaction amount
- Origin account balances
- Destination account balances
- Fraud labels
- Flagged-fraud indicators
- Time steps

The full dataset used for training contains approximately **6.36 million transactions**.

The Streamlit dashboard loads the first **10,000 transactions** for lightweight interactive analysis.

---

## 🧠 Feature Engineering

Two engineered features are used:

### Origin Balance Error

```text
errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
```

### Destination Balance Error

```text
errorBalanceDest = oldbalanceDest + amount - newbalanceDest
```

The account identifiers `nameOrig` and `nameDest` are excluded from model training.

The categorical `type` feature is encoded using a saved `LabelEncoder`.

---

## 🤖 Machine Learning

The project uses an **XGBoost Classifier**.

Because fraud is highly underrepresented, the training pipeline applies **SMOTE only after the train/test split**:

```text
Original Dataset
      ↓
Train / Test Split
      ↓
Training Data → SMOTE → XGBoost
      ↓
Untouched Test Data → Evaluation
```

This prevents resampling from contaminating the test set.

---

## 📈 Model Performance

The model was evaluated on an untouched 30% test set.

| Metric | Result |
|---|---:|
| Accuracy | 99.9909% |
| Precision | 93.7787% |
| Recall | 99.6347% |
| F1 Score | 96.6155% |
| ROC-AUC | 99.9500% |

### Confusion Matrix

```text
                 Predicted
                 Normal   Fraud

Actual Normal    1906159    163
Actual Fraud           9  24551
```

> These are benchmark results on the synthetic PaySim dataset and should not be interpreted as equivalent performance on real-world banking transactions.

---

## 🔍 Explainable AI with SHAP

SHAP is used to explain individual predictions.

The application shows:

- Feature contribution values
- Features increasing fraud risk
- Features decreasing fraud risk
- Top feature impacts
- SHAP waterfall prediction breakdown

This allows the user to understand **why** the model produced a particular fraud prediction.

---

## 🖥️ Streamlit Dashboard

### Dashboard

The main dashboard provides:

- Loaded transaction count
- Fraud count in the loaded sample
- Fraud rate
- Model test ROC-AUC
- Sample transaction table
- Transaction selection
- Fraud probability
- Risk classification
- Actual vs predicted label
- Transaction details
- Balance analysis

### Prediction Breakdown

A dedicated page provides:

- Transaction selection
- Model prediction
- Fraud probability
- Actual dataset label
- Feature contribution table
- Top feature impact
- SHAP waterfall chart

Run the application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
fraud-detection/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── transactions.csv
│
├── models/
│   ├── xgb_model.joblib
│   ├── scaler.joblib
│   └── type_encoder.joblib
│
└── notebook/
    ├── fraud_detection.ipynb
    └── fraud_detection_fixed.ipynb
```

> Large datasets, virtual environments, and generated model artifacts can be excluded from GitHub using `.gitignore`.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fraud-detection
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the PaySim dataset at:

```text
data/transactions.csv
```

### 5. Train the model

Open:

```text
notebook/fraud_detection_fixed.ipynb
```

Run the notebook from top to bottom.

The notebook generates:

```text
models/xgb_model.joblib
models/scaler.joblib
models/type_encoder.joblib
```

### 6. Run the dashboard

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **imbalanced-learn / SMOTE**
- **SHAP**
- **Matplotlib**
- **Joblib**
- **Jupyter Notebook**
- **Streamlit**
- **Git & GitHub**

---

## 💡 Learning Outcomes

This project provided hands-on experience with:

- Imbalanced classification
- Data preprocessing
- Feature engineering
- Preventing data leakage
- SMOTE
- XGBoost
- Classification metrics
- ROC-AUC using prediction probabilities
- Explainable AI with SHAP
- Model serialization
- Streamlit application development
- ML model deployment into an interactive interface

---

## ⚠️ Limitations

- PaySim is a synthetic dataset.
- Synthetic benchmark performance does not guarantee real-world fraud detection performance.
- The dashboard uses a 10,000-row sample for interactive analysis.
- Production fraud detection would require threshold calibration, monitoring, drift detection, retraining, and validation on real-world data.

---

## 🔮 Future Improvements

- Real-time fraud scoring API
- Model monitoring
- Data drift detection
- Automated retraining
- Precision-recall analysis
- Threshold optimization
- Persistent transaction history
- Authentication and role-based access
- Docker deployment
- Cloud deployment
- CI/CD pipeline
- Real-time transaction streaming

---

## 👨‍💻 Author

**Sujal Maity**

Applied machine learning project focused on **fraud detection, explainable AI, and practical ML deployment**.
