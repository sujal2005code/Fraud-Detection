import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection App", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('data/transactions.csv', nrows=10000)
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    return df

@st.cache_resource
def load_assets():
    model = joblib.load('models/xgb_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    encoder = joblib.load('models/type_encoder.joblib')
    return model, scaler, encoder

df = load_data()
model, scaler, encoder = load_assets()

st.title("Fraud Detection Dashboard")

st.subheader("Sample Transactions")
st.dataframe(df.head(20))

st.subheader("Predict a Sample Transaction")
index = st.slider("Choose a sample index", 0, len(df)-1, 0)
sample = df.drop(['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1).iloc[[index]]
sample['type'] = encoder.transform(sample['type'])
sample_scaled = scaler.transform(sample)

pred = model.predict(sample_scaled)[0]
proba = model.predict_proba(sample_scaled)[0][1]

st.write(f"Prediction: **{'Fraud' if pred == 1 else 'Not Fraud'}**")
st.write(f"Probability of Fraud: **{proba:.2f}**")

st.subheader("SHAP Explanation")
explainer = shap.Explainer(model)
shap_values = explainer(sample_scaled)
plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(plt.gcf(), clear_figure=True)