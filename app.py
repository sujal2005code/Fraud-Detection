import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.title("🛡️ Fraud Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Prediction Breakdown"
    ]
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/transactions.csv",
        nrows=10000
    )

    # Feature engineering
    df["errorBalanceOrig"] = (
        df["newbalanceOrig"]
        + df["amount"]
        - df["oldbalanceOrg"]
    )

    df["errorBalanceDest"] = (
        df["oldbalanceDest"]
        + df["amount"]
        - df["newbalanceDest"]
    )

    return df


# ---------------------------------------------------------
# LOAD MODEL ASSETS
# ---------------------------------------------------------

@st.cache_resource
def load_assets():

    model = joblib.load(
        "models/xgb_model.joblib"
    )

    scaler = joblib.load(
        "models/scaler.joblib"
    )

    encoder = joblib.load(
        "models/type_encoder.joblib"
    )

    return model, scaler, encoder


# ---------------------------------------------------------
# INITIALIZE
# ---------------------------------------------------------

df = load_data()

model, scaler, encoder = load_assets()

# ---------------------------------------------------------
# DASHBOARD PAGE
# ---------------------------------------------------------

if page == "Dashboard":

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.title("🛡️ Fraud Detection System")

    st.caption(
        "XGBoost-powered transaction risk analysis "
        "with SHAP explainability"
    )

    st.divider()

# ---------------------------------------------------------
# DASHBOARD METRICS
# ---------------------------------------------------------

total_transactions = len(df)

fraud_transactions = int(
    df["isFraud"].sum()
)

fraud_rate = (
    fraud_transactions
    / total_transactions
) * 100


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )


with col2:

    st.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )


with col3:

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.3f}%"
    )


with col4:

    st.metric(
        "Model ROC-AUC",
        "99.95%"
    )


st.divider()


# ---------------------------------------------------------
# SAMPLE TRANSACTIONS
# ---------------------------------------------------------

st.subheader("📋 Sample Transactions")

st.dataframe(
    df.head(20),
    use_container_width=True
)


st.divider()


# ---------------------------------------------------------
# TRANSACTION PREDICTION
# ---------------------------------------------------------

st.subheader("🔍 Transaction Risk Assessment")

index = st.number_input(
    "Choose a sample transaction",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)


# ---------------------------------------------------------
# PREPARE SAMPLE
# ---------------------------------------------------------

sample = df.drop(
    [
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "nameDest"
    ],
    axis=1
).iloc[[index]].copy()


# Encode transaction type

sample["type"] = encoder.transform(
    sample["type"]
)


# IMPORTANT:
# The XGBoost model was trained using
# unscaled features.

sample_input = sample


# ---------------------------------------------------------
# MODEL PREDICTION
# ---------------------------------------------------------

pred = model.predict(
    sample_input
)[0]

proba = model.predict_proba(
    sample_input
)[0][1]


# ---------------------------------------------------------
# ACTUAL LABEL
# ---------------------------------------------------------

actual = int(
    df.iloc[index]["isFraud"]
)


# ---------------------------------------------------------
# RISK CLASSIFICATION
# ---------------------------------------------------------

if proba >= 0.80:

    risk_level = "HIGH RISK"
    risk_icon = "🚨"

elif proba >= 0.30:

    risk_level = "MEDIUM RISK"
    risk_icon = "⚠️"

else:

    risk_level = "LOW RISK"
    risk_icon = "🟢"


# ---------------------------------------------------------
# RISK ASSESSMENT PANEL
# ---------------------------------------------------------

st.markdown("### 🚨 Risk Assessment")


risk_col1, risk_col2, risk_col3 = st.columns(3)


# ---------------------------------------------------------
# PREDICTION STATUS
# ---------------------------------------------------------

with risk_col1:

    st.metric(
        "Model Prediction",
        "FRAUD" if pred == 1 else "NOT FRAUD"
    )


# ---------------------------------------------------------
# FRAUD PROBABILITY
# ---------------------------------------------------------

with risk_col2:

    st.metric(
        "Fraud Probability",
        f"{proba * 100:.4f}%"
    )


# ---------------------------------------------------------
# RISK LEVEL
# ---------------------------------------------------------

with risk_col3:

    st.metric(
        "Risk Level",
        f"{risk_icon} {risk_level}"
    )


# ---------------------------------------------------------
# LARGE RISK STATUS
# ---------------------------------------------------------

st.markdown("### Transaction Status")


if pred == 1:

    st.error(
        "🚨 FRAUD DETECTED\n\n"
        f"This transaction has a fraud probability of "
        f"{proba * 100:.4f}%."
    )

else:

    st.success(
        "✅ TRANSACTION APPEARS LEGITIMATE\n\n"
        f"This transaction has a fraud probability of "
        f"{proba * 100:.4f}%."
    )


# ---------------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------------

comparison_col1, comparison_col2 = st.columns(2)


with comparison_col1:

    st.write("**Actual Dataset Label**")

    if actual == 1:

        st.error("🚨 Fraud")

    else:

        st.success("✅ Not Fraud")


with comparison_col2:

    st.write("**Model Prediction**")

    if pred == 1:

        st.error("🚨 Fraud")

    else:

        st.success("✅ Not Fraud")


# ---------------------------------------------------------
# PREDICTION VALIDATION
# ---------------------------------------------------------

if pred == actual:

    st.success(
        "✅ Model prediction matches the dataset label."
    )

else:

    st.warning(
        "⚠️ Model prediction differs from the dataset label."
    )


st.divider()


# ---------------------------------------------------------
# TRANSACTION DETAILS
# ---------------------------------------------------------

st.subheader("💳 Transaction Details")

transaction = df.iloc[index]

detail_col1, detail_col2 = st.columns(2)


with detail_col1:

    st.markdown("#### 💰 Transaction Information")

    st.write(
        f"**Transaction Type:** `{transaction['type']}`"
    )

    st.write(
        f"**Transaction Amount:** "
        f"₹{transaction['amount']:,.2f}"
    )

    st.write(
        f"**Time Step:** `{int(transaction['step'])}`"
    )


with detail_col2:

    st.markdown("#### 🏦 Account Information")

    st.write(
        f"**Origin Balance Before:** "
        f"₹{transaction['oldbalanceOrg']:,.2f}"
    )

    st.write(
        f"**Origin Balance After:** "
        f"₹{transaction['newbalanceOrig']:,.2f}"
    )

    st.write(
        f"**Destination Balance Before:** "
        f"₹{transaction['oldbalanceDest']:,.2f}"
    )

    st.write(
        f"**Destination Balance After:** "
        f"₹{transaction['newbalanceDest']:,.2f}"
    )


# ---------------------------------------------------------
# BALANCE ANALYSIS
# ---------------------------------------------------------

st.markdown("#### 🧮 Balance Analysis")

balance_col1, balance_col2 = st.columns(2)


with balance_col1:

    st.write(
        f"**Origin Balance Error:** "
        f"`{transaction['errorBalanceOrig']:,.2f}`"
    )

    st.write(
        f"**Destination Balance Error:** "
        f"`{transaction['errorBalanceDest']:,.2f}`"
    )


with balance_col2:

    origin_change = (
        transaction["oldbalanceOrg"]
        - transaction["newbalanceOrig"]
    )

    destination_change = (
        transaction["newbalanceDest"]
        - transaction["oldbalanceDest"]
    )

    st.write(
        f"**Origin Balance Change:** "
        f"₹{origin_change:,.2f}"
    )

    st.write(
        f"**Destination Balance Change:** "
        f"₹{destination_change:,.2f}"
    )


st.divider()


# ---------------------------------------------------------
# SHAP EXPLANATION
# ---------------------------------------------------------

st.subheader("🧠 SHAP Explainability")

st.caption(
    "SHAP explains which transaction features "
    "increased or decreased the model's fraud prediction."
)


# Create SHAP explainer
explainer = shap.Explainer(model)


# Calculate SHAP values
shap_values = explainer(sample_input)


# ---------------------------------------------------------
# FEATURE CONTRIBUTIONS
# ---------------------------------------------------------

st.markdown("#### 📊 Feature Contributions")

shap_array = shap_values[0].values

feature_names = sample_input.columns

contributions = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": shap_array
})


# Sort by absolute contribution
contributions["Absolute Contribution"] = (
    contributions["Contribution"].abs()
)

contributions = contributions.sort_values(
    "Absolute Contribution",
    ascending=False
)


# Display the most important features
display_contributions = contributions[
    ["Feature", "Contribution"]
].copy()


display_contributions["Impact"] = (
    display_contributions["Contribution"]
    .apply(
        lambda x:
        "🔴 Increases Fraud Risk"
        if x > 0
        else "🟢 Decreases Fraud Risk"
    )
)


display_contributions["Contribution"] = (
    display_contributions["Contribution"]
    .round(6)
)


st.dataframe(
    display_contributions,
    use_container_width=True,
    hide_index=True
)


st.markdown("#### 💡 Top Feature Impact")

top_features = contributions.head(5)


for _, row in top_features.iterrows():

    feature = row["Feature"]
    contribution = row["Contribution"]

    if contribution > 0:

        st.write(
            f"🔴 **{feature}** "
            f"increased the fraud prediction "
            f"by `{contribution:.6f}`"
        )

    else:

        st.write(
            f"🟢 **{feature}** "
            f"decreased the fraud prediction "
            f"by `{abs(contribution):.6f}`"
        )


st.divider()

# ---------------------------------------------------------
# PREDICTION BREAKDOWN PAGE
# ---------------------------------------------------------

if page == "Prediction Breakdown":

    st.title("🧠 Prediction Breakdown")

    st.caption(
        "Detailed SHAP analysis showing how each feature "
        "influenced the selected transaction."
    )

    st.divider()

    # -----------------------------------------------------
    # TRANSACTION SELECTION
    # -----------------------------------------------------

    index = st.number_input(
        "Choose a transaction",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    # -----------------------------------------------------
    # PREPARE SAMPLE
    # -----------------------------------------------------

    sample = df.drop(
        [
            "isFraud",
            "isFlaggedFraud",
            "nameOrig",
            "nameDest"
        ],
        axis=1
    ).iloc[[index]].copy()

    sample["type"] = encoder.transform(
        sample["type"]
    )

    sample_input = sample

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    pred = model.predict(
        sample_input
    )[0]

    proba = model.predict_proba(
        sample_input
    )[0][1]

    actual = int(
        df.iloc[index]["isFraud"]
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prediction",
            "FRAUD" if pred == 1 else "NOT FRAUD"
        )

    with col2:

        st.metric(
            "Fraud Probability",
            f"{proba * 100:.4f}%"
        )

    with col3:

        st.metric(
            "Actual Label",
            "FRAUD" if actual == 1 else "NOT FRAUD"
        )

    st.divider()

    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    explainer = shap.Explainer(model)

    shap_values = explainer(
        sample_input
    )

    # -----------------------------------------------------
    # FEATURE CONTRIBUTIONS
    # -----------------------------------------------------

    st.subheader("📊 Feature Contributions")

    shap_array = shap_values[0].values

    contributions = pd.DataFrame({
        "Feature": sample_input.columns,
        "Contribution": shap_array
    })

    contributions["Absolute Contribution"] = (
        contributions["Contribution"].abs()
    )

    contributions = contributions.sort_values(
        "Absolute Contribution",
        ascending=False
    )

    display_contributions = contributions[
        ["Feature", "Contribution"]
    ].copy()

    display_contributions["Impact"] = (
        display_contributions["Contribution"]
        .apply(
            lambda x:
            "🔴 Increases Fraud Risk"
            if x > 0
            else "🟢 Decreases Fraud Risk"
        )
    )

    display_contributions["Contribution"] = (
        display_contributions["Contribution"]
        .round(6)
    )

    st.dataframe(
        display_contributions,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # TOP FEATURE IMPACT
    # -----------------------------------------------------

    st.subheader("💡 Top Feature Impact")

    for _, row in contributions.head(5).iterrows():

        feature = row["Feature"]
        contribution = row["Contribution"]

        if contribution > 0:

            st.write(
                f"🔴 **{feature}** increased "
                f"the fraud prediction by "
                f"`{contribution:.6f}`"
            )

        else:

            st.write(
                f"🟢 **{feature}** decreased "
                f"the fraud prediction by "
                f"`{abs(contribution):.6f}`"
            )

    st.divider()

    # -----------------------------------------------------
    # PREDICTION BREAKDOWN GRAPH
    # -----------------------------------------------------

    st.subheader("🌊 Prediction Breakdown")

    st.caption(
        "The waterfall chart shows how each feature "
        "moves the prediction from the model baseline."
    )

    fig, ax = plt.subplots()

    shap.plots.waterfall(
        shap_values[0],
        show=False
    )

    st.pyplot(
        fig,
        clear_figure=True
    )