import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="📊",
    layout="wide"
)

# --------------------------
# Load Model and Columns
# --------------------------
model = joblib.load("models/bank_marketing_rf.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# --------------------------
# Header Section
# --------------------------
st.title("📊 Bank Marketing Subscription Predictor")
st.markdown(
    "This ML-powered application predicts whether a customer will subscribe "
    "to a bank term deposit based on campaign data."
)

st.divider()

# --------------------------
# Layout (2 Columns)
# --------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 95, 30)
    duration = st.number_input("Call Duration (seconds)", 0, 5000, 200)
    campaign = st.number_input("Number of Contacts During Campaign", 1, 50, 1)
    pdays = st.number_input("Days Since Last Contact", -1, 1000, 999)

with col2:
    euribor3m = st.number_input("Euribor 3 Month Rate", 0.0, 10.0, 4.0)
    emp_var_rate = st.number_input("Employment Variation Rate", -5.0, 5.0, 1.0)
    nr_employed = st.number_input("Number of Employees", 4000.0, 6000.0, 5191.0)

st.divider()

# --------------------------
# Create Input DataFrame
# --------------------------
input_dict = {
    "age": age,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "euribor3m": euribor3m,
    "emp.var.rate": emp_var_rate,
    "nr.employed": nr_employed
}

input_df = pd.DataFrame([input_dict])

# Add missing columns (important for model compatibility)
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]

# --------------------------
# Prediction
# --------------------------
if st.button("🔮 Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.success("✅ Customer is likely to SUBSCRIBE")
    else:
        st.error("❌ Customer is likely to NOT SUBSCRIBE")

    st.markdown(f"### 📈 Subscription Probability: **{probability:.2%}**")

    # Progress bar
    st.progress(float(probability))

    # Risk Interpretation
    if probability > 0.75:
        st.info("🔥 High likelihood of subscription.")
    elif probability > 0.40:
        st.warning("⚠️ Moderate likelihood of subscription.")
    else:
        st.write("Low likelihood of subscription.")

    st.divider()

    # --------------------------
    # Feature Importance
    # --------------------------
    st.subheader("📊 Top 10 Important Features (Random Forest)")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    top_features = pd.DataFrame({
        "Feature": np.array(feature_columns)[indices],
        "Importance": importances[indices]
    })

    st.bar_chart(top_features.set_index("Feature"))

    st.divider()

# --------------------------
# Model Info Section
# --------------------------
st.markdown("### 🤖 Model Information")
st.write("• Algorithm Used: Random Forest Classifier")
st.write("• Training Accuracy: ~91%")
st.write("• Handles Imbalanced Data using class_weight='balanced'")
st.write("• Features engineered using one-hot encoding")

st.divider()

st.caption("Developed by Krisha Trivedi | Machine Learning Project")
