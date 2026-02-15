import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Bank Marketing Predictor", layout="wide")

MODEL_PATH = "models/bank_marketing_rf.pkl"
FEATURE_PATH = "models/feature_columns.pkl"
DATA_PATH = "data/bank-additional-full.csv"

# --------------------------
# Train model if not exists
# --------------------------
def train_and_save_model():
    df = pd.read_csv(DATA_PATH, sep=";")
    df['y'] = df['y'].map({'no': 0, 'yes': 1})

    X = df.drop('y', axis=1)
    y = df['y']

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURE_PATH)

    return model, X.columns.tolist()

# --------------------------
# Load or Train Model
# --------------------------
if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
else:
    model, feature_columns = train_and_save_model()

# --------------------------
# Title
# --------------------------
st.title("📊 Bank Marketing Subscription Predictor")
st.write("Enter customer details to predict subscription outcome.")
st.divider()

# --------------------------
# Layout
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
# Prepare Input
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

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Customer is likely to SUBSCRIBE")
    else:
        st.error("❌ Customer is likely to NOT SUBSCRIBE")

    st.write(f"📈 Probability of Subscription: **{probability:.2%}**")
    st.progress(float(probability))

    st.divider()

    st.subheader("📊 Top Feature Importance")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    top_features = pd.DataFrame({
        "Feature": np.array(feature_columns)[indices],
        "Importance": importances[indices]
    })

    st.bar_chart(top_features.set_index("Feature"))

st.divider()
st.caption("Developed by Krisha Trivedi | Machine Learning Project")
