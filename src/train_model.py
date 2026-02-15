import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("data/bank-additional-full.csv", sep=";")

# Convert target to binary
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# Separate features and target
X = df.drop('y', axis=1)
y = df['y']

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Save feature column names
feature_columns = X.columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==================================
# LOGISTIC REGRESSION
# ==================================
print("\n===== LOGISTIC REGRESSION =====")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=3000, class_weight='balanced')
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, log_pred))
print("\nClassification Report:\n", classification_report(y_test, log_pred))


# ==================================
# RANDOM FOREST
# ==================================
print("\n===== RANDOM FOREST =====")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("\nClassification Report:\n", classification_report(y_test, rf_pred))


# ==================================
# SAVE MODELS
# ==================================

os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/bank_marketing_rf.pkl")
joblib.dump(log_model, "models/bank_marketing_log.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")

print("\nModels and feature columns saved successfully!")
