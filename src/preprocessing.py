import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/bank-additional-full.csv", sep=";")

# Convert target variable to binary
df['y'] = df['y'].map({'no': 0, 'yes': 1})

print("Target distribution:")
print(df['y'].value_counts())

# Separate features and target
X = df.drop('y', axis=1)
y = df['y']

# Convert categorical columns using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

print("\nShape after encoding:")
print(X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)
