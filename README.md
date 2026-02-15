# 📊 Bank Marketing Subscription Predictor

A Machine Learning web application that predicts whether a customer will subscribe to a bank term deposit using Logistic Regression and Random Forest models.

🚀 Live App: https://krisha575-bank-marketing-subscription-predictor-srcapp-ufjgd.streamlit.app

---

## 📌 Project Overview

This project analyzes the Bank Marketing Dataset and builds classification models to predict customer subscription behavior.

The deployed Streamlit app allows users to:
- Enter customer details
- View subscription prediction
- See probability score
- Visualize top feature importance

---

## 🧠 Machine Learning Models Used

- Logistic Regression
- Random Forest (Final Model - ~91% Accuracy)

---

## ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Git & GitHub

---

## 📊 Key Features

- Handles imbalanced dataset
- Feature engineering with one-hot encoding
- Model comparison (Logistic vs Random Forest)
- Interactive UI with probability score
- Deployed on Streamlit Cloud

---

## 📂 Project Structure

```
bank-marketing-subscription-predictor/
│
├── data/
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── app.py
│
├── requirements.txt
└── .gitignore
```

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

---

## 👩‍💻 Author

Krisha Trivedi  
Machine Learning Enthusiast  
