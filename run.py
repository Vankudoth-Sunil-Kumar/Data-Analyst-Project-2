# ===============================
# Financial Fraud Detection System
# ===============================

# ----------- IMPORTS -----------
import pandas as pd
import numpy as np
import sqlite3
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE

# ----------- LOAD DATA -----------
DATA_PATH = "credit_card_fraud_dataset.csv"

df = pd.read_csv(DATA_PATH)

# ----------- PREPROCESSING -----------

# Convert date
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

# Feature engineering
df['day'] = df['TransactionDate'].dt.day
df['month'] = df['TransactionDate'].dt.month

# Encode categorical columns
df = pd.get_dummies(df, columns=['TransactionType', 'Location'], drop_first=True)

# Scale amount
scaler = RobustScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Separate features & target
X = df.drop(['IsFraud', 'TransactionDate'], axis=1)
y = df['IsFraud']

# ----------- HANDLE IMBALANCE -----------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ----------- TRAIN TEST SPLIT -----------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

# ----------- MODEL TRAINING -----------
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ----------- EVALUATION -----------
y_pred = model.predict(X_test)
roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print("\nMODEL PERFORMANCE\n")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc)

# ----------- STORE RESULTS IN DATABASE -----------
conn = sqlite3.connect("fraud_detection.db")

df['Prediction'] = model.predict(X)

df.to_sql("transactions", conn, if_exists="replace", index=False)
conn.close()

# ----------- STREAMLIT DASHBOARD -----------

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Financial Fraud Detection Dashboard")

conn = sqlite3.connect("fraud_detection.db")
data = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()

# ----------- METRICS -----------
col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", len(data))
col2.metric("Fraud Transactions", int(data['IsFraud'].sum()))
col3.metric("Fraud Rate (%)", round(data['IsFraud'].mean() * 100, 2))

# ----------- VISUALIZATIONS -----------

st.subheader("📊 Fraud vs Non-Fraud")
st.bar_chart(data['IsFraud'].value_counts())

st.subheader("💰 Transaction Amount Distribution")
st.line_chart(data['Amount'].sample(1000))

st.subheader("🚨 Recent Fraudulent Transactions")
frauds = data[data['IsFraud'] == 1]
st.dataframe(frauds.tail(10))

st.subheader("🧾 Recent Transactions")
st.dataframe(data.tail(10))
