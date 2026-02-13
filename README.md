# Data-Analyst-Project-2

1) Project Summary

This project is a Live Financial Fraud Detection Dashboard that:

trains a fraud classifier using historical transactions (credit_card_fraud_dataset.csv)

generates synthetic live transactions continuously

predicts fraud in real-time

stores transactions + predictions in SQLite (fraud_detection.db)

visualizes trends, alerts, and model insights in a Streamlit dashboard

2) Key Features

ML Model: RandomForestClassifier

Imbalance handling: SMOTE oversampling

Preprocessing:

One-hot encoding (TransactionType, Location)

Robust scaling for Amount

Optional time-based features (day, month, hour)

Live Stream Simulation:

generates a new transaction periodically

predicts fraud probability/class

appends record to SQLite

Dashboard Pages:

Overview (metrics + charts)

Fraud Monitor (alerts)

Amount / Time / Location / Merchant analysis

Model feature importance

Live transactions feed

3) Tech Stack

Python

Streamlit (UI)

Pandas / NumPy (data)

Scikit-learn (ML)

Imbalanced-learn (SMOTE)

SQLite (storage)

Seaborn / Matplotlib (plots)
