🏦💳 LIVE FINANCIAL FRAUD DETECTION SYSTEM
Real-Time AI-Powered Fraud Monitoring Dashboard
🏛️ Institution Logo

(Insert your College/University Logo here)

🧠 Technology Logos

(Insert Python, Streamlit, Scikit-learn, and SQLite logos here)

📄 PROJECT REPORT
1️⃣ Abstract

The Live Financial Fraud Detection System is an AI-powered real-time fraud monitoring dashboard developed using Machine Learning and Streamlit.

The system detects suspicious credit card transactions using a trained Random Forest model and visualizes fraud trends dynamically. It integrates:

Machine Learning (Random Forest)

Data Preprocessing & SMOTE (Handling Imbalanced Data)

SQLite Database

Real-time Transaction Simulation

Interactive Streamlit Dashboard

This project demonstrates practical implementation of fraud detection used in banking and financial institutions.

2️⃣ Introduction

Financial fraud is a major challenge in digital banking systems. With the rapid increase in online transactions, detecting fraudulent activities in real time is critical.

This project simulates a live transaction system where:

New transactions are generated continuously.

A trained ML model predicts fraud instantly.

Results are stored in a database.

A live dashboard displays fraud analytics.

3️⃣ Objectives

The main objectives of this project are:

Develop a fraud detection ML model.

Handle class imbalance using SMOTE.

Create a real-time data streaming simulation.

Build an interactive dashboard.

Store predictions in a database.

Visualize fraud risk patterns.

4️⃣ System Architecture
🔷 High-Level Architecture

User → Live Transaction Generator → ML Model → SQLite Database → Streamlit Dashboard

🔷 Components

Data Preprocessing Module

Date conversion

Feature engineering

One-hot encoding

Robust scaling

Machine Learning Module

Random Forest Classifier

SMOTE for class balancing

ROC-AUC evaluation

Database Layer

SQLite

Stores transactions and predictions

Visualization Layer

Streamlit

Matplotlib

Seaborn

5️⃣ Technologies Used
Technology	Purpose
🐍 Python	Core programming
📊 Pandas & NumPy	Data handling
🤖 Scikit-learn	Machine Learning
⚖ SMOTE (Imbalanced-learn)	Handling imbalanced dataset
🗄 SQLite	Database storage
📈 Matplotlib & Seaborn	Visualization
🌐 Streamlit	Interactive dashboard
6️⃣ Dataset Description

The dataset contains historical credit card transactions with the following features:

TransactionID

TransactionDate

Amount

MerchantID

TransactionType

Location

IsFraud (Target variable)

Fraud cases are highly imbalanced (~3%), making SMOTE necessary.

7️⃣ Machine Learning Process
Step 1: Data Cleaning

Convert date columns

Remove irrelevant features

Step 2: Feature Engineering

Extract day and month

One-hot encoding for categorical features

Robust scaling for amount normalization

Step 3: Handling Imbalance

SMOTE was applied to oversample fraud cases.

Step 4: Model Training

Random Forest Classifier:

150 trees

Parallel processing enabled

Random state = 42

Step 5: Evaluation

Classification Report

ROC-AUC Score

8️⃣ Live Simulation

The system generates synthetic live transactions using:

generate_live_transaction()


Each new transaction:

Is processed

Predicted by the ML model

Stored in SQLite

Displayed on dashboard

9️⃣ Dashboard Pages
📊 Overview

Total transactions

Fraud detected

Fraud rate

Real-time charts

🚨 Fraud Monitor

Fraud alerts

Fraud distribution

💰 Amount Analysis

Amount trends

Fraud vs Non-fraud comparison

⏰ Time Analysis

Fraud per hour

📍 Location Analysis

High-risk locations

🏪 Merchant Analysis

Merchant risk insights

📈 Model Insights

Feature importance

Training sample count

🔴 Live Feed

Latest transactions
