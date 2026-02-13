# ======================================
# LIVE FINANCIAL FRAUD DETECTION DASHBOARD
# ======================================

import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Live Fraud Detection", layout="wide")

# ======================================
# LOAD BASE DATA
# ======================================

@st.cache_data
def load_base_data():
    df = pd.read_csv("credit_card_fraud_dataset.csv")
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    return df

df = load_base_data()

# ======================================
# LIVE DATA GENERATOR
# ======================================

def generate_live_transaction():
    return {
        "TransactionID": np.random.randint(1000000, 9999999),
        "TransactionDate": datetime.now(),
        "Amount": np.random.uniform(10, 5000),
        "MerchantID": np.random.choice(df["MerchantID"]),
        "TransactionType": np.random.choice(["Purchase", "Refund"]),
        "Location": np.random.choice(df["Location"]),
        "IsFraud": np.random.choice([0, 1], p=[0.97, 0.03])
    }

# ======================================
# DATABASE INIT
# ======================================

conn = sqlite3.connect("fraud_detection.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    TransactionID INTEGER,
    TransactionDate TEXT,
    Amount REAL,
    MerchantID TEXT,
    TransactionType TEXT,
    Location TEXT,
    IsFraud INTEGER,
    Prediction INTEGER
)
""")
conn.commit()

# ======================================
# TRAIN MODEL
# ======================================

df_ml = pd.get_dummies(df, columns=["TransactionType", "Location"], drop_first=True)
df_ml["Amount"] = RobustScaler().fit_transform(df_ml[["Amount"]])

X = df_ml.drop(["IsFraud", "TransactionDate"], axis=1)
y = df_ml["IsFraud"]

X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_res, y_res)

# ======================================
# STREAM LIVE DATA
# ======================================

if "live_data" not in st.session_state:
    st.session_state.live_data = []

new_tx = generate_live_transaction()
new_df = pd.DataFrame([new_tx])

# preprocess
proc = pd.get_dummies(new_df, columns=["TransactionType", "Location"], drop_first=True)
proc = proc.reindex(columns=X.columns, fill_value=0)
proc["Amount"] = RobustScaler().fit(df[["Amount"]]).transform(proc[["Amount"]])

prediction = model.predict(proc)[0]
new_tx["Prediction"] = prediction

st.session_state.live_data.append(new_tx)

pd.DataFrame([new_tx]).to_sql("transactions", conn, if_exists="append", index=False)

# ======================================
# LOAD LIVE DB
# ======================================

live_df = pd.read_sql("SELECT * FROM transactions", conn)

# ======================================
# AUTO REFRESH
# ======================================
time.sleep(2)
st.experimental_rerun()

# ======================================
# SIDEBAR
# ======================================

page = st.sidebar.radio(
    "📊 Live Dashboard Pages",
    [
        "Overview",
        "Fraud Monitor",
        "Amount Analysis",
        "Time Analysis",
        "Location Analysis",
        "Merchant Analysis",
        "Model Insights",
        "Live Transactions"
    ]
)

# ======================================
# PAGE 1: OVERVIEW
# ======================================

if page == "Overview":
    st.title("📊 Live System Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", len(live_df))
    c2.metric("Frauds Detected", live_df["Prediction"].sum())
    c3.metric("Live Fraud Rate (%)", round(live_df["Prediction"].mean() * 100, 2))

    st.bar_chart(live_df["Prediction"].value_counts())
    st.line_chart(live_df["Amount"])
    st.dataframe(live_df.tail(10))

# ======================================
# PAGE 2: FRAUD MONITOR
# ======================================

elif page == "Fraud Monitor":
    st.title("🚨 Live Fraud Alerts")

    st.bar_chart(live_df.groupby("Prediction")["Amount"].sum())
    st.dataframe(live_df[live_df["Prediction"] == 1].tail(10))

    fig, ax = plt.subplots()
    sns.countplot(x="Prediction", data=live_df, ax=ax)
    st.pyplot(fig)

# ======================================
# PAGE 3: AMOUNT ANALYSIS
# ======================================

elif page == "Amount Analysis":
    st.title("💰 Live Amount Patterns")

    st.line_chart(live_df["Amount"])
    st.bar_chart(live_df.groupby("Prediction")["Amount"].mean())

    fig, ax = plt.subplots()
    sns.histplot(live_df["Amount"], bins=40, ax=ax)
    st.pyplot(fig)

# ======================================
# PAGE 4: TIME ANALYSIS
# ======================================

elif page == "Time Analysis":
    st.title("⏰ Time-Based Fraud")

    live_df["hour"] = pd.to_datetime(live_df["TransactionDate"]).dt.hour
    st.bar_chart(live_df.groupby("hour")["Prediction"].sum())
    st.line_chart(live_df.groupby("hour")["Amount"].mean())

# ======================================
# PAGE 5: LOCATION ANALYSIS
# ======================================

elif page == "Location Analysis":
    st.title("📍 Location Risk")

    st.bar_chart(live_df.groupby("Location")["Prediction"].sum())
    st.bar_chart(live_df.groupby("Location")["Amount"].mean())

# ======================================
# PAGE 6: MERCHANT ANALYSIS
# ======================================

elif page == "Merchant Analysis":
    st.title("🏪 Merchant Risk")

    st.bar_chart(live_df.groupby("MerchantID")["Prediction"].sum())
    st.bar_chart(live_df.groupby("MerchantID")["Amount"].mean())

# ======================================
# PAGE 7: MODEL INSIGHTS
# ======================================

elif page == "Model Insights":
    st.title("📈 Model Insights")

    st.bar_chart(pd.Series(model.feature_importances_, index=X.columns).head(15))
    st.metric("Model Type", "Random Forest")
    st.metric("Training Samples", len(X_res))

# ======================================
# PAGE 8: LIVE TRANSACTIONS
# ======================================

elif page == "Live Transactions":
    st.title("🔴 Live Transaction Feed")

    st.dataframe(live_df.tail(50))
    st.metric("Live Records", len(live_df))
