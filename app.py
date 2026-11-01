import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# Page Configuration
st.set_page_config(page_title="Stock Trend Prediction",layout="wide")
st.title("Stock Price Trend Prediction")
st.write("Machine Learning Based Stock Prediction System")
# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/aapl.us.txt")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df
try:
    df = load_data()
    # Feature Engineering
    df["Return_1d"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Volatility_5"] = df["Return_1d"].rolling(5).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["High_Low_Range"] = ((df["High"] - df["Low"]) / df["Close"])
    df["Close_Open_Change"] = ((df["Close"] - df["Open"]) / df["Open"])
    # Target
    df["Next_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)
    features = [
        "Open", "High", "Low", "Close", "Volume","Return_1d", "MA_5", "MA_10", "MA_20",
        "Volatility_5", "Volume_Change","Return_5d", "Return_10d","High_Low_Range", "Close_Open_Change"]
    # Clean Data
    df = df.replace([np.inf, -np.inf], np.nan)
    model_data = df.dropna(subset=features + ["Next_Close"]).copy()
    X = model_data[features]
    y = model_data["Target"]
    # Chronological Split
    split = int(len(model_data) * 0.8)
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    # XGBoost Model
    model = XGBClassifier(
        n_estimators=300,max_depth=3,learning_rate=0.03,subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5,eval_metric="logloss",random_state=42)
    # Training
    model.fit(X_train, y_train)
    # Accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    baseline = max(y_test.mean(), 1 - y_test.mean()) * 100
    st.subheader("XGBoost Model Results")
    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy", f"{accuracy:.2f}%")
    col2.metric("Baseline Accuracy", f"{baseline:.2f}%")
    st.caption("Accuracy is measured on later historical data.")
    # Historical Stock Chart
    st.subheader("Historical Stock Prices")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    ax.set_title("Apple Stock Price")
    st.pyplot(fig)
    plt.close(fig)
    # Next Day Prediction
    st.subheader("Next Day Trend Prediction")
    latest = df.dropna(subset=features).iloc[[-1]]
    # Train final model on all available labeled data
    final_model = XGBClassifier(
        n_estimators=300,max_depth=3,learning_rate=0.03,subsample=0.8,
        colsample_bytree=0.8,reg_lambda=5,eval_metric="logloss",random_state=42)
    final_model.fit(X, y)
    result = final_model.predict(latest[features])[0]
    confidence = (final_model.predict_proba(latest[features])[0].max()* 100)
    if result == 1:
        st.success("Predicted Trend: UP")
    else:
        st.warning("Predicted Trend: DOWN")
    st.write(f"Model Confidence: {confidence:.2f}%")
    st.caption("Prediction is an estimate, not a guarantee.")
    # Recent Stock Data
    st.subheader("Recent Stock Data")
    st.dataframe(
        df[["Date", "Open", "High", "Low","Close", "Volume"]].tail(10),
        use_container_width=True)
except FileNotFoundError:
    st.error("Dataset not found. Check data/aapl.us.txt")
except Exception as e:
    st.error(f"Something went wrong: {e}")