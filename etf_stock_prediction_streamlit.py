
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, confusion_matrix
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Data Loaders
# -------------------------------

def load_etf_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['price_date'] = pd.to_datetime(df['price_date'])
    df = df.drop_duplicates().dropna()
    df = df[(df['open'] != 0) & (df['close'] != 0)]
    df['lag_open'] = df.groupby('fund_symbol')['open'].shift(1)
    df['lag_close'] = df.groupby('fund_symbol')['close'].shift(1)
    df = df.dropna()
    return df

def load_stock_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns={'date': 'price_date', 'Name': 'fund_symbol'})
    df['price_date'] = pd.to_datetime(df['price_date'], errors='coerce')
    df = df.drop_duplicates().dropna()
    df = df[(df['open'] != 0) & (df['close'] != 0)]
    df['lag_open'] = df.groupby('fund_symbol')['open'].shift(1)
    df['lag_close'] = df.groupby('fund_symbol')['close'].shift(1)
    df = df.dropna()
    return df

# -------------------------------
# Core Modeling & Features
# -------------------------------

def train_per_etf_models(df):
    predictions = []
    for etf in df['fund_symbol'].unique():
        etf_data = df[df['fund_symbol'] == etf].copy()
        X = etf_data[['lag_open', 'lag_close']]
        y_open = etf_data['open']
        y_close = etf_data['close']
        if len(etf_data) < 10:
            continue
        X_train, X_test, y_train_open, y_test_open, y_train_close, y_test_close = train_test_split(
            X, y_open, y_close, test_size=0.2, shuffle=False)
        model_open = LinearRegression().fit(X_train, y_train_open)
        model_close = LinearRegression().fit(X_train, y_train_close)
        y_pred_open = model_open.predict(X_test)
        y_pred_close = model_close.predict(X_test)
        mae_open = mean_absolute_error(y_test_open, y_pred_open)
        r2_open = r2_score(y_test_open, y_pred_open)
        mae_close = mean_absolute_error(y_test_close, y_pred_close)
        r2_close = r2_score(y_test_close, y_pred_close)
        last_row = etf_data.tail(1)[['lag_open', 'lag_close']]
        pred_open = model_open.predict(last_row)[0]
        pred_close = model_close.predict(last_row)[0]
        predictions.append({
            'fund_symbol': etf,
            'pred_open': pred_open,
            'pred_close': pred_close,
            'mae_open': mae_open,
            'r2_open': r2_open,
            'mae_close': mae_close,
            'r2_close': r2_close
        })
    return pd.DataFrame(predictions)

def add_return_features(df):
    df['price_change'] = df['pred_close'] - df['pred_open']
    df['return_percentage'] = (df['price_change'] / df['pred_open']) * 100
    df['category'] = np.where(df['return_percentage'] < 0, 'Loss', None)
    non_negative = df[df['return_percentage'] >= 0]
    non_negative['category'] = pd.qcut(non_negative['return_percentage'], q=3, labels=['Low', 'Medium', 'High'])
    df.update(non_negative)
    return df

def train_classifier(df):
    X = df[['price_change', 'return_percentage']]
    y = df['category']
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    df['predicted_category'] = clf.predict(X)
    return clf, df, classification_report(y, df['predicted_category'], output_dict=True)

# -------------------------------
# Forecasting
# -------------------------------

def forecast_with_prophet(etf_df):
    prophet_df = etf_df[['price_date', 'close']].rename(columns={'price_date': 'ds', 'close': 'y'})
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    fig = m.plot(forecast)
    st.pyplot(fig)

def forecast_with_arima(etf_df):
    etf_df = etf_df.set_index('price_date').sort_index()
    series = etf_df['close']
    model = ARIMA(series, order=(1,1,1)).fit()
    forecast = model.forecast(steps=30)
    fig, ax = plt.subplots(figsize=(10, 4))
    series.plot(ax=ax, label='Historical')
    forecast.plot(ax=ax, label='ARIMA Forecast')
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# Export Utility
# -------------------------------

def get_table_download_link(df, filename="filtered_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'

# -------------------------------
# Streamlit Interface
# -------------------------------

st.set_page_config(layout="wide")
st.title("📈 ETF/Stock Return Classification & Forecasting")


# 📷 Quick preview of expected CSV format
from PIL import Image
image = Image.open("etl.JPG")
st.image(image, caption="CSV Format: etl", use_container_width=True, output_format="JPEG")
image1 = Image.open("stock.JPG")
st.image(image1, caption="CSV Format: stocks", use_container_width=True, output_format="JPEG")

file_type = st.radio("Select File Type", ["ETF CSV", "Stock CSV"])
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = load_etf_data(uploaded_file) if file_type == "ETF CSV" else load_stock_data(uploaded_file)
    st.success(f"{file_type} loaded successfully!")

    predictions_df = train_per_etf_models(df)
    predictions_df = add_return_features(predictions_df)
    clf, predictions_df, report_dict = train_classifier(predictions_df)

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report_dict).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(predictions_df['category'], predictions_df['predicted_category'], labels=clf.classes_)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_, ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.subheader("Return Distribution")
    fig, ax = plt.subplots()
    sns.histplot(predictions_df['return_percentage'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Price History Line Chart")
    selected_symbol = st.selectbox("Select Symbol to View Price History", df['fund_symbol'].unique())
    selected_df = df[df['fund_symbol'] == selected_symbol].sort_values(by='price_date')
    fig, ax = plt.subplots()
    ax.plot(selected_df['price_date'], selected_df['close'], label='Close Price')
    ax.set_title(f"Price History for {selected_symbol}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Export Filtered Predictions")
    st.markdown(get_table_download_link(predictions_df), unsafe_allow_html=True)

    st.subheader("Forecast (Prophet)")
    forecast_with_prophet(selected_df)

    st.subheader("Forecast (ARIMA)")
    forecast_with_arima(selected_df)
else:
    st.info("Upload a valid CSV file to begin.")
