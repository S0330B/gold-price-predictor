import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import requests

# Load trained model
model = joblib.load('models/GoldPricePredictor.joblib')

# Load dataset
df = pd.read_csv('data/raw/gold_price_data.csv')

# Page config
st.set_page_config(page_title="Gold Price Predictor", layout="wide")
st.title("💰 Gold Price Predictor")

# Sidebar for feature explanations
st.sidebar.header("Feature Explanation")
st.sidebar.markdown("""
**SPX:** S&P 500 Index – Represents U.S. stock market performance.  
**USO:** United States Oil Fund – Tracks crude oil prices.  
**SLV:** iShares Silver Trust – Tracks silver prices.  
**EUR/USD:** Euro to U.S. Dollar exchange rate – Currency exchange rate impacting gold prices.
""")

# User input
st.subheader("Enter Feature Values")
spx = st.number_input("SPX", value=float(df['SPX'].mean()))
uso = st.number_input("USO", value=float(df['USO'].mean()))
slv = st.number_input("SLV", value=float(df['SLV'].mean()))
eur_usd = st.number_input("EUR/USD", value=float(df['EUR/USD'].mean()))

# Predict
if st.button("Predict Gold Price"):
    input_df = pd.DataFrame([[spx, uso, slv, eur_usd]],
                            columns=['SPX','USO','SLV','EUR/USD'])
    prediction = model.predict(input_df)
    st.write(f"💵 Predicted Gold Price: ${prediction[0]:.2f}")

# Convert the predicted amount to NPR
if st.checkbox("Convert to NPR"):
    input_df = pd.DataFrame([[spx, uso, slv, eur_usd]],
                            columns=['SPX','USO','SLV','EUR/USD'])
    prediction = model.predict(input_df)
    def convert_usd_to_npr(prediction):
        app_id = "3f6dfe0891594de3803b522f7dc1db58"
        url = f"https://openexchangerates.org/api/latest.json?app_id={app_id}&base=USD"
        response = requests.get(url)
        data = response.json()
        npr_rate = data['rates']['NPR']
        return prediction * npr_rate

    npr_amount = convert_usd_to_npr(prediction)
    st.success(f"${prediction[0]:.2f} = Rs. {npr_amount[0]:,.2f}")

# Predicted vs Actual chart
if st.checkbox("Show Predicted vs Actual Gold Prices"):
    from src.model_training import train_model
    X = df.drop(columns=['Date','GLD'])
    y = df['GLD']
    _, X_test, y_test, predictions = train_model(X, y, save_path='models/GoldPricePredictor.joblib')
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(list(y_test), color='blue', label='Actual')
    ax.plot(predictions, color='green', label='Predicted')
    ax.set_title("Actual vs Predicted Gold Prices")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Gold Price")
    ax.legend()
    st.pyplot(fig)

# Historical gold price trend
if st.checkbox("Show Historical Gold Price Trend"):
    fig2, ax2 = plt.subplots(figsize=(12,12))
    ax2.plot(pd.to_datetime(df['Date']), df['GLD'], color='gold')
    ax2.set_title("Historical Gold Price Trend")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Gold Price")
    st.pyplot(fig2)
