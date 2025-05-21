import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go
from datetime import datetime, timedelta
import os

# Streamlit page configuration
st.set_page_config(page_title="AI-Enhanced Stock Forecasting Dashboard", layout="wide")

# Function to fetch stock data
def fetch_stock_data(symbol):
    try:
        # Append .NS for Indian stocks if not already present
        if symbol.upper() in ['INFY', 'TCS', 'RELIANCE'] and not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Function to perform ML forecasting
def forecast_stock_price(df):
    df['Date'] = pd.to_datetime(df.index)
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']].values
    y = df['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    # Forecast next 7 days
    last_day = df['Days'].iloc[-1]
    future_days = np.array([[last_day + i] for i in range(1, 8)])
    forecast_prices = model.predict(future_days)

    # Calculate MAE
    train_pred = model.predict(X)
    mae = mean_absolute_error(y, train_pred)

    # Determine trend
    trend = "Uptrend" if forecast_prices[-1] > y[-1] else "Downtrend"

    return forecast_prices, mae, trend

# Function to analyze PDF
def analyze_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Extract company name (assuming it's mentioned early in the document)
        company_name = text.split('\n')[0] if text else "Unknown Company"

        # Sentiment analysis using TextBlob
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

        # Generate summary (simple extraction of first few sentences)
        sentences = blob.sentences[:3]
        summary = " ".join(str(sentence) for sentence in sentences)

        # Trend bias based on sentiment
        trend_bias = "Likely Positive" if sentiment_score > 0 else "Likely Negative" if sentiment_score < 0 else "Neutral"

        return company_name, sentiment, summary, trend_bias
    except Exception as e:
        st.error(f"Error analyzing PDF: {e}")
        return None, None, None, None

# Function to generate final recommendation
def generate_recommendation(ml_trend, ai_trend_bias):
    if ml_trend == "Uptrend" and ai_trend_bias == "Likely Positive":
        recommendation = "BUY"
        explanation = "Positive market outlook combined with an uptrend forecast."
    elif ml_trend == "Downtrend" and ai_trend_bias == "Likely Negative":
        recommendation = "SELL"
        explanation = "Negative market outlook combined with a downtrend forecast."
    else:
        recommendation = "HOLD"
        explanation = "Mixed signals from market outlook and price forecast."
    return recommendation, explanation

# Dashboard UI
st.title("AI-Enhanced Stock Forecasting Dashboard")

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., INFY, TCS, RELIANCE)", value="INFY")
with col2:
    uploaded_file = st.file_uploader("Upload PDF Report (Optional)", type=["pdf"])

# Sample PDF mapping (for demo purposes)
pdf_mapping = {
    "INFY": "infosys_report.pdf",
    "TCS": "tcs_report.pdf",
    "RELIANCE": "reliance_report.pdf"
}

if st.button("Analyze"):
    # Fetch stock data
    stock_data = fetch_stock_data(stock_symbol)
    if stock_data is not None:
        # Section 1: ML Prediction
        st.header("Section 1: ML Prediction")
        forecast_prices, mae, trend = forecast_stock_price(stock_data)

        # Create forecast plot
        last_date = stock_data.index[-1]
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index[-30:], y=stock_data['Close'][-30:], name="Historical"))
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices, name="Forecast", line=dict(dash='dash')))
        fig.update_layout(title=f"{stock_symbol} Stock Price Forecast", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Trend Direction:** {trend}")

        # Section 2: AI Prediction
        st.header("Section 2: AI Prediction")
        pdf_path = uploaded_file if uploaded_file else pdf_mapping.get(stock_symbol.split('.')[0].upper(), None)
        if pdf_path:
            if uploaded_file:
                # Save uploaded file temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_path = "temp.pdf"

            company_name, sentiment, summary, trend_bias = analyze_pdf(pdf_path)
            if company_name:
                st.write(f"**Company Name:** {company_name}")
                st.write(f"**Sentiment:** {sentiment}")
                st.write(f"**Summary:** {summary}")
                st.write(f"**Trend Bias:** {trend_bias}")

                # Section 3: Final Recommendation
                st.header("Section 3: Final Recommendation")
                recommendation, explanation = generate_recommendation(trend, trend_bias)
                st.write(f"**Recommendation:** {recommendation}")
                st.write(f"**Explanation:** {explanation}")
            else:
                st.error("Failed to analyze PDF.")
        else:
            st.error("No PDF report available for this stock symbol. Please upload a PDF.")
        
        # Clean up temporary file
        if uploaded_file and os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
    else:
        st.error("Failed to fetch stock data. Please check the symbol and try again.")