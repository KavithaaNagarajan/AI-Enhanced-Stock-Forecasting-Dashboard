# AI-Enhanced Stock Forecasting Dashboard
# Overview
This project implements an interactive web-based dashboard for stock price forecasting and sentiment analysis, built using Streamlit. The dashboard enables users to input a stock symbol (e.g., INFY, TCS, RELIANCE), fetch 1-year historical stock data, perform machine learning-based price forecasting, analyze a PDF report for sentiment, and generate a Buy/Hold/Sell recommendation based on combined ML and AI insights.
# Features

Stock Symbol Input: Accepts stock symbols (e.g., INFY, TCS, RELIANCE) and fetches 1-year historical data using yfinance. Automatically appends .NS for Indian stocks (e.g., RELIANCE becomes RELIANCE.NS).
PDF Analysis: Extracts company name, sentiment (Positive/Negative/Neutral), summary, and trend bias from a pre-mapped PDF report (e.g., reliance_report.pdf for RELIANCE) or a user-uploaded PDF using PyMuPDF and TextBlob.
ML Prediction: Uses Linear Regression to forecast stock prices for the next 7 days, displaying a Plotly chart, Mean Absolute Error (MAE), and trend direction (Uptrend/Downtrend).
AI Prediction: Performs sentiment analysis on PDF content, providing sentiment, a summary, and trend bias (Likely Positive/Likely Negative/Neutral).
Final Recommendation: Combines ML and AI results to recommend Buy, Hold, or Sell with a concise explanation.
Interactive UI: Built with Streamlit, featuring three sections (ML Prediction, AI Prediction, Final Recommendation) and an interactive Plotly chart for visualization.

# Tech Stack

Python Libraries: streamlit, yfinance, pandas, numpy, scikit-learn, PyMuPDF, textblob, plotly, nltk
ML Model: Linear Regression (scikit-learn)
NLP: TextBlob for sentiment analysis
PDF Processing: PyMuPDF for text extraction
Frontend: Streamlit with Plotly for interactive charts

# Prerequisites

Python 3.7 or higher
A LaTeX installation (e.g., TeX Live) or access to Overleaf for compiling sample PDFs
Internet connection for fetching yfinance data

# Installation

Clone the Repository:
git clone <repository-url>
cd stock-forecasting-dashboard


Install Dependencies:Create a requirements.txt file with the following:
streamlit
yfinance
pandas
numpy
scikit-learn
PyMuPDF
textblob
plotly
nltk

#Install using:
pip install -r requirements.txt


# Download NLTK Corpora:
Required for TextBlob sentiment analysis:
python -m textblob.download_corpora


# Prepare Sample PDFs:
Place the following text-based PDFs in the project directory:

infosys_report.pdf (for INFY)
tcs_report.pdf (for TCS)
reliance_report.pdf (for RELIANCE)

Alternatively, generate these PDFs using the provided LaTeX files (infosys_report.tex, tcs_report.tex, reliance_report.tex) via Overleaf or a local LaTeX installation (e.g., pdflatex reliance_report.tex). Users can also upload custom PDFs via the dashboard.


# Running the Dashboard

Run the Streamlit app:
streamlit run app.py


Access the dashboard at http://localhost:8501 in your browser.


# Usage

Open the dashboard (http://localhost:8501).
Enter a stock symbol (e.g., RELIANCE) in the input field. The dashboard automatically appends .NS (e.g., RELIANCE.NS).
Optionally upload a PDF report. If none is uploaded, the dashboard uses the pre-mapped PDF (e.g., reliance_report.pdf for RELIANCE).
Click Analyze to view:
ML Prediction: 7-day price forecast chart, MAE, and trend direction.
AI Prediction: Company name, sentiment, summary, and trend bias from the PDF.
Final Recommendation: Buy/Hold/Sell recommendation with an explanation.



# Sample Inputs

Stock Symbols: INFY, TCS, RELIANCE
# Sample PDFs:
infosys_report.pdf: Infosys Limited Annual Report 2025
tcs_report.pdf: Tata Consultancy Services Annual Report 2025
reliance_report.pdf: Reliance Industries Limited Annual Report 2025



# Deployment
To deploy the dashboard on Render (an open-source cloud platform):

# Prepare the Repository:

Ensure all project files (including stock_dashboard.py, requirements.txt, and sample PDFs) are in your GitHub repository.
Create a Procfile with:web: streamlit run app.py --server.port $PORT




# Set Up Render:

Sign up at https://render.com and create a new Web Service.
Connect your GitHub repository.
Configure:
Build Command: pip install -r requirements.txt && python -m textblob.download_corpora
Start Command: Handled by the Procfile
Environment: Python 3


Add environment variables if needed (e.g., for NLTK data paths).
Ensure sample PDFs are included in the repository or configure the app to handle user uploads only.


# Deploy:

Trigger a deployment on Render. The dashboard will be accessible via a public URL (e.g., https://your-app.onrender.com).



# Sample Output
For stock symbol : INFY
![image](https://github.com/user-attachments/assets/8de91856-a451-490c-8885-2cdc0a7953b6)

# ML Prediction:
7-day forecast: Slight price increase (e.g., 2% uptrend).
MAE: ~2.5% (example).
Trend: Uptrend.
![image](https://github.com/user-attachments/assets/1c3df803-df34-436b-b150-4d48e9de61e2)


# AI Prediction:
Company: Infoys Limited
Sentiment: Positive (e.g., polarity 0.3)
Summary: “Infosys reports strong Q1 2025 growth in energy and retail.”
Trend Bias: Likely Positive
![image](https://github.com/user-attachments/assets/269f6b57-455d-4e47-9dd7-48908157dbbd)


# Recommendation:
Buy: “ML predicts an uptrend, and positive PDF sentiment indicates strong fundamentals.”

![image](https://github.com/user-attachments/assets/390489cd-ee22-4676-8f2f-5ccc9caf9ad6)


# Notes

Ensure a stable internet connection for yfinance data fetching.
For better ML accuracy, consider experimenting with advanced models like LSTM or Prophet.
TextBlob’s sentiment analysis is basic; for production, consider transformers (e.g., BERT) for more robust NLP.
Sample PDFs must be text-based (not scanned images) for PyMuPDF to extract content.


