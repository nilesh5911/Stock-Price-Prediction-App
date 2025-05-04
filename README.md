# 📈 Stock Price Prediction App

This is an interactive Streamlit application for forecasting stock prices using historical data. It leverages the **Prophet** model by Facebook for time series forecasting and uses **yFinance** for real-time stock data. The app also visualizes raw stock data, rolling averages, and daily volatility.

## 🚀 Features

- 📊 Select from a list of predefined stocks (e.g., AAPL, MSFT, TSLA, BTC-USD, etc.)
- 📅 Predict future stock prices for 1 to 4 years
- 📈 Visualizations:
  - Raw Time Series (Open & Close)
  - 30-day Rolling Average
  - Daily Volatility (% Change)
  - Prophet Forecast and Components
- 🧠 Built using Facebook Prophet for robust forecasting
- 💾 Downloadable forecasted and historical data in CSV format

## 🧰 Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [yFinance](https://pypi.org/project/yfinance/)
- [Facebook Prophet](https://facebook.github.io/prophet/)
- [Plotly](https://plotly.com/python/)

## 📦 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/stock-price-prediction-app.git
   cd stock-price-prediction-app
   
## Install dependencies
  pip install -r requirements.txt
## Run the Streamlit app
  streamlit run app.py
