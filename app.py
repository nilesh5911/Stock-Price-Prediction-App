import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# --- Constants ---
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# --- Streamlit App Start ---
st.set_page_config(page_title="📈 Stock Price Prediction App", layout="wide")
st.title("📈 Stock Price Prediction App")

# --- Sidebar for User Inputs ---
st.sidebar.header('User Input Features')

# Predefined list of stocks
stocks = ("AAPL", "GOOG", "MSFT", "GME", "NVDA", "INTC", "TSLA", "DELL", "AMZN", "BTC-USD", "DOGE-USD", "ESTC")

# Select the stock from the list
selected_stock = st.sidebar.selectbox("Select Dataset for Prediction", stocks)
n_years = st.sidebar.slider("Years of prediction:", 1, 4)
period = n_years * 365

# --- Load Data Function ---
@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        if data.empty:
            return None
        data = data.dropna(how='all', axis=1)  # Drop empty columns
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# --- Main ---
data_load_state = st.text('⏳ Loading data...')
data = load_data(selected_stock)

if data is not None:
    data_load_state.text('✅ Data Loaded Successfully!')

    # Show Raw Data
    if st.checkbox('Show raw data'):
        st.subheader('Raw Data')
        st.write(data.tail())

    # --- Plot Raw Data ---
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open Price"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price"))
        fig.update_layout(title_text='Time Series Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    st.subheader('Price vs Time')
    plot_raw_data()

    # --- Rolling Average ---
    st.subheader('Rolling Average')
    rolling_avg = data['Close'].rolling(window=30).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price"))
    fig2.add_trace(go.Scatter(x=data['Date'], y=rolling_avg, name="30-day Rolling Avg", line=dict(dash='dash')))
    fig2.update_layout(title_text='Stock Price with Rolling Average')
    st.plotly_chart(fig2)

    # --- Volatility ---
    st.subheader('Stock Price Volatility')
    data['Volatility'] = data['Close'].pct_change() * 100
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data['Date'], y=data['Volatility'], name="Daily Volatility", line=dict(color='red')))
    fig3.update_layout(title_text='Stock Price Volatility (Daily % Change)', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)

    # --- Forecasting ---
    df_train = data[['Date', 'Close']].copy()
    df_train.columns = ['ds', 'y']  # Prophet requires ds, y

    # Ensure datetime and numeric formats
    df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')

    if isinstance(df_train['y'], pd.Series):
        df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
        df_train['y'] = df_train['y'].interpolate(method='linear')
        df_train['y'] = df_train['y'].fillna(method='bfill').fillna(method='ffill')
    else:
        st.error("❌ 'y' column is invalid. Please check the data.")
        st.stop()

    m = Prophet()
    m.fit(df_train)

    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # --- Show Forecasted Data ---
    st.subheader('Forecasted Data')
    if st.checkbox('Show forecasted data table'):
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # --- Forecast Plot ---
    st.subheader('Forecast Plot')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # --- Forecast Components ---
    st.subheader('Forecast Components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    # --- Summary ---
    st.subheader("Prediction Summary")
    forecast_summary = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1)
    st.write(f"**Predicted closing price for {forecast_summary['ds'].values[0]}:**")
    st.write(f"Point prediction: ${forecast_summary['yhat'].values[0]:.2f}")
    st.write(f"Prediction range: ${forecast_summary['yhat_lower'].values[0]:.2f} - ${forecast_summary['yhat_upper'].values[0]:.2f}")

    # --- Download Buttons ---
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_forecast = convert_df(forecast)
    st.download_button(
        label="📥 Download Forecasted Data as CSV",
        data=csv_forecast,
        file_name=f'{selected_stock}_forecast.csv',
        mime='text/csv',
    )

    csv_historical = convert_df(data)
    st.download_button(
        label="📥 Download Historical Data as CSV",
        data=csv_historical,
        file_name=f'{selected_stock}_historical_data.csv',
        mime='text/csv',
    )

else:
    data_load_state.text("❌ Failed to load data.")
    st.error("⚠️ No data available for the selected stock symbol.")

# --- Footer ---
st.markdown("---")
st.caption("🚀 Built with Streamlit, Prophet, yFinance & Plotly")
