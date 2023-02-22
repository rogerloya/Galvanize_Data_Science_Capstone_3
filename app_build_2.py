import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
import streamlit as st
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Dollar Index Prediction App")

forex = ("EURUSD=X", "GBPUSD=X", "EURGBP=X")
selected_currency = st.selectbox("Select currency pair for prediction", forex)

n_years = st.slider("Years of prediction:", 1, 3)
period = n_years * 365

@st.cache

def load_data(ticker):
    df = yf.download(ticker, START, TODAY)
    del df['Adj Close']
    del df['Volume']
    df.reset_index(inplace=True)
    return df

data_load_state = st.text("Load data...")
df = load_data(selected_currency)
data_load_state.text("Loading data...done")

st.subheader('Raw data')
st.write(df.tail())

def plot_raw_data():
    fig = go.Figure(
    data = [
            go.Candlestick(
                x = df.index, 
                low = df.Low, 
                high = df.High,
                close=df.Close, 
                open = df.Open, 
                increasing_line_color = 'green', 
                decreasing_line_color='red')])
    fig.update_layout(title='Dollar Index')
    st.plotly_chart(fig)

plot_raw_data()


#Forecasting
df_train = df[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close':'y'})

changepoint_prior_scale = 0.02
seasonality_prior_scale = 10

m = Prophet(seasonality_prior_scale=seasonality_prior_scale,changepoint_prior_scale=changepoint_prior_scale)

m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast compnents')

fig2 = m.plot_components(forecast)

st.write(fig2)