import streamlit as st
from datetime import date
import numpy as np

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Economic Prediction App")

stocks = ("GC=F", "^DJI", "EURUSD=X", "DX-Y.NYB")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 3)
period = n_years * 365

@st.cache

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close':'y'})

changepoint_prior_scale = .01
seasonality_prior_scale = 5

m = Prophet(seasonality_prior_scale=seasonality_prior_scale,changepoint_prior_scale=changepoint_prior_scale)

m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('Forecast Data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('FBProphet Forecast Components')

fig2 = m.plot_components(forecast)

st.write(fig2)
