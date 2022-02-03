# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly import graph_objs as go
from prophet import Prophet
from datetime import date
from prophet.plot import plot_plotly, plot_components_plotly
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle


st.title("Stock Prediction Model")

stocks = ('AAPL','AMZN','GOOG','FB','MSFT','NVDA','TSLA')
start = '2017-01-01'
today = date.today().strftime('%Y-%m-%d')

select_stock = st.sidebar.selectbox('Choose a stock', stocks)

section = st.sidebar.radio('Choose Section', ['Exploratory Analysis',
                                              'Forecast Engine'])
#this section is for functions
@st.cache
def cal_ema(prices, days, smoothing = 2):
    ema = []
    for price in prices [:days]:
        ema.append(sum(prices[:days]) / days)
    for price in prices[days:]:
        ema.append((price * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
    return ema

@st.cache(allow_output_mutation=True)
def load_data(ticker):
    df = yf.download(ticker, start, today).reset_index()
    df['dow']  = df['Date'].dt.dayofweek
    df['month']   = df['Date'].dt.month
    df['year']   = df['Date'].dt.year
    df['day']   = df['Date'].dt.day
    df['week']   = df['Date'].dt.week
    df['SMA_20'] = df['Close'].rolling(20).mean().shift().bfill()
    df['SMA_50'] = df['Close'].rolling(50).mean().shift().bfill()
    df['SMA_200'] = df['Close'].rolling(200).mean().shift().bfill()
    df['EMA_12'] = cal_ema(df['Close'],12,smoothing = 2)
    df['EMA_26'] = cal_ema(df['Close'],26,smoothing = 2)
    return df

df = load_data(select_stock)

@st.cache
def load_data_xgb():
    df['Close_log'] = np.log(df['Close'])
    df['FwdRet1']   = df['Close_log'].shift(-1)   - df['Close_log']
    df['FwdRet5']   = df['Close_log'].shift(-5)   - df['Close_log']
    df['FwdRet30']  = df['Close_log'].shift(-30)  - df['Close_log']
    df['FwdRet126'] = df['Close_log'].shift(-126) - df['Close_log']
    df['FwdRet252'] = df['Close_log'].shift(-252) - df['Close_log']
    avgs = [5, 10, 30, 126, 252]
    for avg in avgs:
        df[f'sma-{avg}-rat'] = df['Close'].rolling(avg).mean() / df['Close']
        df[f'vol-{avg}-chg'] = df['Volume'].pct_change(avg)
    df['Volatility'] = ((df['High'] - df['Low']) / df['Close']).pct_change()
    fwd_rets = [1, 5, 30, 126, 252]

    for ret in fwd_rets:
        for i in range(1, 3):
            df[f'FwdRet{ret}-shft-{i}'] = df[f'FwdRet{ret}'].shift(i)
    thirty_cols = ['FwdRet30', 'sma-5-rat', 'Volatility',
       'vol-5-chg', 'sma-10-rat', 'vol-10-chg', 'sma-30-rat', 'vol-30-chg',
       'sma-126-rat', 'vol-126-chg', 'sma-252-rat', 'vol-252-chg', 'FwdRet1-shft-1', 'FwdRet1-shft-2', 'FwdRet5-shft-1',
       'FwdRet5-shft-2', 'FwdRet30-shft-1', 'FwdRet30-shft-2',
       'FwdRet126-shft-1', 'FwdRet126-shft-2', 'FwdRet252-shft-1',
       'FwdRet252-shft-2','dow','day','month','year','week']

    df30 = df[thirty_cols]
    

@st.cache
def load_model():
    with open('stockxgb.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

#this section lays out visualizations on page
if section == 'Exploratory Analysis':
    st.subheader('Historic Trend')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df['Date'], y = df['Close'], name = 'Stock_close'))
    fig.add_trace(go.Scatter(x = df['Date'], y = df['Open'], name = 'Stock_open'))
    fig.add_trace(go.Scatter(x = df['Date'], y = df['EMA_26'], name = 'EMA_26'))
    fig.add_trace(go.Scatter(x = df['Date'], y = df['SMA_200'], name = 'SMA_200'))
    fig.layout.update(title_text = select_stock, xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
    
    st.subheader('Raw Data')
    st.write(df.tail(100))
    
if section =='Forecast Engine':
    model = st.sidebar.selectbox('Choose Model', ['Prophet',
                                              'XGboost', 
                                              'Classifier'])
    st.caption('Disclaimer: This dashbord is built for a data science project and does not act as financial advice')
    
    if model =='Prophet':
        st. subheader('This model aims to predict price trend for the next 365 days')
        prophet_data = df[['Date', 'Close']].rename(columns = {'Date':'ds','Close':'y' })
        fbp = Prophet(daily_seasonality = True)
        fbp.fit(prophet_data)
        fut = fbp.make_future_dataframe(periods=365) 
        forecast = fbp.predict(fut)
        fig1 = plot_plotly(fbp, forecast)
        
        
        st.write('Forcast Data')
        st.plotly_chart(fig1)   
        
        st.write('Forecast components')
        fig2 = fbp.plot_components(forecast)
        st.write(fig2)
    
        st.write(forecast.tail(50))
        
#    if model =='XGboost':
 #       st. subheader('This model aims to predict price return in 30 days')
  #      load_data_xgb()
   #     mod = load_model()
    #    
     #   vol = st.selectbox("volatility",
      #                              df['id'].unique().tolist())
        
        
        