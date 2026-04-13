import pandas as pd
import numpy as np

# 1. Load and Prepare Data
df = pd.read_csv('stocks_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date'])


# Define a helper for RSI (Relative Strength Index)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 2. Factor Generation (Grouped by Ticker)
# We use groupby('Ticker') to ensure rolling windows stay within the same stock
grouped = df.groupby('Ticker')

# Factor 1: Daily Log Returns (Statistically more robust than pct_change)
df['f_log_ret'] = grouped['Adj Close'].transform(lambda x: np.log(x / x.shift(1)))

# Factor 2: Short-term Momentum (1-Month / 21 Trading Days)
df['f_mom_1m'] = grouped['Adj Close'].pct_change(21)

# Factor 3: Intermediate Momentum (6-Month / 126 Trading Days)
df['f_mom_6m'] = grouped['Adj Close'].pct_change(126)

# Factor 4: Historical Volatility (20-day rolling standard deviation of returns)
df['f_vol_20d'] = grouped['f_log_ret'].rolling(window=20).std().reset_index(0, drop=True)

# Factor 5: Relative Strength Index (RSI - 14 Day)
df['f_rsi_14'] = grouped['Adj Close'].transform(calculate_rsi)

# Factor 6: Trend Strength (Price / 50-Day Moving Average)
# Values > 1 indicate an uptrend relative to the average
df['f_sma_50_ratio'] = df['Adj Close'] / grouped['Adj Close'].transform(lambda x: x.rolling(50).mean())

# Factor 7: Volume Momentum (Current Volume / 20-Day Average Volume)
# High values indicate unusual trading activity
df['f_vol_shock'] = df['Volume'] / grouped['Volume'].transform(lambda x: x.rolling(20).mean())

# Factor 8: Intraday Range (Proxy for daily uncertainty)
df['f_day_range'] = (df['High'] - df['Low']) / df['Low']

# Factor 9: Liquidity (Dollar Volume)
# Total value traded in a day
df['f_dollar_volume'] = df['Adj Close'] * df['Volume']

# Factor 10: Mean Reversion (Z-Score of price relative to 20-day mean)
# Measures how far the price has deviated from its recent average in standard deviations
roll_mean = grouped['Adj Close'].transform(lambda x: x.rolling(20).mean())
roll_std = grouped['Adj Close'].transform(lambda x: x.rolling(20).std())
df['f_zscore_20d'] = (df['Adj Close'] - roll_mean) / roll_std

# Factor 11: Distance from 52-Week High
# Captures "anchoring" bias and breakout potential
df['f_dist_52w_high'] = df['Adj Close'] / grouped['Adj Close'].transform(lambda x: x.rolling(252).max())

# 3. Create the Target Variable (Forward 1-day Return)
# This shifts the next day's return back to "today's" row for training/analysis
df['target_next_day_ret'] = grouped['f_log_ret'].shift(-1)

# 4. Clean and Save
# Drop rows that have NaNs due to rolling window startup (e.g., first 252 days for the 52w high)
factor_db = df.dropna().set_index(['Date', 'Ticker'])

print(factor_db.head())
factor_db.to_csv('factor_database.csv')