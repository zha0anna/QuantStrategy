import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
# Ensure your CSV has: Date, Ticker, Open, High, Low, Close, Adj Close, Volume
df = pd.read_csv('stocks_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date'])

# Helper for RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 2. FACTOR CONSTRUCTION (The Database)
# ==========================================
grouped = df.groupby('Ticker')

# Theme: Momentum
df['f_mom_1m'] = grouped['Adj Close'].pct_change(21)
df['f_mom_6m'] = grouped['Adj Close'].pct_change(126)
df['f_sma_50_ratio'] = df['Adj Close'] / grouped['Adj Close'].transform(lambda x: x.rolling(50).mean())
df['f_dist_52w_high'] = df['Adj Close'] / grouped['Adj Close'].transform(lambda x: x.rolling(252).max())

# Theme: Mean Reversion / Technical
df['f_log_ret'] = grouped['Adj Close'].transform(lambda x: np.log(x / x.shift(1)))
df['f_rsi_14'] = grouped['Adj Close'].transform(calculate_rsi)
roll_mean = grouped['Adj Close'].transform(lambda x: x.rolling(20).mean())
roll_std = grouped['Adj Close'].transform(lambda x: x.rolling(20).std())
df['f_zscore_20d'] = (df['Adj Close'] - roll_mean) / roll_std

# Theme: Volatility & Risk
df['f_vol_20d'] = df['f_log_ret'].rolling(window=20).std()
df['f_day_range'] = (df['High'] - df['Low']) / df['Low']

# Theme: Liquidity & Volume
df['f_dollar_volume'] = df['Adj Close'] * df['Volume']
df['f_vol_shock'] = df['Volume'] / grouped['Volume'].transform(lambda x: x.rolling(20).mean())

# TARGET: Forward 1-day Return
df['target_next_day_ret'] = grouped['f_log_ret'].shift(-1)

# Clean up
df = df.dropna().set_index(['Date', 'Ticker'])
factors = [col for col in df.columns if col.startswith('f_')]

# ==========================================
# 3. PERIOD-BY-PERIOD PERFORMANCE (IC/IR)
# ==========================================
def calculate_ic(group):
    ic_values = {f: spearmanr(group[f], group['target_next_day_ret'])[0] for f in factors}
    return pd.Series(ic_values)

# Daily ICs
ic_series = df.groupby('Date').apply(calculate_ic)

# Aggregate Summary
ic_summary = pd.DataFrame({
    'Mean_IC': ic_series.mean(),
    'Std_IC': ic_series.std(),
    'IR': ic_series.mean() / ic_series.std(),
    'Hit_Rate': (ic_series > 0).mean()
}).sort_values('IR', key=abs, ascending=False)

# Yearly Performance (Requirement: Period-by-Period Ranking)
yearly_ic = ic_series.groupby(ic_series.index.year).mean()

print("--- Aggregate Factor Performance ---")
print(ic_summary)
print("\n--- Yearly Mean IC (Period-by-Period) ---")
print(yearly_ic)

# ==========================================
# 4. CORRELATION & FINAL SELECTION
# ==========================================
factor_corr = df[factors].corr()

# Visualize
plt.figure(figsize=(12, 10))
sns.heatmap(factor_corr, annot=True, cmap='vlag', center=0, fmt=".2f")
plt.title("Factor Correlation Matrix (Orthogonality Check)")
plt.show()

# Selection Logic with Silo Enforcement
ic_summary['abs_IR'] = ic_summary['IR'].abs()
ranked_candidates = ic_summary.sort_values('abs_IR', ascending=False)

final_selection = []
correlation_threshold = 0.3 # Strict threshold to ensure thematic diversity

print("\n--- Selection Log ---")
for factor in ranked_candidates.index:
    if len(final_selection) >= 3:
        break
    
    if not final_selection:
        final_selection.append(factor)
        print(f"Selected: {factor:<15} (Top Absolute IR)")
    else:
        # Check if current candidate is too correlated with already selected ones
        is_redundant = any(abs(factor_corr.loc[factor, f]) > correlation_threshold for f in final_selection)
        
        if not is_redundant:
            final_selection.append(factor)
            print(f"Selected: {factor:<15} (Unique/Orthogonal Signal)")
        else:
            offending = [f for f in final_selection if abs(factor_corr.loc[factor, f]) > correlation_threshold][0]
            print(f"Skipped:  {factor:<15} (Redundant with {offending}: {factor_corr.loc[factor, offending]:.2f})")

print(f"\nFINAL STRATEGY FACTORS: {final_selection}")