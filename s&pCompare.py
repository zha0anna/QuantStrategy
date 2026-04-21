import yfinance as yf
import pandas as pd

START = "2015-01-01"
END = "2025-12-31"

# Download S&P 500 index
sp = yf.download(
    "^GSPC",   # THIS is the key
    start=START,
    end="2026-01-01",
    interval="1d",
    auto_adjust=False
)

# Keep only needed columns
sp = sp[["Adj Close"]].copy()

# Reset index for consistency
sp = sp.reset_index()

# Save
sp.to_csv("sp500_index.csv", index=False)

print(sp.head())