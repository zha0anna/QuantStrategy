import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("factor_database.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"]).set_index(["Date", "Ticker"])


factors = pd.read_csv("selected_factors.csv", header=None)[0].tolist() #use best factors from factor_selection.py

# Backtesting Parameters
freq = "M" #monthly rebalance
top_q = 0.1 #each month long top 10%
bot_q = 0.1 #each month short bottom 10%
cost_bps = 10 #transaction cost, .1% per trade
ann = 12


start = None
end = None

if start is not None:
   df = df.loc[df.index.get_level_values("Date") >= pd.Timestamp(start)]
if end is not None:
   df = df.loc[df.index.get_level_values("Date") <= pd.Timestamp(end)]

req = factors + ["target_next_day_ret"]
miss = [c for c in req if c not in df.columns]
if miss:
   raise ValueError(f"Missing columns: {miss}")


#convert daily to monthly data by taking the last observation of each month for each ticker
mnth = (
   df.reset_index()
     .sort_values(["Ticker", "Date"])
     .groupby(["Ticker", pd.Grouper(key="Date", freq=freq)])
     .last()
     .dropna(subset=factors)
     .reset_index()
     .set_index(["Date", "Ticker"])
     .sort_index()
)


if "Adj Close" not in mnth.columns:
   raise ValueError("Adj Close missing")

#calculate next month returns
mnth = mnth.reset_index().sort_values(["Ticker", "Date"])
mnth["ret"] = mnth.groupby("Ticker")["Adj Close"].shift(-1) / mnth["Adj Close"] - 1
mnth = mnth.set_index(["Date", "Ticker"]).sort_index()
mnth = mnth.dropna(subset=["ret"])

#standardize factors by date to get z-scores
def z(x):
   s = x.std()
   if pd.isna(s) or s == 0:
       return pd.Series(0.0, index=x.index)
   return (x - x.mean()) / s


for f in factors:
   mnth[f] = mnth.groupby(level="Date")[f].transform(z)

#factor directions
factor_signs = {
   "f_mom_6m": 1,
   "f_rsi_14": -1,
   "f_vol_20d": 1,
   "f_vol_shock": 1
}

for f in factors:
   mnth[f] = mnth[f] * factor_signs.get(f, 1)

#factor weights
# Momentum gets the largest weight because it had the strongest IR
factor_weights = {
   "f_mom_6m": 0.65,
   "f_rsi_14": 0.15,
   "f_vol_20d": 0.10,
   "f_vol_shock": 0.10
}

used_weights = {f: factor_weights.get(f, 1.0) for f in factors}
weight_sum = sum(used_weights.values())
used_weights = {f: w / weight_sum for f, w in used_weights.items()}
mnth["score"] = sum(used_weights[f] * mnth[f] for f in factors) #combine factors into a single score using the specified weights

#protfolio construction
def w_fn(g):
   n = len(g)
   if n < 10:
       return pd.Series(0.0, index=g.index)


   lc = g["score"].quantile(1 - top_q) #top 10% cutoff
   sc = g["score"].quantile(bot_q)


   long = g["score"] >= lc
   short = g["score"] <= sc


   w = pd.Series(0.0, index=g.index)


   nl = long.sum()
   ns = short.sum()


   if nl > 0:
       w.loc[long] = 1.0 / nl
   if ns > 0:
       w.loc[short] = -1.0 / ns


   return w


mnth["m"] = mnth.groupby(level="Date", group_keys=False).apply(w_fn)

mnth["p_ret"] = mnth["m"] * mnth["ret"]

port = mnth.groupby(level="Date")["p_ret"].sum().to_frame("gross")

w_mat = mnth["m"].unstack("Ticker").fillna(0.0)
to = w_mat.diff().abs().sum(axis=1).fillna(0.0)

cost = cost_bps / 10000.0
port["to"] = to
port["cost"] = port["to"] * cost
port["net"] = port["gross"] - port["cost"]


def stats(r, a=12):
   r = r.dropna()
   if len(r) == 0:
       return {}


   cum = (1 + r).cumprod()
   tot = cum.iloc[-1] - 1
   ar = (1 + tot) ** (a / len(r)) - 1
   vol = r.std() * np.sqrt(a)
   sr = ar / vol if vol != 0 else np.nan


   peak = cum.cummax()
   dd = cum / peak - 1
   mdd = dd.min()


   return {
       "Return": tot,
       "Ann Return": ar,
       "Vol": vol,
       "Sharpe": sr,
       "Max DD": mdd,
       "Avg": r.mean(),
       "Hit": (r > 0).mean(),
   }

res = stats(port["net"], ann)

print("\nPerformance:")
for k, v in res.items():
   print(f"{k}: {v:.4f}")

port["cum_net"] = (1 + port["net"]).cumprod()
port["cum_gross"] = (1 + port["gross"]).cumprod()


port.to_csv("results.csv") #monthly performance of gross, to, cost, net, cum_net, cum_gross (port)
mnth.reset_index().to_csv("holdings.csv", index=False) #what stocks were held each month, with their factors, scores, weights, and returns (mnth)


plt.figure(figsize=(10, 6))
plt.plot(port.index, port["cum_gross"], label="Gross")
plt.plot(port.index, port["cum_net"], label="Net")
plt.title("Cumulative Performance")
plt.xlabel("Date")
plt.ylabel("Growth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# BENCHMARK (S&P 500)
# =========================
sp = pd.read_csv("sp500_index.csv")
sp["Adj Close"] = pd.to_numeric(sp["Adj Close"], errors="coerce")
sp["Date"] = pd.to_datetime(sp["Date"])
sp = sp.sort_values("Date")

# Convert to monthly (same as your strategy)
sp = (
   sp.set_index("Date")
     .resample("M")
     .last()
)

# Compute returns
sp["ret"] = sp["Adj Close"].pct_change()
sp["cum"] = (1 + sp["ret"]).cumprod()

# Align with your strategy dates
sp = sp.reindex(port.index)

# =========================
# Plot comparison
# =========================
plt.figure(figsize=(10,6))
plt.plot(port.index, port["cum_net"], label="Strategy")
plt.plot(sp.index, sp["cum"], label="S&P 500")
plt.title("Strategy vs Benchmark")
plt.legend()
plt.grid(True)
plt.show()