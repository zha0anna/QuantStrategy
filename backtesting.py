import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("factor_database.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"]).set_index(["Date", "Ticker"])

factors = ["f_log_ret", "f_mom_6m", "f_dollar_volume"]

freq = "W-FRI"
top_q = 0.2
bot_q = 0.2
cost_bps = 10
ann = 52

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

wk = (
    df.reset_index()
      .sort_values(["Ticker", "Date"])
      .groupby(["Ticker", pd.Grouper(key="Date", freq=freq)])
      .last()
      .dropna(subset=factors)
      .reset_index()
      .set_index(["Date", "Ticker"])
      .sort_index()
)

if "Adj Close" not in wk.columns:
    raise ValueError("Adj Close missing")

wk = wk.reset_index().sort_values(["Ticker", "Date"])
wk["ret"] = wk.groupby("Ticker")["Adj Close"].shift(-1) / wk["Adj Close"] - 1
wk = wk.set_index(["Date", "Ticker"]).sort_index()
wk = wk.dropna(subset=["ret"])

def z(x):
    s = x.std()
    if pd.isna(s) or s == 0:
        return pd.Series(0.0, index=x.index)
    return (x - x.mean()) / s

wk["rev"] = wk.groupby(level="Date")["f_log_ret"].transform(z)
wk["mom"] = wk.groupby(level="Date")["f_mom_6m"].transform(z)
wk["liq"] = wk.groupby(level="Date")["f_dollar_volume"].transform(z)

wk["rev"] = -wk["rev"]

wk["score"] = (wk["rev"] + wk["mom"] + wk["liq"]) / 3.0

def w_fn(g):
    n = len(g)
    if n < 10:
        return pd.Series(0.0, index=g.index)

    lc = g["score"].quantile(1 - top_q)
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

wk["w"] = wk.groupby(level="Date", group_keys=False).apply(w_fn)

wk["p_ret"] = wk["w"] * wk["ret"]

port = wk.groupby(level="Date")["p_ret"].sum().to_frame("gross")

w_mat = wk["w"].unstack("Ticker").fillna(0.0)
to = w_mat.diff().abs().sum(axis=1).fillna(0.0)

cost = cost_bps / 10000.0
port["to"] = to
port["cost"] = port["to"] * cost
port["net"] = port["gross"] - port["cost"]

def stats(r, a=52):
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

port.to_csv("results.csv")
wk.reset_index().to_csv("holdings.csv", index=False)

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