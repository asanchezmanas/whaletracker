"""
QDN Real Data Test v4 — Antifragile (Purchases Only)

Taleb antifragile = buy what survives crisis, sells exponential in bull.
Insider PURCHASES are the signal. Sales = noise (compensation, tax, diversification).

This test:
1. Fetches insider PURCHASES across diverse tickers (including small/mid-cap)
2. Features emphasize: buying during crisis, high VIX, post-crash
3. Model scores = purchase conviction quality
"""

import sys
sys.path.insert(0, "whaletracker/src")

import numpy as np
import pandas as pd
import torch
import httpx
import time
import re
import warnings
from datetime import datetime, timedelta
from io import StringIO

warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.WARNING)

print("=" * 60)
print("QDN ANTIFRAGILE TEST — Insider Purchases Only")
print("=" * 60)

# Single shared HTTP client
_http = httpx.Client(
    timeout=30,
    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    follow_redirects=True,
)


def fetch_purchases(days_back: int = 1500, min_value: int = 1000) -> pd.DataFrame:
    """
    Fetch insider PURCHASES from OpenInsider.com.
    Uses the cross-market purchase screener.
    """
    all_dfs = []
    for page in [1, 2, 3]:
        url = (
            f"http://openinsider.com/screener?"
            f"s=&o=&pl={min_value}&ph=&ll=&lh=&fd={days_back}&fdr=&td=0&tdr="
            f"&feession=&cession=&sidTL=&tidTL=&tiession=&ession="
            f"&sicTL=&sicession=&pession=&cnt=1000&page={page}"
        )
        
        print(f"  Page {page}...", end=" ", flush=True)
        r = _http.get(url)
        
        if r.status_code != 200:
            print(f"error {r.status_code}")
            continue
        
        try:
            tables = pd.read_html(StringIO(r.text))
            for tbl in tables:
                cols = [str(c).lower() for c in tbl.columns]
                if any("trade" in c for c in cols) and len(tbl) > 5:
                    all_dfs.append(tbl)
                    print(f"{len(tbl)} rows")
                    break
            else:
                print("no table")
        except Exception as e:
            print(f"error: {e}")
        time.sleep(0.5)
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


# ═══════════════════════════════════════
# 1. Fetch insider purchases
# ═══════════════════════════════════════
print("\n[1/5] Fetching insider PURCHASES...")

raw = fetch_purchases(days_back=1500, min_value=1000)

if raw.empty:
    print("❌ No data"); sys.exit(1)

# Normalize column names (OpenInsider uses non-breaking spaces)
raw.columns = [str(c).replace('\xa0', ' ').strip() for c in raw.columns]
print(f"\n  Columns: {list(raw.columns)}")

# Build column mapping dynamically
col_map = {}
for i, c in enumerate(raw.columns):
    cl = c.lower()
    if "filing" in cl and "date" in cl: col_map["filing_date"] = c
    elif "trade" in cl and "date" in cl: col_map["trade_date"] = c
    elif cl == "ticker": col_map["ticker"] = c
    elif "insider" in cl and "name" in cl: col_map["insider_name"] = c
    elif cl == "title": col_map["title"] = c
    elif "trade" in cl and "type" in cl: col_map["trade_type"] = c
    elif cl == "price": col_map["price"] = c
    elif cl == "qty": col_map["qty"] = c
    elif cl == "owned": col_map["owned"] = c
    elif cl == "value": col_map["value"] = c

print(f"  Mapped: {list(col_map.keys())}")

# Parse into standard format
trades_list = []
for _, row in raw.iterrows():
    try:
        trade_type = str(row[col_map["trade_type"]]) if "trade_type" in col_map else ""
        
        # Only purchases
        if "Purchase" not in trade_type and "Buy" not in trade_type:
            continue
        
        ticker = str(row[col_map["ticker"]]).strip() if "ticker" in col_map else None
        if not ticker or ticker == "nan":
            continue
        
        record = {
            "ticker": ticker,
            "filing_date": str(row[col_map.get("filing_date", "")]) if "filing_date" in col_map else None,
            "transaction_date": str(row[col_map.get("trade_date", "")]) if "trade_date" in col_map else None,
            "insider_name": str(row[col_map.get("insider_name", "")]) if "insider_name" in col_map else "Unknown",
            "insider_title": str(row[col_map.get("title", "")]) if "title" in col_map else "",
            "transaction_code": "P",
            "source": "sec_form4",
        }
        
        # Parse numeric fields
        for field, key in [("price", "price"), ("shares", "qty"), 
                            ("ownership_after", "owned"), ("value", "value")]:
            if key in col_map:
                val_str = str(row[col_map[key]]).replace(',', '').replace('$', '').replace('+', '')
                val_clean = re.sub(r'[^\d.]', '', val_str)
                try:
                    record[field] = float(val_clean) if val_clean else None
                except:
                    record[field] = None
            else:
                record[field] = None
        
        trades_list.append(record)
    except Exception:
        continue

trades = pd.DataFrame(trades_list)
trades["transaction_date"] = pd.to_datetime(trades["transaction_date"], errors="coerce")
trades["filing_date"] = pd.to_datetime(trades["filing_date"], errors="coerce")
trades = trades.dropna(subset=["transaction_date", "ticker"])

# Filter date range: need 6 months forward for labels
trades = trades[
    (trades["transaction_date"] >= datetime(2021, 1, 1)) &
    (trades["transaction_date"] <= datetime(2024, 6, 1))
].copy()

print(f"\n  Purchases: {len(trades)}")
print(f"  Unique tickers: {trades['ticker'].nunique()}")
print(f"  Unique insiders: {trades['insider_name'].nunique()}")

# Top tickers by purchase count
top_tickers = trades["ticker"].value_counts().head(20)
print(f"\n  Top tickers by purchase volume:")
for tk, cnt in top_tickers.items():
    print(f"    {tk}: {cnt}")

has_price = trades["price"].notna().sum()
has_val = trades["value"].notna().sum()
print(f"\n  Price: {has_price}/{len(trades)} | Value: {has_val}/{len(trades)}")


# ═══════════════════════════════════════
# 2. Stock prices (only for tickers with enough data)
# ═══════════════════════════════════════
print("\n[2/5] Stock prices...")
import yfinance as yf
from qdn.data.market_connector import MarketConnector

mc = MarketConnector()
tickers = sorted(trades["ticker"].unique())
price_cache = {}
company_cache = {}

# Limit to top 30 tickers
if len(tickers) > 30:
    top30 = trades["ticker"].value_counts().head(30).index.tolist()
    tickers = sorted(top30)
    trades = trades[trades["ticker"].isin(tickers)].copy()
    print(f"  Limiting to top {len(tickers)} tickers")

for t in tickers:
    try:
        p = yf.download(t, start="2020-09-01", end="2025-01-01", progress=False)
        if isinstance(p.columns, pd.MultiIndex):
            p.columns = p.columns.get_level_values(0)
        if not p.empty:
            price_cache[t] = p
            company_cache[t] = mc.get_company_info(t)
    except:
        company_cache[t] = {"ticker": t}
    time.sleep(0.1)

print(f"  {len(price_cache)} tickers with price data")


# ═══════════════════════════════════════
# 3. Features + labels (6-month forward return)
# ═══════════════════════════════════════
print("\n[3/5] Computing features + 6-month forward returns...")
from qdn.features.engineer import FeatureEngineer, FEATURE_NAMES

eng = FeatureEngineer()
feats, labs, dates, metas = [], [], [], []
skip = 0

for idx, row in trades.iterrows():
    tk = row["ticker"]
    td = row["transaction_date"]
    if tk not in price_cache:
        skip += 1; continue
    
    close = price_cache[tk]["Close"]
    try:
        future = close[close.index > td]
        past = close[close.index <= td]
        if len(future) < 100 or past.empty:
            skip += 1; continue
        ret = (float(future.iloc[min(126,len(future)-1)]) - float(past.iloc[-1])) / float(past.iloc[-1])
    except:
        skip += 1; continue

    try:
        f = eng.compute_features(
            transaction=row.to_dict(),
            historical_transactions=trades[trades.index < idx],
            company_info=company_cache.get(tk, {}),
            market_data=price_cache[tk],
            macro_snapshot={"vix": 20, "yield_curve": 0.5, "dxy": 103},
        )
        feats.append(f); labs.append(ret); dates.append(td)
        metas.append({"ticker": tk, "insider": row["insider_name"],
                       "value": row.get("value"), "price": row.get("price")})
    except:
        skip += 1

print(f"  Valid: {len(feats)} | Skipped: {skip}")

if len(feats) < 20:
    print("❌ Not enough data"); sys.exit(1)

X = np.array(feats, dtype=np.float32)
y = np.array(labs, dtype=np.float32)
D = np.array(dates)
M = pd.DataFrame(metas)

# Label analysis
print(f"\n  Returns: mean={y.mean():.2%}, median={np.median(y):.2%}, std={y.std():.2%}")
print(f"  Win rate: {(y > 0).mean():.1%}")
print(f"  Losses > -20%: {(y < -0.2).sum()} ({(y < -0.2).mean():.1%})")
print(f"  Gains > 50%: {(y > 0.5).sum()} ({(y > 0.5).mean():.1%})")

# Antifragile analysis: what happens when insiders buy during high vol?
vol_feat = X[:, FEATURE_NAMES.index("volatility_90d")]
high_vol = vol_feat > np.percentile(vol_feat, 75)
low_vol = vol_feat < np.percentile(vol_feat, 25)
if high_vol.sum() > 5 and low_vol.sum() > 5:
    print(f"\n  ── Antifragile Signal ──")
    print(f"  Buys during HIGH vol ({high_vol.sum()}): {y[high_vol].mean():.2%} mean return")
    print(f"  Buys during LOW vol ({low_vol.sum()}):  {y[low_vol].mean():.2%} mean return")
    delta = y[high_vol].mean() - y[low_vol].mean()
    print(f"  Delta: {delta:+.2%} ({'✅ antifragile' if delta > 0 else '❌ fragile'})")

# Feature coverage
print(f"\n  Feature coverage:")
cov = (X != 0).mean(axis=0)
for i, name in enumerate(FEATURE_NAMES):
    bar = "█" * int(cov[i] * 20)
    print(f"    {name:30s} {cov[i]:5.0%} {bar}")


# ═══════════════════════════════════════
# 4. Train
# ═══════════════════════════════════════
print("\n[4/5] Training (purchases only — antifragile scoring)...")
from qdn.config import QDNConfig
from qdn.trainer import QDNTrainer

cfg = QDNConfig()
cfg.training.max_epochs = 80
cfg.training.early_stopping_patience = 20
cfg.training.batch_size = min(32, max(8, len(X) // 6))
cfg.training.lr = 1e-3

# Temporal sort
si = np.argsort(D)
X, y, D = X[si], y[si], D[si]
M = M.iloc[si].reset_index(drop=True)

sp = int(len(X) * 0.7)
tx, vx, ty, vy = X[:sp], X[sp:], y[:sp], y[sp:]
vm = M.iloc[sp:].reset_index(drop=True)

print(f"  Train: {len(tx)} | Val: {len(vx)}")

trainer = QDNTrainer(cfg)
vr = trainer.train(tx, ty, vx, vy)


# ═══════════════════════════════════════
# 5. Results
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print("[5/5] ANTIFRAGILE RESULTS — Insider Purchases Only")
print("=" * 60)

model = trainer.model
model.eval()
with torch.no_grad():
    scores = model(torch.tensor(vx))["convexity_score"].numpy().flatten()

print(f"\n  Scores: mean={scores.mean():.1f} std={scores.std():.1f} "
      f"[{scores.min():.0f}, {scores.max():.0f}]")

from qdn.evaluation import evaluate_predictions

print(f"\n  {'Thr':>5} {'N':>5} {'Sortino':>8} {'Win%':>7} {'MeanRet':>9} {'MaxDD':>9} {'Convex':>8}")
print(f"  {'─'*5} {'─'*5} {'─'*8} {'─'*7} {'─'*9} {'─'*9} {'─'*8}")

for t in [80, 70, 60, 50, 40, 30, 20]:
    r = evaluate_predictions(scores, vy, threshold=t)
    if r.num_trades > 0:
        print(f"  {t:>5} {r.num_trades:>5} {r.sortino:>8.2f} {r.win_rate:>6.1%} "
              f"{r.mean_return:>8.2%} {r.max_drawdown:>8.2%} {r.convexity_ratio:>8.2f}")
    else:
        print(f"  {t:>5} {0:>5} {'─':>8} {'─':>7} {'─':>9} {'─':>9} {'─':>8}")

# Baseline
print(f"\n  ── Buy-All-Purchases Baseline ──")
print(f"  {len(vy)} purchases: mean={vy.mean():.2%}, win={( vy > 0).mean():.1%}")
print(f"  Losses>{'-20%'}: {(vy < -0.2).sum()}, Gains>{'50%'}: {(vy > 0.5).sum()}")

# Score vs return correlation
from scipy import stats
corr, pval = stats.spearmanr(scores, vy) if len(scores) > 10 else (0, 1)
print(f"\n  Score-Return Spearman: {corr:.3f} (p={pval:.4f})")

# Top 10 vs Bottom 10 analysis
if len(scores) >= 20:
    top_idx = np.argsort(scores)[-10:]
    bot_idx = np.argsort(scores)[:10]
    print(f"\n  Top 10 scored: mean={vy[top_idx].mean():.2%}, win={(vy[top_idx]>0).mean():.0%}")
    print(f"  Bot 10 scored: mean={vy[bot_idx].mean():.2%}, win={(vy[bot_idx]>0).mean():.0%}")

print(f"\n  Best Sortino: {trainer.best_sortino:.2f}")
print("=" * 60)

# Cleanup
import os
cp = os.path.join(cfg.checkpoint_dir, "best_model.pth")
if os.path.exists(cp):
    os.remove(cp)
    try: os.rmdir(cfg.checkpoint_dir)
    except: pass
