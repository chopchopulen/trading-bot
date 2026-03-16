"""Diagnose which buy conditions are blocking trades."""
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = "PK22XEELBFYNU7QMJHJOGRJ6V6"
SECRET_KEY = "3arXWSeJW69nWfZHKW9nABMWwMkK1Ct964VakJdT7PXV"
BASE_URL = "https://paper-api.alpaca.markets"
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

STOCKS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]
LOOKBACK_DAYS = 30
REGIME_EMA = 50

def get_bars(stock, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    all_bars = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=7), end)
        try:
            bars = api.get_bars(stock, tradeapi.rest.TimeFrame.Minute,
                start=current.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end=chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"), feed="iex").df
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(stock, level="symbol")
            if len(bars) > 0:
                all_bars.append(bars)
        except: pass
        current = chunk_end
    if not all_bars: return None
    combined = pd.concat(all_bars)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    combined.index = pd.to_datetime(combined.index)
    return combined.between_time("09:30", "16:00")

def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()
def calc_rsi(s, p):
    d = s.diff(); g = d.where(d>0,0); l = -d.where(d<0,0)
    return 100 - (100 / (1 + g.rolling(p).mean() / l.rolling(p).mean()))
def calc_bb(s, p, std):
    m = s.rolling(p).mean(); d = s.rolling(p).std()
    return m + std*d, m, m - std*d
def calc_macd(s):
    ml = s.ewm(span=12,adjust=False).mean() - s.ewm(span=26,adjust=False).mean()
    sl = ml.ewm(span=9,adjust=False).mean()
    return ml, sl
def calc_vwap(bars):
    tp = (bars["high"]+bars["low"]+bars["close"])/3
    tpv = tp * bars["volume"]
    return tpv.groupby(tpv.index.date).cumsum() / bars["volume"].groupby(bars["volume"].index.date).cumsum()

# Get SPY for regime
print("Fetching SPY...")
spy_bars = get_bars("SPY", LOOKBACK_DAYS)
spy_close = spy_bars["close"]
spy_ema = calc_ema(spy_close, REGIME_EMA)
spy_uptrend = spy_close > spy_ema

pct_spy_uptrend = spy_uptrend.sum() / len(spy_uptrend) * 100
print(f"\nSPY REGIME: {pct_spy_uptrend:.1f}% of bars in uptrend (SPY > 50 EMA)")
print(f"  SPY last price: ${spy_close.iloc[-1]:.2f}")
print(f"  SPY 50 EMA:     ${spy_ema.iloc[-1]:.2f}")
print()

# Test params: use the ones from bot.py
FAST, SLOW = 6, 10
RSI_PERIOD, RSI_BUY = 7, 35
BB_PERIOD, BB_STD = 15, 1.5

for stock in STOCKS:
    print(f"{'='*70}")
    print(f"DIAGNOSING {stock} (EMA {FAST}/{SLOW}, RSI<{RSI_BUY}, BB {BB_PERIOD}/{BB_STD})")
    print(f"{'='*70}")
    
    bars = get_bars(stock, LOOKBACK_DAYS)
    if bars is None:
        print(f"  No data for {stock}")
        continue
    
    close = bars["close"]
    fast_ema = calc_ema(close, FAST)
    slow_ema = calc_ema(close, SLOW)
    rsi = calc_rsi(close, RSI_PERIOD)
    bb_upper, bb_mid, bb_lower = calc_bb(close, BB_PERIOD, BB_STD)
    macd_line, signal_line = calc_macd(close)
    vwap = calc_vwap(bars)
    
    # Align SPY to stock bars
    spy_aligned = spy_uptrend.reindex(close.index, method="ffill")
    
    start_idx = max(SLOW, BB_PERIOD, RSI_PERIOD, REGIME_EMA, 35) + 1
    total_bars = len(bars) - start_idx
    
    # Count each condition independently
    c1 = spy_aligned.iloc[start_idx:] == True
    c2 = fast_ema.iloc[start_idx:] > slow_ema.iloc[start_idx:]
    c3 = rsi.iloc[start_idx:] < RSI_BUY
    c4 = close.iloc[start_idx:] <= bb_lower.iloc[start_idx:]
    c5 = macd_line.iloc[start_idx:] > signal_line.iloc[start_idx:]
    c6 = (vwap.iloc[start_idx:].isna()) | (close.iloc[start_idx:] < vwap.iloc[start_idx:])
    
    print(f"  Total scannable bars: {total_bars}")
    print(f"  ---")
    print(f"  1. SPY uptrend:         {c1.sum():5d} bars ({c1.sum()/total_bars*100:5.1f}%)")
    print(f"  2. Fast EMA > Slow EMA: {c2.sum():5d} bars ({c2.sum()/total_bars*100:5.1f}%)")
    print(f"  3. RSI < {RSI_BUY}:            {c3.sum():5d} bars ({c3.sum()/total_bars*100:5.1f}%)")
    print(f"  4. Price <= BB lower:   {c4.sum():5d} bars ({c4.sum()/total_bars*100:5.1f}%)")
    print(f"  5. MACD bullish:        {c5.sum():5d} bars ({c5.sum()/total_bars*100:5.1f}%)")
    print(f"  6. Price < VWAP:        {c6.sum():5d} bars ({c6.sum()/total_bars*100:5.1f}%)")
    print(f"  ---")
    
    # Progressive AND
    and_12 = c1 & c2
    and_123 = and_12 & c3
    and_1234 = and_123 & c4
    and_12345 = and_1234 & c5
    and_all = and_12345 & c6
    
    print(f"  1+2 (regime+EMA):       {and_12.sum():5d} bars ({and_12.sum()/total_bars*100:5.1f}%)")
    print(f"  1+2+3 (+RSI):           {and_123.sum():5d} bars ({and_123.sum()/total_bars*100:5.1f}%)")
    print(f"  1+2+3+4 (+BB):          {and_1234.sum():5d} bars ({and_1234.sum()/total_bars*100:5.1f}%)")
    print(f"  1+2+3+4+5 (+MACD):      {and_12345.sum():5d} bars ({and_12345.sum()/total_bars*100:5.1f}%)")
    print(f"  ALL 6 conditions:       {and_all.sum():5d} bars ({and_all.sum()/total_bars*100:5.1f}%)")
    print()
    
    # Also test relaxed: RSI < 45
    c3_relaxed = rsi.iloc[start_idx:] < 45
    relaxed = c1 & c2 & c3_relaxed & c4 & c5 & c6
    print(f"  [RELAXED RSI<45]:       {relaxed.sum():5d} bars")
    
    # Test without EMA requirement
    no_ema = c1 & c3 & c4 & c5 & c6
    print(f"  [NO EMA requirement]:   {no_ema.sum():5d} bars")
    
    # Test without regime
    no_regime = c2 & c3 & c4 & c5 & c6
    print(f"  [NO regime filter]:     {no_regime.sum():5d} bars")
    
    # RSI distribution
    rsi_valid = rsi.iloc[start_idx:]
    print(f"\n  RSI stats: min={rsi_valid.min():.1f} max={rsi_valid.max():.1f} mean={rsi_valid.mean():.1f} median={rsi_valid.median():.1f}")
    print(f"  RSI < 30: {(rsi_valid<30).sum()} | RSI < 35: {(rsi_valid<35).sum()} | RSI < 40: {(rsi_valid<40).sum()} | RSI < 45: {(rsi_valid<45).sum()}")
    print()

