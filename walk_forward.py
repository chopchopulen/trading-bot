import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import itertools
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from collections import Counter

load_dotenv()

API_KEY = "PK22XEELBFYNU7QMJHJOGRJ6V6"
SECRET_KEY = "3arXWSeJW69nWfZHKW9nABMWwMkK1Ct964VakJdT7PXV"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

STOCKS_TO_TEST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "AMD", "NFLX", "CRM",
    "JPM", "BAC", "GS", "V", "SPY", "QQQ",
    "XOM", "CVX", "JNJ", "PFE", "UNH",
    "WMT", "COST", "NKE", "INTC", "QCOM"
]
STOCK = "AAPL"
STARTING_CASH = 100000
QUANTITY = 10
LOOKBACK_DAYS = 60

# ── Walk forward settings ─────────────────────────────────────────
TRAIN_DAYS = 40
TEST_DAYS = 20

# ── Risk management ───────────────────────────────────────────────
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5
ATR_PROFIT_MULT = 3.0
REGIME_EMA = 50

# ── Fine grained parameter ranges ────────────────────────────────
# ── Stage 1 — EMA + RSI (coarse) ─────────────────────────────────
EMA_FAST_RANGE = range(2, 10, 2)      # 2, 4, 6, 8
EMA_SLOW_RANGE = range(7, 25, 3)      # 7, 10, 13, 16, 19, 22
RSI_BUY_RANGE = range(30, 50, 5)      # 30, 35, 40, 45
RSI_SELL_RANGE = range(50, 70, 5)     # 50, 55, 60, 65
RSI_PERIOD = 7

# ── Stage 2 — BB (tested separately after EMA+RSI found) ─────────
BB_PERIOD_RANGE = [15, 20, 25]
BB_STD_RANGE = [1.5, 2.0, 2.5]

# ── Transaction cost model ───────────────────────────────────────
COST_MODEL_ENABLED = True

# Estimated half-spread per stock (dollars) — typical IEX minute bar fills
SPREAD_ESTIMATES = {
    "AAPL": 0.02, "MSFT": 0.02, "NVDA": 0.03, "GOOGL": 0.05,
    "AMZN": 0.04, "TSLA": 0.05, "META": 0.03, "AMD": 0.03,
    "NFLX": 0.05, "CRM": 0.04, "JPM": 0.02, "BAC": 0.01,
    "GS": 0.05, "V": 0.03, "SPY": 0.01, "QQQ": 0.01,
    "XOM": 0.02, "CVX": 0.02, "JNJ": 0.02, "PFE": 0.01,
    "UNH": 0.05, "WMT": 0.02, "COST": 0.05, "NKE": 0.03,
    "INTC": 0.02, "QCOM": 0.03
}
DEFAULT_SPREAD = 0.03
SLIPPAGE_MULT = 0.5     # 50% additional slippage on top of spread
SEC_FEE_RATE = 0.0000278 # SEC fee on sells: $0.00278 per $100

def calculate_trade_cost(stock, price, quantity, side):
    """Calculate total transaction cost for a single trade."""
    if not COST_MODEL_ENABLED:
        return 0.0
    half_spread = SPREAD_ESTIMATES.get(stock, DEFAULT_SPREAD)
    spread_cost = half_spread * quantity
    slippage_cost = spread_cost * SLIPPAGE_MULT
    sec_fee = 0.0
    if side == "sell":
        sec_fee = price * quantity * SEC_FEE_RATE
    return spread_cost + slippage_cost + sec_fee

# ── Kelly optimization ranges ─────────────────────────────────────
KELLY_FRACTION_RANGE = [0.1, 0.15, 0.2, 0.25, 0.5]

# ── Fetch minute bars ─────────────────────────────────────────────
def get_minute_bars(stock, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    all_bars = []
    current_start = start

    while current_start < end:
        chunk_end = min(current_start + timedelta(days=7), end)
        try:
            bars = api.get_bars(
                stock,
                tradeapi.rest.TimeFrame.Minute,
                start=current_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end=chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                feed="iex"
            ).df
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(stock, level="symbol")
            if len(bars) > 0:
                all_bars.append(bars)
        except Exception as e:
            print(f"  Chunk error for {stock}: {e}")
        current_start = chunk_end

    if not all_bars:
        return None

    combined = pd.concat(all_bars)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()
    combined.index = pd.to_datetime(combined.index)
    combined = combined.between_time("09:30", "16:00")
    return combined

def get_spy_bars(days):
    end = datetime.now()
    start = end - timedelta(days=days)
    all_bars = []
    current_start = start

    while current_start < end:
        chunk_end = min(current_start + timedelta(days=7), end)
        try:
            bars = api.get_bars(
                "SPY",
                tradeapi.rest.TimeFrame.Minute,
                start=current_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end=chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                feed="iex"
            ).df
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs("SPY", level="symbol")
            if len(bars) > 0:
                all_bars.append(bars)
        except Exception as e:
            pass
        current_start = chunk_end

    if not all_bars:
        return None

    combined = pd.concat(all_bars)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()
    combined.index = pd.to_datetime(combined.index)
    combined = combined.between_time("09:30", "16:00")
    return combined

# ── Calculate indicators ──────────────────────────────────────────
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_bb(series, period, std):
    middle = series.rolling(window=period).mean()
    deviation = series.rolling(window=period).std()
    return middle + (std * deviation), middle, middle - (std * deviation)

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_atr(bars, period=ATR_PERIOD):
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_vwap(bars):
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    tp_vol = typical_price * bars["volume"]
    cum_tp_vol = tp_vol.groupby(tp_vol.index.date).cumsum()
    cum_vol = bars["volume"].groupby(bars["volume"].index.date).cumsum()
    return cum_tp_vol / cum_vol

# ── Run single backtest ───────────────────────────────────────────
def run_backtest(bars, spy_bars, fast, slow, rsi_buy, rsi_sell, bb_period, bb_std):
    close = bars["close"]
    fast_ema = calculate_ema(close, fast)
    slow_ema = calculate_ema(close, slow)
    rsi = calculate_rsi(close, RSI_PERIOD)
    bb_upper, bb_mid, bb_lower = calculate_bb(close, bb_period, bb_std)
    macd_line, signal_line = calculate_macd(close)
    atr = calculate_atr(bars)
    vwap = calculate_vwap(bars)

    spy_close = spy_bars["close"].reindex(close.index, method="ffill")
    spy_ema = calculate_ema(spy_close, REGIME_EMA)

    cash = STARTING_CASH
    position = 0
    buy_price = 0
    buy_atr = 0
    buy_cost_stored = 0
    pnls = []
    last_trade = -10

    for i in range(max(slow, bb_period, RSI_PERIOD, REGIME_EMA, 35) + 1, len(bars)):
        if i - last_trade < 2:
            continue

        price = close.iloc[i]
        uptrend = spy_close.iloc[i] > spy_ema.iloc[i]
        macd_bullish = macd_line.iloc[i] > signal_line.iloc[i]
        macd_bearish = macd_line.iloc[i] < signal_line.iloc[i]
        cur_vwap = vwap.iloc[i]

        # ATR-based stop loss and take profit
        if position > 0:
            stop_price = buy_price - (ATR_STOP_MULT * buy_atr)
            target_price = buy_price + (ATR_PROFIT_MULT * buy_atr)
            if price <= stop_price or price >= target_price:
                sell_cost = calculate_trade_cost(STOCK, price, position, "sell")
                pnl = (price - buy_price) * position - buy_cost_stored - sell_cost
                cash += (price * position) - sell_cost
                pnls.append(pnl)
                position = 0
                last_trade = i
                continue

        # Buy — matches live bot: OR logic for RSI/BB + (MACD OR VWAP)
        if (uptrend and
            fast_ema.iloc[i] > slow_ema.iloc[i] and
            (rsi.iloc[i] < rsi_buy or price <= bb_lower.iloc[i]) and
            (macd_bullish or (pd.isna(cur_vwap) or price < cur_vwap)) and
            position == 0 and cash >= price * QUANTITY and
            not pd.isna(atr.iloc[i])):
            buy_cost = calculate_trade_cost(STOCK, price, QUANTITY, "buy")
            position = QUANTITY
            cash -= (price * QUANTITY) + buy_cost
            buy_price = price
            buy_atr = atr.iloc[i]
            buy_cost_stored = buy_cost
            last_trade = i

        # Sell — matches live bot: OR logic for RSI/BB + (MACD OR VWAP)
        elif (fast_ema.iloc[i] < slow_ema.iloc[i] and
              (rsi.iloc[i] > rsi_sell or price >= bb_upper.iloc[i]) and
              (macd_bearish or (pd.isna(cur_vwap) or price > cur_vwap)) and
              position > 0):
            sell_cost = calculate_trade_cost(STOCK, price, position, "sell")
            pnl = (price - buy_price) * position - buy_cost_stored - sell_cost
            cash += (price * position) - sell_cost
            pnls.append(pnl)
            position = 0
            last_trade = i

    if position > 0:
        final_price = close.iloc[-1]
        sell_cost = calculate_trade_cost(STOCK, final_price, position, "sell")
        pnl = (final_price - buy_price) * position - buy_cost_stored - sell_cost
        cash += (final_price * position) - sell_cost
        pnls.append(pnl)

    total_return = ((cash - STARTING_CASH) / STARTING_CASH) * 100
    wins = [p for p in pnls if p > 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    sharpe = 0
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252)

    return {
        "return": total_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trades": len(pnls),
        "score": sharpe if len(pnls) >= 3 else -999
    }

# ── Split data ────────────────────────────────────────────────────
def split_data(bars, spy_bars, test_days):
    cutoff = bars.index[-1] - timedelta(days=test_days)
    train_bars = bars[bars.index < cutoff]
    test_bars = bars[bars.index >= cutoff]
    train_spy = spy_bars[spy_bars.index < cutoff]
    test_spy = spy_bars[spy_bars.index >= cutoff]
    return train_bars, test_bars, train_spy, test_spy

# ── Build parameter grid ──────────────────────────────────────────
def build_param_grid():
    params = []
    for fast, slow, rsi_buy, rsi_sell, bb_period, bb_std in itertools.product(
        EMA_FAST_RANGE, EMA_SLOW_RANGE,
        RSI_BUY_RANGE, RSI_SELL_RANGE,
        BB_PERIOD_RANGE, BB_STD_RANGE
    ):
        if fast < slow:
            params.append({
                "fast": fast, "slow": slow,
                "rsi_buy": rsi_buy, "rsi_sell": rsi_sell,
                "bb_period": bb_period, "bb_std": bb_std
            })
    return params

# ── Main walk forward optimizer ───────────────────────────────────
def run_walk_forward():
    global STOCK
    print(f"\nFetching data for {STOCK}...")
    bars = get_minute_bars(STOCK, LOOKBACK_DAYS)
    spy_bars = get_spy_bars(LOOKBACK_DAYS)

    if bars is None or spy_bars is None:
        print(f"Failed to fetch data for {STOCK}!")
        return None

    print(f"Got {len(bars)} minute bars for {STOCK}")

    train_bars, test_bars, train_spy, test_spy = split_data(
        bars, spy_bars, TEST_DAYS
    )

    print(f"Train: {len(train_bars)} bars | Test: {len(test_bars)} bars")

    param_grid = build_param_grid()
    total = len(param_grid)
    print(f"\n🔍 Testing {total:,} combinations on training data...")

    results = []
    for i, p in enumerate(param_grid):
        if i % 1000 == 0:
            pct = i / total * 100
            print(f"   Progress: {pct:.0f}% ({i:,}/{total:,})", flush=True)

        result = run_backtest(
            train_bars, train_spy,
            p["fast"], p["slow"],
            p["rsi_buy"], p["rsi_sell"],
            p["bb_period"], p["bb_std"]
        )
        result["params"] = p
        results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    top_10 = results[:10]

    print(f"\n{'='*80}")
    print(f"TOP 10 PARAMETER SETS — TRAINING DATA ({STOCK})")
    print(f"{'='*80}")
    print(f"  {'EMA':<10} {'RSI':<12} {'BB':<12} {'Return':<10} {'Trades':<8} {'Win%':<8} {'Sharpe':<8}")
    print(f"  {'-'*75}")

    for r in top_10:
        p = r["params"]
        print(f"  EMA {p['fast']}/{p['slow']:2} | "
              f"{p['rsi_buy']}/{p['rsi_sell']} | "
              f"BB {p['bb_period']}/{p['bb_std']} | "
              f"{r['return']:+6.2f}% | "
              f"{r['trades']:4} trades | "
              f"{r['win_rate']:5.1f}% | "
              f"Sharpe: {r['sharpe']:5.2f}")

    print(f"\n{'='*80}")
    print(f"VERIFYING TOP 10 ON UNSEEN TEST DATA ({STOCK})")
    print(f"{'='*80}")
    print(f"  {'EMA':<10} {'RSI':<12} {'BB':<12} {'Train':<10} {'Test':<10} {'Verdict':<10}")
    print(f"  {'-'*70}")

    verified_results = []
    for r in top_10:
        p = r["params"]
        test_result = run_backtest(
            test_bars, test_spy,
            p["fast"], p["slow"],
            p["rsi_buy"], p["rsi_sell"],
            p["bb_period"], p["bb_std"]
        )
        test_result["params"] = p
        test_result["train_return"] = r["return"]
        test_result["train_sharpe"] = r["sharpe"]

        if test_result["return"] > 0 and test_result["trades"] >= 2 and test_result["sharpe"] > 1.0:
            verdict = "✅ PASS"
        elif test_result["return"] > -0.1:
            verdict = "⚠️ WEAK"
        else:
            verdict = "❌ FAIL"

        verified_results.append({**test_result, "verdict": verdict})

        print(f"  EMA {p['fast']}/{p['slow']:2} | "
              f"{p['rsi_buy']}/{p['rsi_sell']} | "
              f"BB {p['bb_period']}/{p['bb_std']} | "
              f"Train: {r['return']:+.2f}% | "
              f"Test: {test_result['return']:+.2f}% | "
              f"{verdict}")

    passed = [r for r in verified_results if r["verdict"] == "✅ PASS"]
    weak = [r for r in verified_results if r["verdict"] == "⚠️ WEAK"]

    if passed:
        best = max(passed, key=lambda x: x["sharpe"])
        status = "PASS"
    elif weak:
        best = max(weak, key=lambda x: x["return"])
        status = "WEAK"
    else:
        print(f"\n⚠️ No parameters passed for {STOCK}")
        print(f"   Market conditions too different between train/test periods")
        return None

    best_p = best["params"]
    print(f"\n{'='*80}")
    print(f"🏆 BEST VERIFIED PARAMETERS FOR {STOCK} ({status}):")
    print(f"{'='*80}")
    print(f"   EMA Fast:    {best_p['fast']}")
    print(f"   EMA Slow:    {best_p['slow']}")
    print(f"   RSI Buy:     {best_p['rsi_buy']}")
    print(f"   RSI Sell:    {best_p['rsi_sell']}")
    print(f"   BB Period:   {best_p['bb_period']}")
    print(f"   BB Std:      {best_p['bb_std']}")
    print(f"   Train Return:{best['train_return']:+.2f}%")
    print(f"   Test Return: {best['return']:+.2f}%")
    print(f"   Test Sharpe: {best['sharpe']:.2f}")
    print(f"   Win Rate:    {best['win_rate']:.1f}%")

    best["status"] = status
    return best

# ── Run all stocks ────────────────────────────────────────────────
try:
    all_verified = {}
    spy_bars_cache = None

    os.makedirs("walk_forward_results", exist_ok=True)

    for stock in STOCKS_TO_TEST:
        STOCK = stock
        print(f"\n{'='*80}")
        print(f"🔬 Testing {stock}...")
        print(f"{'='*80}")
        result = run_walk_forward()
        if result:
            all_verified[stock] = result
            # Save per-stock JSON
            json_out = {
                "stock": stock,
                "status": result["status"],
                "best_params": result["params"],
                "train_return": result["train_return"],
                "test_return": result["return"],
                "test_sharpe": result["sharpe"],
                "test_trades": result["trades"],
                "test_win_rate": result["win_rate"],
                "timestamp": datetime.now().isoformat()
            }
            with open(f"walk_forward_results/{stock}.json", "w") as f:
                json.dump(json_out, f, indent=2)

    # Cross stock summary
    print(f"\n{'='*80}")
    print("CROSS STOCK SUMMARY")
    print(f"{'='*80}")

    if all_verified:
        print("✅ Stocks with verified parameters:")
        for stock, result in all_verified.items():
            p = result["params"]
            print(f"  {stock}: EMA {p['fast']}/{p['slow']} | "
                  f"RSI {p['rsi_buy']}/{p['rsi_sell']} | "
                  f"BB {p['bb_period']}/{p['bb_std']} | "
                  f"Return: {result['return']:+.2f}% | "
                  f"Sharpe: {result['sharpe']:.2f}")

        # Find most common EMA settings across stocks
        ema_combos = [f"{r['params']['fast']}/{r['params']['slow']}"
                     for r in all_verified.values()]
        most_common_ema = Counter(ema_combos).most_common(1)[0]

        print(f"\n📋 RECOMMENDED SETTINGS FOR bot.py:")
        print(f"   Most consistent EMA: {most_common_ema[0]} "
              f"(appeared in {most_common_ema[1]}/{len(all_verified)} stocks)")

        if len(all_verified) > 0:
            best_overall = max(all_verified.values(), key=lambda x: x["return"])
            best_p = best_overall["params"]
            print(f"\n   Best single performer settings:")
            print(f"   FAST_MA = {best_p['fast']}")
            print(f"   SLOW_MA = {best_p['slow']}")
            print(f"   RSI_OVERSOLD = {best_p['rsi_buy']}")
            print(f"   RSI_OVERBOUGHT = {best_p['rsi_sell']}")
            print(f"   BB_PERIOD = {best_p['bb_period']}")
            print(f"   BB_STD = {best_p['bb_std']}")

    else:
        print("  ⚠️ No stocks passed verification")
        print("  Market is in an unusual period — keep current settings")
        print("  Recommendation: Re-run optimizer when market stabilizes")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
