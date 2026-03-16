"""
Overnight Pipeline — Master Testing Script
============================================
Runs all validation phases sequentially overnight:
  Phase 1: Backtest all stocks not yet backtested
  Phase 2: Walk-forward on untested stocks
  Phase 3: Permutation test on profitable stocks
  Phase 4: Generate master summary report

Usage:
  python3 overnight_pipeline.py              # Full run
  python3 overnight_pipeline.py --skip-phase 1  # Skip phase 1
  python3 overnight_pipeline.py --dry-run    # Show what would run

Estimated runtime: 2-4 hours
"""

import os
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

API_KEY = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
BASE_URL = os.environ["ALPACA_BASE_URL"]

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

# ── Stock lists ──────────────────────────────────────────────────
ORIGINAL_26 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "AMD", "NFLX", "CRM",
    "JPM", "BAC", "GS", "V", "SPY", "QQQ",
    "XOM", "CVX", "JNJ", "PFE", "UNH",
    "WMT", "COST", "NKE", "INTC", "QCOM"
]

NEW_STOCKS = ["UBER", "PLTR", "COIN", "SHOP", "SQ", "ROKU", "ABNB", "PYPL", "SPOT", "ZM", "HOOD"]

ALL_STOCKS = ORIGINAL_26 + NEW_STOCKS

# Stocks to backtest in Phase 1 (all original 26 minus SPY which is used as regime filter)
BACKTEST_STOCKS = [
    "META", "AMD", "GS", "MSFT", "NFLX", "QCOM", "CRM", "TSLA", "AMZN", "NVDA",
    "WMT", "COST", "JPM", "BAC", "V", "QQQ", "XOM", "CVX", "JNJ", "PFE", "UNH",
    "NKE", "INTC", "GOOGL", "AAPL", "SPY"
]

# Stocks to walk-forward in Phase 2
WALK_FORWARD_STOCKS = NEW_STOCKS

# ── Settings ─────────────────────────────────────────────────────
LOOKBACK_DAYS = 90
PERMUTATION_COUNT = 500
RESULTS_DIR = "backtest_results"
SUMMARY_DIR = "results"
SKIP_IF_RECENT_HOURS = 12  # Skip stocks with results less than this many hours old

# ── Backtest settings (match backtest_minutes.py) ────────────────
STARTING_CASH = 100000
QUANTITY = 10
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5
ATR_PROFIT_MULT = 3.0
REGIME_EMA = 50
BB_PERIOD = 15
BB_STD = 1.5
SIMPLE_SIGNAL_MODE = False

# ── Transaction cost model ───────────────────────────────────────
SPREAD_ESTIMATES = {
    "AAPL": 0.02, "MSFT": 0.02, "NVDA": 0.03, "GOOGL": 0.05,
    "AMZN": 0.04, "TSLA": 0.05, "META": 0.03, "AMD": 0.03,
    "NFLX": 0.05, "CRM": 0.04, "JPM": 0.02, "BAC": 0.01,
    "GS": 0.05, "V": 0.03, "SPY": 0.01, "QQQ": 0.01,
    "XOM": 0.02, "CVX": 0.02, "JNJ": 0.02, "PFE": 0.01,
    "UNH": 0.05, "WMT": 0.02, "COST": 0.05, "NKE": 0.03,
    "INTC": 0.02, "QCOM": 0.03,
    "UBER": 0.03, "PLTR": 0.02, "COIN": 0.05, "SHOP": 0.04,
    "SQ": 0.04, "ROKU": 0.04, "ABNB": 0.04, "PYPL": 0.03,
    "SPOT": 0.04, "ZM": 0.04, "HOOD": 0.03
}
DEFAULT_SPREAD = 0.03
SLIPPAGE_MULT = 0.5
SEC_FEE_RATE = 0.0000278

# ── Parameter grid (same as backtest_minutes.py) ─────────────────
PARAM_GRID = [
    {"fast": 6,  "slow": 10, "rsi_period": 7, "rsi_buy": 35, "rsi_sell": 55},
    {"fast": 6,  "slow": 10, "rsi_period": 7, "rsi_buy": 30, "rsi_sell": 60},
    {"fast": 6,  "slow": 10, "rsi_period": 7, "rsi_buy": 40, "rsi_sell": 55},
    {"fast": 4,  "slow": 10, "rsi_period": 7, "rsi_buy": 35, "rsi_sell": 55},
    {"fast": 8,  "slow": 13, "rsi_period": 7, "rsi_buy": 35, "rsi_sell": 55},
    {"fast": 8,  "slow": 22, "rsi_period": 7, "rsi_buy": 25, "rsi_sell": 60},
    {"fast": 2,  "slow": 13, "rsi_period": 7, "rsi_buy": 35, "rsi_sell": 55},
    {"fast": 2,  "slow": 22, "rsi_period": 7, "rsi_buy": 25, "rsi_sell": 55},
    {"fast": 6,  "slow": 7,  "rsi_period": 7, "rsi_buy": 35, "rsi_sell": 55},
    {"fast": 8,  "slow": 22, "rsi_period": 7, "rsi_buy": 25, "rsi_sell": 70},
]


def calculate_trade_cost(stock, price, quantity, side):
    half_spread = SPREAD_ESTIMATES.get(stock, DEFAULT_SPREAD)
    spread_cost = half_spread * quantity
    slippage_cost = spread_cost * SLIPPAGE_MULT
    sec_fee = 0.0
    if side == "sell":
        sec_fee = price * quantity * SEC_FEE_RATE
    return spread_cost + slippage_cost + sec_fee


# ── Data fetching (chunk-based with IEX feed) ───────────────────
def fetch_minute_bars(stock, days):
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
            print(f"    Chunk error for {stock}: {e}")
        current_start = chunk_end

    if not all_bars:
        return None

    combined = pd.concat(all_bars)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()
    combined.index = pd.to_datetime(combined.index)
    combined = combined.between_time("09:30", "16:00")
    return combined


def fetch_spy_bars(days):
    return fetch_minute_bars("SPY", days)


# ── Indicators ───────────────────────────────────────────────────
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


def calculate_bollinger_bands(series, period, std):
    middle = series.rolling(window=period).mean()
    deviation = series.rolling(window=period).std()
    upper = middle + (std * deviation)
    lower = middle - (std * deviation)
    return upper, middle, lower


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


# ── Backtest engine ──────────────────────────────────────────────
def run_backtest(stock, bars, spy_bars, fast, slow, rsi_period, rsi_buy, rsi_sell):
    close = bars["close"]

    fast_ema = calculate_ema(close, fast)
    slow_ema = calculate_ema(close, slow)
    rsi = calculate_rsi(close, rsi_period)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    macd_line, signal_line = calculate_macd(close)
    atr = calculate_atr(bars)
    vwap = calculate_vwap(bars)

    spy_close = spy_bars["close"].reindex(close.index, method="ffill")
    spy_regime_ema = calculate_ema(spy_close, REGIME_EMA)

    cash = STARTING_CASH
    position = 0
    trades = []
    buy_price = 0
    buy_atr = 0
    buy_cost_stored = 0
    last_trade_bar = -10

    for i in range(max(slow, BB_PERIOD, rsi_period, REGIME_EMA, 35) + 1, len(bars)):
        if i - last_trade_bar < 2:
            continue

        price = close.iloc[i]
        fast_val = fast_ema.iloc[i]
        slow_val = slow_ema.iloc[i]
        rsi_val = rsi.iloc[i]
        upper = bb_upper.iloc[i]
        lower = bb_lower.iloc[i]
        spy_price = spy_close.iloc[i]
        spy_ema = spy_regime_ema.iloc[i]
        macd_bullish = macd_line.iloc[i] > signal_line.iloc[i]
        macd_bearish = macd_line.iloc[i] < signal_line.iloc[i]
        cur_vwap = vwap.iloc[i]

        market_uptrend = spy_price > spy_ema

        # ATR-based stop loss and take profit check
        if position > 0:
            stop_price = buy_price - (ATR_STOP_MULT * buy_atr)
            target_price = buy_price + (ATR_PROFIT_MULT * buy_atr)
            if price <= stop_price:
                sell_cost = calculate_trade_cost(stock, price, position, "sell")
                pnl = (price - buy_price) * position - buy_cost_stored - sell_cost
                cash += (price * position) - sell_cost
                trades.append({"action": "STOP", "price": price, "pnl": pnl})
                position = 0
                last_trade_bar = i
                continue
            elif price >= target_price:
                sell_cost = calculate_trade_cost(stock, price, position, "sell")
                pnl = (price - buy_price) * position - buy_cost_stored - sell_cost
                cash += (price * position) - sell_cost
                trades.append({"action": "TAKE PROFIT", "price": price, "pnl": pnl})
                position = 0
                last_trade_bar = i
                continue

        # Buy signal
        if SIMPLE_SIGNAL_MODE:
            buy_signal = (market_uptrend and
                          fast_val > slow_val and
                          rsi_val < rsi_buy)
        else:
            buy_signal = (market_uptrend and
                          fast_val > slow_val and
                          (rsi_val < rsi_buy or price <= lower) and
                          (macd_bullish or (pd.isna(cur_vwap) or price < cur_vwap)))

        if (buy_signal and
            position == 0 and
            cash >= price * QUANTITY and
            not pd.isna(atr.iloc[i])):
            buy_cost = calculate_trade_cost(stock, price, QUANTITY, "buy")
            position = QUANTITY
            cash -= (price * QUANTITY) + buy_cost
            buy_price = price
            buy_atr = atr.iloc[i]
            buy_cost_stored = buy_cost
            last_trade_bar = i
            trades.append({"action": "BUY", "price": price})

        # Sell signal
        elif position > 0:
            if SIMPLE_SIGNAL_MODE:
                sell_signal = (fast_val < slow_val and
                               rsi_val > rsi_sell)
            else:
                sell_signal = (fast_val < slow_val and
                               (rsi_val > rsi_sell or price >= upper) and
                               (macd_bearish or (pd.isna(cur_vwap) or price > cur_vwap)))

            if not sell_signal:
                continue
            sell_cost = calculate_trade_cost(stock, price, position, "sell")
            pnl = (price - buy_price) * position - buy_cost_stored - sell_cost
            cash += (price * position) - sell_cost
            last_trade_bar = i
            trades.append({"action": "SELL", "price": price, "pnl": pnl})
            position = 0

    if position > 0:
        final_price = close.iloc[-1]
        sell_cost = calculate_trade_cost(stock, final_price, position, "sell")
        pnl = (final_price - buy_price) * position - buy_cost_stored - sell_cost
        cash += (final_price * position) - sell_cost
        trades.append({"action": "SELL", "price": final_price, "pnl": pnl})

    total_return = ((cash - STARTING_CASH) / STARTING_CASH) * 100
    completed = [t for t in trades if t["action"] != "BUY"]
    pnls = [t["pnl"] for t in completed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    sharpe = 0
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252)
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else (999 if wins else 0)

    # Max drawdown
    max_drawdown = 0
    peak = STARTING_CASH
    running_cash = STARTING_CASH
    for t in trades:
        if t["action"] == "BUY":
            running_cash -= t["price"] * QUANTITY
        else:
            running_cash += t["price"] * QUANTITY
        if running_cash > peak:
            peak = running_cash
        dd = (peak - running_cash) / peak * 100 if peak > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    edge_survives = total_return > 0 and sharpe > 0

    return {
        "total_return": total_return,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "trades": len(completed),
        "edge_survives_costs": edge_survives,
        "cash": cash
    }


def is_result_recent(path, max_age_hours):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        ts = data.get("timestamp", "")
        if not ts:
            return False
        result_time = datetime.fromisoformat(ts)
        age = (datetime.now() - result_time).total_seconds() / 3600
        return age < max_age_hours
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
# PHASE 1 — Backtest all stocks
# ══════════════════════════════════════════════════════════════════
def run_phase1(spy_bars):
    print(f"\n{'=' * 80}")
    print(f"  PHASE 1 — BACKTEST ALL STOCKS")
    print(f"  Grid search across {len(PARAM_GRID)} parameter sets per stock")
    print(f"{'=' * 80}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    for idx, stock in enumerate(BACKTEST_STOCKS):
        result_path = f"{RESULTS_DIR}/{stock}.json"
        if is_result_recent(result_path, SKIP_IF_RECENT_HOURS):
            print(f"  [{idx+1}/{len(BACKTEST_STOCKS)}] {stock} — skipping (recent result exists)")
            try:
                with open(result_path) as f:
                    results[stock] = json.load(f)
            except Exception:
                pass
            continue

        print(f"  [{idx+1}/{len(BACKTEST_STOCKS)}] {stock} — fetching data...", flush=True)
        try:
            bars = fetch_minute_bars(stock, LOOKBACK_DAYS)
            if bars is None or len(bars) < 200:
                print(f"    Not enough data for {stock}, skipping")
                continue

            print(f"    Got {len(bars)} bars, running grid search...")

            best_result = None
            best_params = None

            for p in PARAM_GRID:
                result = run_backtest(
                    stock, bars, spy_bars,
                    p["fast"], p["slow"],
                    p["rsi_period"], p["rsi_buy"], p["rsi_sell"]
                )
                if best_result is None or result["sharpe"] > best_result["sharpe"]:
                    best_result = result
                    best_params = p

            out = {
                "stock": stock,
                "total_return": best_result["total_return"],
                "profit_factor": best_result["profit_factor"],
                "sharpe": best_result["sharpe"],
                "win_rate": best_result["win_rate"],
                "max_drawdown": best_result["max_drawdown"],
                "trades": best_result["trades"],
                "edge_survives_costs": best_result["edge_survives_costs"],
                "best_params": best_params,
                "timestamp": datetime.now().isoformat()
            }

            with open(result_path, "w") as f:
                json.dump(out, f, indent=2)

            results[stock] = out
            pf_emoji = "+" if best_result["profit_factor"] > 1.0 else "-"
            print(f"    Return: {best_result['total_return']:+.2f}% | "
                  f"PF: {best_result['profit_factor']:.2f} {pf_emoji} | "
                  f"Sharpe: {best_result['sharpe']:.2f} | "
                  f"Trades: {best_result['trades']} | "
                  f"Win: {best_result['win_rate']:.0f}%")

        except Exception as e:
            print(f"    ERROR: {stock} failed — {e}")
            traceback.print_exc()

    print(f"\n  Phase 1 complete: {len(results)} stocks backtested")
    profitable = [s for s, r in results.items() if r.get("profit_factor", 0) > 1.0]
    print(f"  Profitable (PF > 1.0): {len(profitable)} — {', '.join(profitable)}")
    return results


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — Walk-forward on untested stocks
# ══════════════════════════════════════════════════════════════════
def run_phase2():
    print(f"\n{'=' * 80}")
    print(f"  PHASE 2 — WALK-FORWARD OPTIMIZATION ON NEW STOCKS")
    print(f"  Testing: {', '.join(WALK_FORWARD_STOCKS)}")
    print(f"{'=' * 80}\n")

    import walk_forward

    walk_forward.LOOKBACK_DAYS = LOOKBACK_DAYS
    walk_forward.SIMPLE_SIGNAL_MODE = SIMPLE_SIGNAL_MODE

    os.makedirs("walk_forward_results", exist_ok=True)
    results = {}

    for idx, stock in enumerate(WALK_FORWARD_STOCKS):
        result_path = f"walk_forward_results/{stock}.json"
        if is_result_recent(result_path, SKIP_IF_RECENT_HOURS):
            print(f"  [{idx+1}/{len(WALK_FORWARD_STOCKS)}] {stock} — skipping (recent result exists)")
            try:
                with open(result_path) as f:
                    results[stock] = json.load(f)
            except Exception:
                pass
            continue

        print(f"  [{idx+1}/{len(WALK_FORWARD_STOCKS)}] {stock} — running walk-forward...", flush=True)
        try:
            walk_forward.STOCK = stock

            # Add spread estimate for new stocks if missing
            if stock not in walk_forward.SPREAD_ESTIMATES:
                walk_forward.SPREAD_ESTIMATES[stock] = SPREAD_ESTIMATES.get(stock, DEFAULT_SPREAD)

            result = walk_forward.run_walk_forward()
            if result:
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
                with open(result_path, "w") as f:
                    json.dump(json_out, f, indent=2)
                results[stock] = json_out
                print(f"    {stock}: {result['status']} | "
                      f"Return: {result['return']:+.2f}% | "
                      f"Sharpe: {result['sharpe']:.2f} | "
                      f"Trades: {result['trades']}")
            else:
                results[stock] = {"stock": stock, "status": "FAIL", "timestamp": datetime.now().isoformat()}
                print(f"    {stock}: FAIL — no parameters passed verification")

        except Exception as e:
            print(f"    ERROR: {stock} failed — {e}")
            traceback.print_exc()
            results[stock] = {"stock": stock, "status": "ERROR", "error": str(e), "timestamp": datetime.now().isoformat()}

    passed = [s for s, r in results.items() if r.get("status") in ("PASS", "WEAK")]
    print(f"\n  Phase 2 complete: {len(results)} stocks tested")
    print(f"  Passed walk-forward: {len(passed)} — {', '.join(passed) if passed else 'none'}")
    return results


# ══════════════════════════════════════════════════════════════════
# PHASE 3 — Permutation test on eligible stocks
# ══════════════════════════════════════════════════════════════════
def run_phase3(backtest_results, wf_results):
    print(f"\n{'=' * 80}")
    print(f"  PHASE 3 — PERMUTATION TEST")
    print(f"  {PERMUTATION_COUNT} permutations per stock")
    print(f"{'=' * 80}\n")

    # Determine which stocks to test
    perm_candidates = set()

    # From Phase 1: stocks with profit_factor > 1.0
    for stock, r in backtest_results.items():
        if r.get("profit_factor", 0) > 1.0:
            perm_candidates.add(stock)

    # From Phase 2: stocks that passed walk-forward
    for stock, r in wf_results.items():
        if r.get("status") in ("PASS", "WEAK"):
            perm_candidates.add(stock)

    # Also include any stock with existing walk-forward PASS/WEAK result
    for stock in ALL_STOCKS:
        wf_path = f"walk_forward_results/{stock}.json"
        if os.path.exists(wf_path):
            try:
                with open(wf_path) as f:
                    data = json.load(f)
                if data.get("status") in ("PASS", "WEAK") and data.get("test_trades", 0) >= 2:
                    perm_candidates.add(stock)
            except Exception:
                pass

    # Filter to only stocks with walk-forward results (required for permutation test)
    eligible = []
    for stock in sorted(perm_candidates):
        wf_path = f"walk_forward_results/{stock}.json"
        if os.path.exists(wf_path):
            try:
                with open(wf_path) as f:
                    data = json.load(f)
                if data.get("status") in ("PASS", "WEAK") and data.get("test_trades", 0) >= 2:
                    eligible.append(stock)
            except Exception:
                pass

    if not eligible:
        print("  No eligible stocks for permutation testing.")
        return {}

    # Skip stocks with recent permutation results
    to_test = []
    already_done = {}
    for stock in eligible:
        perm_path = f"permutation_test_results/{stock}.json"
        if is_result_recent(perm_path, SKIP_IF_RECENT_HOURS):
            print(f"  {stock} — skipping (recent permutation result exists)")
            try:
                with open(perm_path) as f:
                    already_done[stock] = json.load(f)
            except Exception:
                to_test.append(stock)
        else:
            to_test.append(stock)

    print(f"  Eligible: {len(eligible)} stocks")
    print(f"  Already done: {len(already_done)}")
    print(f"  To test: {len(to_test)} — {', '.join(to_test)}")
    print()

    if to_test:
        import permutation_test
        permutation_test.run_all(stocks=to_test, n_perms=PERMUTATION_COUNT)

    # Collect all permutation results
    all_perm = dict(already_done)
    for stock in to_test:
        perm_path = f"permutation_test_results/{stock}.json"
        if os.path.exists(perm_path):
            try:
                with open(perm_path) as f:
                    all_perm[stock] = json.load(f)
            except Exception:
                pass

    print(f"\n  Phase 3 complete: {len(all_perm)} stocks with permutation results")
    return all_perm


# ══════════════════════════════════════════════════════════════════
# PHASE 4 — Generate master summary report
# ══════════════════════════════════════════════════════════════════
def run_phase4(backtest_results, wf_results, perm_results):
    print(f"\n{'=' * 80}")
    print(f"  PHASE 4 — GENERATING MASTER SUMMARY REPORT")
    print(f"{'=' * 80}\n")

    os.makedirs(SUMMARY_DIR, exist_ok=True)

    # Collect data for all stocks
    all_data = {}
    for stock in ALL_STOCKS:
        data = {"stock": stock}

        # Backtest results
        bt = backtest_results.get(stock)
        if bt is None:
            bt_path = f"{RESULTS_DIR}/{stock}.json"
            if os.path.exists(bt_path):
                try:
                    with open(bt_path) as f:
                        bt = json.load(f)
                except Exception:
                    pass
        if bt:
            data["bt_return"] = bt.get("total_return", 0)
            data["bt_profit_factor"] = bt.get("profit_factor", 0)
            data["bt_sharpe"] = bt.get("sharpe", 0)
            data["bt_win_rate"] = bt.get("win_rate", 0)
            data["bt_trades"] = bt.get("trades", 0)
            data["bt_max_drawdown"] = bt.get("max_drawdown", 0)
            data["bt_edge_survives"] = bt.get("edge_survives_costs", False)

        # Walk-forward results
        wf = wf_results.get(stock)
        if wf is None:
            wf_path = f"walk_forward_results/{stock}.json"
            if os.path.exists(wf_path):
                try:
                    with open(wf_path) as f:
                        wf = json.load(f)
                except Exception:
                    pass
        if wf:
            data["wf_status"] = wf.get("status", "N/A")
            data["wf_sharpe"] = wf.get("test_sharpe", 0)
            data["wf_return"] = wf.get("test_return", 0)
            data["wf_trades"] = wf.get("test_trades", 0)

        # Permutation results
        pm = perm_results.get(stock)
        if pm is None:
            pm_path = f"permutation_test_results/{stock}.json"
            if os.path.exists(pm_path):
                try:
                    with open(pm_path) as f:
                        pm = json.load(f)
                except Exception:
                    pass
        if pm:
            data["perm_p"] = pm.get("combined_p", pm.get("day_shuffle_p", 1.0))
            data["perm_day_p"] = pm.get("day_shuffle_p", 1.0)
            data["perm_shift_p"] = pm.get("signal_shift_p", 1.0)

        all_data[stock] = data

    # Tier classification
    tier1 = []  # p < 0.05 AND profit_factor > 2.0
    tier2 = []  # p < 0.15 AND profit_factor > 1.5
    tier3 = []  # p < 0.30
    dropped = []

    for stock, d in all_data.items():
        p = d.get("perm_p", 1.0)
        pf = d.get("bt_profit_factor", 0)

        if p < 0.05 and pf > 2.0:
            tier1.append(stock)
        elif p < 0.05 and pf > 1.0:
            tier1.append(stock)  # Strong p-value even if PF not > 2
        elif p < 0.15 and pf > 1.5:
            tier2.append(stock)
        elif p < 0.15 and pf > 1.0:
            tier2.append(stock)  # Promising p with positive PF
        elif p < 0.30:
            tier3.append(stock)
        else:
            dropped.append(stock)

    # Compute Fisher combined p-value
    tested_p_values = [d.get("perm_p") for d in all_data.values() if d.get("perm_p") is not None]
    fisher_p = 1.0
    if tested_p_values:
        from scipy import stats
        p_arr = np.array(tested_p_values, dtype=float)
        p_arr = np.clip(p_arr, 0.001, 1.0)
        test_stat = -2.0 * np.sum(np.log(p_arr))
        df = 2 * len(p_arr)
        fisher_p = float(1.0 - stats.chi2.cdf(test_stat, df))

    # Build the report
    now = datetime.now()
    lines = []
    lines.append("=" * 80)
    lines.append("  OVERNIGHT PIPELINE — MASTER SUMMARY REPORT")
    lines.append(f"  Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    lines.append("")
    lines.append("-" * 80)
    lines.append("  ALL STOCKS — COMPLETE TABLE")
    lines.append("-" * 80)
    lines.append(f"  {'Stock':<7} {'BT Ret':>8} {'PF':>6} {'Sharpe':>7} {'Win%':>6} "
                 f"{'Trades':>7} {'WF':>5} {'Perm p':>8} {'Tier':>6}")
    lines.append(f"  {'-' * 72}")

    for stock in ALL_STOCKS:
        d = all_data[stock]
        bt_ret = f"{d.get('bt_return', 0):+.2f}%" if "bt_return" in d else "   N/A"
        pf = f"{d.get('bt_profit_factor', 0):.2f}" if "bt_profit_factor" in d else "  N/A"
        sh = f"{d.get('bt_sharpe', 0):.2f}" if "bt_sharpe" in d else "  N/A"
        wr = f"{d.get('bt_win_rate', 0):.0f}%" if "bt_win_rate" in d else " N/A"
        tr = f"{d.get('bt_trades', 0)}" if "bt_trades" in d else "N/A"
        wf = d.get("wf_status", "N/A")[:4]
        pp = f"{d.get('perm_p', 1.0):.4f}" if "perm_p" in d else "   N/A"

        if stock in tier1:
            tier_label = "T1"
        elif stock in tier2:
            tier_label = "T2"
        elif stock in tier3:
            tier_label = "T3"
        else:
            tier_label = "DROP"

        lines.append(f"  {stock:<7} {bt_ret:>8} {pf:>6} {sh:>7} {wr:>6} "
                     f"{tr:>7} {wf:>5} {pp:>8} {tier_label:>6}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("  TIERED RECOMMENDATIONS")
    lines.append("-" * 80)

    if tier1:
        lines.append(f"  TIER 1 — Trade full size (p < 0.05, PF > 1.0):")
        lines.append(f"    {', '.join(tier1)}")
    if tier2:
        lines.append(f"  TIER 2 — Trade reduced size (p < 0.15, PF > 1.0):")
        lines.append(f"    {', '.join(tier2)}")
    if tier3:
        lines.append(f"  TIER 3 — Monitor only (p < 0.30):")
        lines.append(f"    {', '.join(tier3)}")
    if dropped:
        lines.append(f"  DROP — No evidence of edge:")
        lines.append(f"    {', '.join(dropped)}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("  RECOMMENDED STOCKS LIST FOR bot.py")
    lines.append("-" * 80)
    lines.append(f'  TIER1_STOCKS = {json.dumps(tier1)}')
    lines.append(f'  TIER2_STOCKS = {json.dumps(tier2)}')
    lines.append(f'  STOCKS = TIER1_STOCKS + TIER2_STOCKS')

    lines.append("")
    lines.append("-" * 80)
    lines.append("  STATISTICAL SUMMARY")
    lines.append("-" * 80)
    lines.append(f"  Fisher combined p-value: {fisher_p:.6f}")
    lines.append(f"  Stocks with permutation results: {len(tested_p_values)}")
    lines.append(f"  Significant (p < 0.05): {sum(1 for p in tested_p_values if p < 0.05)}")
    lines.append(f"  Promising (p < 0.15): {sum(1 for p in tested_p_values if 0.05 <= p < 0.15)}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("  PHASE TIMING")
    lines.append("-" * 80)
    lines.append(f"  Report generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    lines.append("=" * 80)

    report = "\n".join(lines)

    # Print to console
    print(report)

    # Save to file
    summary_path = f"{SUMMARY_DIR}/overnight_summary.txt"
    with open(summary_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved to {summary_path}")

    # Also save machine-readable JSON
    json_summary = {
        "timestamp": now.isoformat(),
        "fisher_combined_p": fisher_p,
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "dropped": dropped,
        "all_data": {s: {k: v for k, v in d.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
                     for s, d in all_data.items()},
    }
    json_path = f"{SUMMARY_DIR}/overnight_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"  JSON saved to {json_path}")

    return tier1, tier2, tier3, dropped


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    skip_phases = set()
    dry_run = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--skip-phase" and i + 1 < len(args):
            skip_phases.add(int(args[i + 1]))
            i += 2
        elif args[i] == "--dry-run":
            dry_run = True
            i += 1
        else:
            print(f"Unknown argument: {args[i]}")
            print("Usage: python3 overnight_pipeline.py [--skip-phase N] [--dry-run]")
            sys.exit(1)

    start_time = time.time()

    print("=" * 80)
    print("  OVERNIGHT PIPELINE — MASTER TESTING SCRIPT")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Phases: {', '.join(str(p) for p in range(1,5) if p not in skip_phases)}")
    print("=" * 80)

    if dry_run:
        print("\n  DRY RUN — showing what would be tested:\n")
        print(f"  Phase 1 — Backtest {len(BACKTEST_STOCKS)} stocks: {', '.join(BACKTEST_STOCKS)}")
        print(f"  Phase 2 — Walk-forward {len(WALK_FORWARD_STOCKS)} stocks: {', '.join(WALK_FORWARD_STOCKS)}")
        print(f"  Phase 3 — Permutation test on profitable stocks ({PERMUTATION_COUNT} perms each)")
        print(f"  Phase 4 — Generate master summary report")
        sys.exit(0)

    # Fetch SPY bars once for all phases
    print("\nFetching SPY bars for regime detection...")
    spy_bars = fetch_spy_bars(LOOKBACK_DAYS)
    if spy_bars is None:
        print("FATAL: Could not fetch SPY bars. Exiting.")
        sys.exit(1)
    print(f"  Got {len(spy_bars)} SPY bars\n")

    backtest_results = {}
    wf_results = {}
    perm_results = {}

    # Phase 1
    if 1 not in skip_phases:
        phase1_start = time.time()
        backtest_results = run_phase1(spy_bars)
        elapsed = (time.time() - phase1_start) / 60
        print(f"  Phase 1 took {elapsed:.1f} minutes\n")
    else:
        print("\n  Phase 1 SKIPPED\n")

    # Phase 2
    if 2 not in skip_phases:
        phase2_start = time.time()
        wf_results = run_phase2()
        elapsed = (time.time() - phase2_start) / 60
        print(f"  Phase 2 took {elapsed:.1f} minutes\n")
    else:
        print("\n  Phase 2 SKIPPED\n")

    # Phase 3
    if 3 not in skip_phases:
        phase3_start = time.time()
        perm_results = run_phase3(backtest_results, wf_results)
        elapsed = (time.time() - phase3_start) / 60
        print(f"  Phase 3 took {elapsed:.1f} minutes\n")
    else:
        print("\n  Phase 3 SKIPPED\n")

    # Phase 4
    if 4 not in skip_phases:
        run_phase4(backtest_results, wf_results, perm_results)

    total_elapsed = (time.time() - start_time) / 60
    print(f"\n{'=' * 80}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {total_elapsed:.1f} minutes")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")
