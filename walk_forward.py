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
import signals

load_dotenv()

API_KEY = "PK22XEELBFYNU7QMJHJOGRJ6V6"
SECRET_KEY = "3arXWSeJW69nWfZHKW9nABMWwMkK1Ct964VakJdT7PXV"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

STOCKS_TO_TEST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "AMD", "NFLX", "CRM",
    "JPM", "BAC", "GS", "V", "SPY", "QQQ",
    "XOM", "CVX", "JNJ", "PFE", "UNH",
    "WMT", "COST", "NKE", "INTC", "QCOM",
    "MS", "MA", "NET", "SQQQ",
    "BLK", "SCHW", "CRWD", "SNOW", "HD", "SBUX"
]
STOCK = "AAPL"
PREMIUM_STOCKS = ["BLK", "AVGO", "LMT", "MA", "V", "MSFT", "GS"]  # High-priced → fewer but larger trades
STARTING_CASH = 100000
QUANTITY = 10
LOOKBACK_DAYS = 120

# ── Walk forward settings ─────────────────────────────────────────
TRAIN_DAYS = 60
TEST_DAYS = 30

# ── Risk management ───────────────────────────────────────────────
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5
ATR_PROFIT_MULT = 3.0

# ── Parameter ranges (gradient scoring — max score 7.0) ──────────
EMA_FAST_RANGE = [3, 5, 8]
EMA_SLOW_RANGE = [12, 18, 25]
RSI_BUY_RANGE = [25, 30, 35]
RSI_SELL_RANGE = [65, 70, 75]
RSI_PERIOD = 7

BB_PERIOD_RANGE = [15, 20]
BB_STD_RANGE = [1.5, 2.0]

# ── Score threshold range (max possible = 7.0) ───────────────────
SCORE_THRESHOLD_RANGE = [3.5, 4.0, 4.5, 5.0]

# ── Minimum hold time ────────────────────────────────────────────
MIN_HOLD_BARS = 5  # Hold at least 5 minutes — no whipsaw exits

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
    "INTC": 0.02, "QCOM": 0.03,
    "UBER": 0.02, "PLTR": 0.02, "COIN": 0.05, "SHOP": 0.04,
    "SQ": 0.03, "ROKU": 0.04, "ABNB": 0.04, "PYPL": 0.03,
    "SPOT": 0.03, "ZM": 0.04, "HOOD": 0.02,
    "AVGO": 0.05, "LLY": 0.08, "MA": 0.03, "PANW": 0.05,
    "CRWD": 0.05, "SNOW": 0.05, "DDOG": 0.04, "NET": 0.03,
    "ADBE": 0.05, "NOW": 0.05, "ORCL": 0.03, "MS": 0.03,
    "BLK": 0.08, "SCHW": 0.02, "HD": 0.03, "MCD": 0.03,
    "SBUX": 0.02, "SQQQ": 0.02,
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

def get_spy_daily_bars(days):
    """Fetch DAILY SPY bars for regime detection.
    Returns DataFrame with daily OHLCV, or None on failure.
    """
    end = datetime.now()
    start = end - timedelta(days=days + 30)  # Extra for EMA warmup
    try:
        bars = api.get_bars(
            "SPY",
            tradeapi.rest.TimeFrame.Day,
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            feed="iex"
        ).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs("SPY", level="symbol")
        bars.index = pd.to_datetime(bars.index)
        return bars
    except Exception as e:
        print(f"  Failed to fetch SPY daily bars: {e}")
        return None


DAILY_REGIME_EMA = 20  # 20-day EMA on daily bars (~1 month lookback)

# Inverse ETFs — flip regime gate (long in downtrend, short in uptrend)
INVERSE_REGIME_STOCKS = {"SQQQ", "UVXY"}

# Short-eligible stocks — only run bear optimization pass on these
SHORT_ELIGIBLE = {"NVDA", "AMD", "AMZN", "AAPL", "MSFT", "META", "NFLX", "COIN", "PLTR"}


def compute_daily_regime(spy_daily):
    """Compute regime from daily SPY bars.
    Returns dict mapping date -> bool (True = uptrend).
    Uses 20-day EMA: SPY close > 20-day EMA = uptrend.
    """
    close = spy_daily["close"]
    ema = close.ewm(span=DAILY_REGIME_EMA, adjust=False).mean()
    regime = close > ema
    # Convert to dict {date -> bool}
    result = {}
    for idx, val in regime.items():
        d = idx.date() if hasattr(idx, 'date') else idx
        result[d] = bool(val)
    return result


def get_regime_for_bar(daily_regime, bar_time):
    """Look up the daily regime for a given minute bar timestamp.
    Returns True (uptrend) or False (downtrend).
    Falls back to True if date not found.
    """
    d = bar_time.date() if hasattr(bar_time, 'date') else bar_time
    return daily_regime.get(d, True)

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

def calculate_atr(bars, period=ATR_PERIOD):
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def split_bars_by_regime(bars, daily_regime):
    """Split minute bars into uptrend and downtrend subsets based on daily regime."""
    bars = bars.copy()
    dates = pd.Series([idx.date() if hasattr(idx, 'date') else idx for idx in bars.index], index=bars.index)
    is_uptrend = dates.map(lambda d: daily_regime.get(d, True))
    uptrend_bars = bars[is_uptrend]
    downtrend_bars = bars[~is_uptrend]
    return uptrend_bars, downtrend_bars


# ── Run single backtest ───────────────────────────────────────────
def run_backtest(bars, daily_regime, fast, slow, rsi_buy, rsi_sell, bb_period, bb_std,
                 score_threshold=6.0, direction="long"):
    close = bars["close"]
    fast_ema = calculate_ema(close, fast)
    slow_ema = calculate_ema(close, slow)
    rsi = calculate_rsi(close, RSI_PERIOD)
    bb_upper, bb_mid, bb_lower = calculate_bb(close, bb_period, bb_std)
    atr = calculate_atr(bars)

    # Volume: 20-period rolling average
    volume = bars["volume"]
    avg_volume = volume.rolling(window=20).mean()

    # Sell/cover threshold is entry threshold - 0.5
    exit_threshold = score_threshold - 0.5

    # Inverse ETF: flip regime
    _is_inverse = STOCK in INVERSE_REGIME_STOCKS

    cash = STARTING_CASH
    position = 0       # >0 = long, <0 = short
    entry_price = 0
    entry_atr = 0
    entry_cost_stored = 0
    entry_bar = 0
    pnls = []

    is_short_mode = (direction == "short")

    start_bar = max(slow, bb_period, RSI_PERIOD) + 1
    for i in range(start_bar, len(bars)):
        price = close.iloc[i]
        raw_uptrend = get_regime_for_bar(daily_regime, bars.index[i])
        uptrend = (not raw_uptrend) if _is_inverse else raw_uptrend
        cur_volume = volume.iloc[i]
        cur_avg_volume = avg_volume.iloc[i]

        # ── Long position management ──
        if position > 0:
            stop_price = entry_price - (ATR_STOP_MULT * entry_atr)
            target_price = entry_price + (ATR_PROFIT_MULT * entry_atr)
            if price <= stop_price or price >= target_price:
                sell_cost = calculate_trade_cost(STOCK, price, position, "sell")
                pnl = (price - entry_price) * position - entry_cost_stored - sell_cost
                cash += (price * position) - sell_cost
                pnls.append(pnl)
                position = 0
                continue
            # Min hold time
            if (i - entry_bar) < MIN_HOLD_BARS:
                continue
            # Sell signal
            sell_score, _ = signals.calculate_sell_score(
                fast_ema.iloc[i], slow_ema.iloc[i],
                rsi.iloc[i], rsi_sell,
                price, bb_upper.iloc[i],
                bb_lower=bb_lower.iloc[i]
            )
            if sell_score >= exit_threshold:
                sell_cost = calculate_trade_cost(STOCK, price, abs(position), "sell")
                pnl = (price - entry_price) * position - entry_cost_stored - sell_cost
                cash += (price * position) - sell_cost
                pnls.append(pnl)
                position = 0
            continue

        # ── Short position management ──
        if position < 0:
            qty = abs(position)
            stop_price = entry_price + (ATR_STOP_MULT * entry_atr)
            target_price = entry_price - (ATR_PROFIT_MULT * entry_atr)
            if price >= stop_price or price <= target_price:
                cover_cost = calculate_trade_cost(STOCK, price, qty, "buy")
                pnl = (entry_price - price) * qty - entry_cost_stored - cover_cost
                cash += pnl
                pnls.append(pnl)
                position = 0
                continue
            # Min hold time
            if (i - entry_bar) < MIN_HOLD_BARS:
                continue
            # Cover signal (bullish reversal = exit short)
            cover_score, _ = signals.calculate_cover_score(
                fast_ema.iloc[i], slow_ema.iloc[i],
                rsi.iloc[i], rsi_buy,
                price, bb_lower.iloc[i],
                bb_upper=bb_upper.iloc[i]
            )
            if cover_score >= exit_threshold:
                cover_cost = calculate_trade_cost(STOCK, price, qty, "buy")
                pnl = (entry_price - price) * qty - entry_cost_stored - cover_cost
                cash += pnl
                pnls.append(pnl)
                position = 0
            continue

        # ── New entry signals (position == 0) ──
        if not is_short_mode:
            # Buy signal (gradient scoring)
            buy_score, _ = signals.calculate_buy_score(
                fast_ema.iloc[i], slow_ema.iloc[i],
                rsi.iloc[i], rsi_buy,
                price, bb_lower.iloc[i],
                bb_upper=bb_upper.iloc[i],
                regime_uptrend=uptrend,
                current_volume=cur_volume, avg_volume=cur_avg_volume
            )
            if (buy_score >= score_threshold and
                cash >= price * QUANTITY and
                not pd.isna(atr.iloc[i])):
                entry_cost = calculate_trade_cost(STOCK, price, QUANTITY, "buy")
                position = QUANTITY
                cash -= (price * QUANTITY) + entry_cost
                entry_price = price
                entry_atr = atr.iloc[i]
                entry_cost_stored = entry_cost
                entry_bar = i
        else:
            # Short signal (gradient scoring)
            short_score, _ = signals.calculate_short_score(
                fast_ema.iloc[i], slow_ema.iloc[i],
                rsi.iloc[i], rsi_sell,
                price, bb_upper.iloc[i],
                bb_lower=bb_lower.iloc[i],
                regime_downtrend=not uptrend,
                current_volume=cur_volume, avg_volume=cur_avg_volume
            )
            if (short_score >= score_threshold and
                cash >= price * QUANTITY and
                not pd.isna(atr.iloc[i])):
                entry_cost = calculate_trade_cost(STOCK, price, QUANTITY, "sell")
                position = -QUANTITY
                entry_price = price
                entry_atr = atr.iloc[i]
                entry_cost_stored = entry_cost
                entry_bar = i

    # Close any remaining position at end
    if position > 0:
        final_price = close.iloc[-1]
        sell_cost = calculate_trade_cost(STOCK, final_price, position, "sell")
        pnl = (final_price - entry_price) * position - entry_cost_stored - sell_cost
        cash += (final_price * position) - sell_cost
        pnls.append(pnl)
    elif position < 0:
        final_price = close.iloc[-1]
        qty = abs(position)
        cover_cost = calculate_trade_cost(STOCK, final_price, qty, "buy")
        pnl = (entry_price - final_price) * qty - entry_cost_stored - cover_cost
        cash += pnl
        pnls.append(pnl)

    total_return = ((cash - STARTING_CASH) / STARTING_CASH) * 100
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    sharpe = 0
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252)

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Quality gates — disqualify bad parameter sets
    # Bear market has fewer opportunities → relaxed thresholds, compensated by Sharpe gate
    num_trades = len(pnls)
    avg_profit = np.mean(pnls) if pnls else 0
    min_trades = 5 if STOCK in PREMIUM_STOCKS else 8
    min_pf = 1.2
    min_wr = 44
    min_avg_profit = 0.25
    min_sharpe = 0.5
    if num_trades < min_trades:
        score = -999       # Not enough data
    elif num_trades > 80:
        score = -999       # Over-trading
    elif profit_factor < min_pf:
        score = -999       # Losing strategy
    elif win_rate < min_wr:
        score = -999       # Too many losers
    elif avg_profit < min_avg_profit:
        score = -999       # Avg $/trade doesn't cover costs
    elif sharpe < min_sharpe:
        score = -999       # Risk-adjusted return too low
    else:
        score = sharpe

    return {
        "return": total_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trades": num_trades,
        "profit_factor": profit_factor,
        "score": score
    }

# ── Split data ────────────────────────────────────────────────────
def split_data(bars, test_days):
    cutoff = bars.index[-1] - timedelta(days=test_days)
    train_bars = bars[bars.index < cutoff]
    test_bars = bars[bars.index >= cutoff]
    return train_bars, test_bars

# ── Build parameter grid ──────────────────────────────────────────
def build_param_grid():
    params = []
    for fast, slow, rsi_buy, rsi_sell, bb_period, bb_std, threshold in itertools.product(
        EMA_FAST_RANGE, EMA_SLOW_RANGE,
        RSI_BUY_RANGE, RSI_SELL_RANGE,
        BB_PERIOD_RANGE, BB_STD_RANGE,
        SCORE_THRESHOLD_RANGE
    ):
        if fast < slow:
            params.append({
                "fast": fast, "slow": slow,
                "rsi_buy": rsi_buy, "rsi_sell": rsi_sell,
                "bb_period": bb_period, "bb_std": bb_std,
                "score_threshold": threshold
            })
    return params

# ── Main walk forward optimizer ───────────────────────────────────
def _optimize_pass(train_bars, test_bars, daily_regime, param_grid, direction="long", label="BULL"):
    """Run one optimization pass (bull or bear) on the given data split.
    Returns (best_params_dict, status_str) or (None, status_str)."""
    total = len(param_grid)
    print(f"\n  [{label}] Testing {total:,} combinations ({direction} direction)...")

    results = []
    for i, p in enumerate(param_grid):
        if i % 1000 == 0:
            pct = i / total * 100
            print(f"   Progress: {pct:.0f}% ({i:,}/{total:,})", flush=True)

        result = run_backtest(
            train_bars, daily_regime,
            p["fast"], p["slow"],
            p["rsi_buy"], p["rsi_sell"],
            p["bb_period"], p["bb_std"],
            p["score_threshold"],
            direction=direction
        )
        result["params"] = p
        results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    top_10 = results[:10]

    print(f"\n  [{label}] TOP 10 — TRAINING DATA ({STOCK})")
    for r in top_10[:5]:
        p = r["params"]
        print(f"    EMA {p['fast']}/{p['slow']:2} | T:{p['score_threshold']:.1f} | "
              f"{r['return']:+.2f}% | {r['trades']} trades | Sharpe:{r['sharpe']:.2f}")

    # Verify on test data
    min_test_trades = 1 if direction == "short" else 2
    min_test_sharpe = 0.5 if direction == "short" else 1.0
    verified_results = []
    for r in top_10:
        p = r["params"]
        test_result = run_backtest(
            test_bars, daily_regime,
            p["fast"], p["slow"],
            p["rsi_buy"], p["rsi_sell"],
            p["bb_period"], p["bb_std"],
            p["score_threshold"],
            direction=direction
        )
        test_result["params"] = p
        test_result["train_return"] = r["return"]
        test_result["train_sharpe"] = r["sharpe"]

        if test_result["return"] > 0 and test_result["trades"] >= min_test_trades and test_result["sharpe"] > min_test_sharpe:
            verdict = "PASS"
        elif test_result["return"] > -0.1:
            verdict = "WEAK"
        else:
            verdict = "FAIL"

        verified_results.append({**test_result, "verdict": verdict})

    passed = [r for r in verified_results if r["verdict"] == "PASS"]
    weak = [r for r in verified_results if r["verdict"] == "WEAK"]

    if passed:
        best = max(passed, key=lambda x: x["sharpe"])
        status = "PASS"
    elif weak:
        best = max(weak, key=lambda x: x["return"])
        status = "WEAK"
    else:
        print(f"  [{label}] No parameters passed for {STOCK}")
        return None, "FAIL"

    best_p = best["params"]
    print(f"  [{label}] ✅ {STOCK} {status}: EMA {best_p['fast']}/{best_p['slow']} | "
          f"T:{best_p['score_threshold']:.1f} | Test:{best['return']:+.2f}% | Sharpe:{best['sharpe']:.2f}")

    return {
        "fast": best_p["fast"], "slow": best_p["slow"],
        "rsi_buy": best_p["rsi_buy"], "rsi_sell": best_p["rsi_sell"],
        "bb_period": best_p["bb_period"], "bb_std": best_p["bb_std"],
        "score_threshold": best_p["score_threshold"],
        "train_return": best["train_return"],
        "test_return": best["return"],
        "test_sharpe": best["sharpe"],
        "test_trades": best["trades"],
        "test_win_rate": best["win_rate"],
    }, status


def run_walk_forward():
    """Dual-pass walk-forward: bull (long) + bear (short) optimization.
    Returns dict with bull_params, bear_params, and backward-compat fields."""
    global STOCK
    print(f"\nFetching data for {STOCK}...")
    bars = get_minute_bars(STOCK, LOOKBACK_DAYS)
    spy_daily = get_spy_daily_bars(LOOKBACK_DAYS)

    if bars is None or spy_daily is None:
        print(f"Failed to fetch data for {STOCK}!")
        return None

    daily_regime = compute_daily_regime(spy_daily)
    regime_days = sum(1 for v in daily_regime.values() if v)
    total_days = len(daily_regime)
    print(f"Got {len(bars)} minute bars for {STOCK}")
    print(f"Regime: {regime_days}/{total_days} uptrend days (daily 20-day EMA)")

    train_bars, test_bars = split_data(bars, TEST_DAYS)
    print(f"Train: {len(train_bars)} bars | Test: {len(test_bars)} bars")

    # Split train/test by regime for regime-specific optimization
    train_up, train_down = split_bars_by_regime(train_bars, daily_regime)
    test_up, test_down = split_bars_by_regime(test_bars, daily_regime)
    print(f"Train uptrend: {len(train_up)} bars | Train downtrend: {len(train_down)} bars")

    param_grid = build_param_grid()

    # ── BULL PASS: optimize longs on uptrend bars ──
    bull_params, bull_status = None, "INSUFFICIENT_DATA"
    if len(train_up) >= 2000:
        bull_params, bull_status = _optimize_pass(
            train_up, test_up, daily_regime, param_grid,
            direction="long", label="BULL")
    else:
        print(f"  [BULL] Skipped — only {len(train_up)} uptrend bars (need 2000+)")
        # Fallback: run bull pass on ALL bars (original behavior)
        bull_params, bull_status = _optimize_pass(
            train_bars, test_bars, daily_regime, param_grid,
            direction="long", label="BULL-ALL")

    # ── BEAR PASS: optimize shorts on downtrend bars ──
    bear_params, bear_status = None, "NOT_SHORT_ELIGIBLE"
    if STOCK in SHORT_ELIGIBLE:
        if len(train_down) >= 1000:
            bear_params, bear_status = _optimize_pass(
                train_down, test_down, daily_regime, param_grid,
                direction="short", label="BEAR")
        else:
            print(f"  [BEAR] Skipped — only {len(train_down)} downtrend bars (need 1000+)")
            bear_status = "INSUFFICIENT_DATA"

    # Build result dict
    result = {
        "bull_params": bull_params,
        "bull_status": bull_status,
        "bear_params": bear_params,
        "bear_status": bear_status,
        # Backward compat: top-level fields use bull params
        "params": bull_params if bull_params else {"score_threshold": 4.0},
        "status": bull_status,
        "return": bull_params["test_return"] if bull_params else 0,
        "sharpe": bull_params["test_sharpe"] if bull_params else 0,
        "trades": bull_params["test_trades"] if bull_params else 0,
        "win_rate": bull_params["test_win_rate"] if bull_params else 0,
        "train_return": bull_params["train_return"] if bull_params else 0,
    }

    print(f"\n{'='*80}")
    print(f"WALK-FORWARD SUMMARY FOR {STOCK}:")
    print(f"  Bull: {bull_status}" + (f" | threshold={bull_params['score_threshold']:.1f}" if bull_params else ""))
    print(f"  Bear: {bear_status}" + (f" | threshold={bear_params['score_threshold']:.1f}" if bear_params else ""))
    print(f"{'='*80}")

    return result

# ── Run all stocks ────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        all_verified = {}

        os.makedirs("walk_forward_results", exist_ok=True)

        for stock in STOCKS_TO_TEST:
            STOCK = stock
            print(f"\n{'='*80}")
            print(f"🔬 Testing {stock}...")
            print(f"{'='*80}")
            result = run_walk_forward()
            if result:
                all_verified[stock] = result
                # Save per-stock JSON (dual-regime format)
                json_out = {
                    "stock": stock,
                    "status": result["status"],
                    "bull_params": result["bull_params"],
                    "bull_status": result["bull_status"],
                    "bear_params": result["bear_params"],
                    "bear_status": result["bear_status"],
                    # Backward compat
                    "best_params": result["params"],
                    "optimal_threshold": result["params"].get("score_threshold", 4.0),
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
            print("Stocks with verified parameters:")
            for stock, result in all_verified.items():
                bull = result.get("bull_params")
                bear = result.get("bear_params")
                bull_s = result.get("bull_status", "?")
                bear_s = result.get("bear_status", "?")
                bull_str = f"T:{bull['score_threshold']:.1f} ret:{bull['test_return']:+.2f}%" if bull else "N/A"
                bear_str = f"T:{bear['score_threshold']:.1f} ret:{bear['test_return']:+.2f}%" if bear else "N/A"
                print(f"  {stock}: Bull[{bull_s}] {bull_str} | Bear[{bear_s}] {bear_str}")

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
