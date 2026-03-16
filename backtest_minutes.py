import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

api = tradeapi.REST(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    os.environ["ALPACA_BASE_URL"]
)

STOCK = "AAPL"
STARTING_CASH = 100000
QUANTITY = 10
LOOKBACK_DAYS = 90

# ── ATR-based stop loss and take profit ──────────────────────────
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5           # Stop loss = entry - 1.5 * ATR
ATR_PROFIT_MULT = 3.0         # Take profit = entry + 3.0 * ATR

# ── Regime detection settings ─────────────────────────────────────
REGIME_EMA = 50             # If price > 50 EMA = uptrend, below = downtrend

# ── Simple signal mode ──────────────────────────────────────────
# When True: use ONLY EMA crossover + RSI + ATR stops (no BB, MACD, VWAP)
# Produces 10-20x more trades for better statistical testing
SIMPLE_SIGNAL_MODE = False

# ── Parameter grid ────────────────────────────────────────────────
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

BB_PERIOD = 15
BB_STD = 1.5

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

# ── Fetch historical minute bars ──────────────────────────────────
def get_minute_bars(stock, days):
    end = datetime.now()
    start = end - timedelta(days=days)

    print(f"Fetching {days} days of minute data for {stock}...")

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
            print(f"  Chunk error: {e}")
        current_start = chunk_end

    if not all_bars:
        print("No data returned!")
        return None

    combined = pd.concat(all_bars)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()
    combined.index = pd.to_datetime(combined.index)
    combined = combined.between_time("09:30", "16:00")

    print(f"Got {len(combined)} minute bars")
    return combined

# ── Fetch SPY for regime detection ────────────────────────────────
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
    # Group by date to reset VWAP daily
    cum_tp_vol = tp_vol.groupby(tp_vol.index.date).cumsum()
    cum_vol = bars["volume"].groupby(bars["volume"].index.date).cumsum()
    vwap = cum_tp_vol / cum_vol
    return vwap

# ── Single backtest ───────────────────────────────────────────────
def run_single_backtest(bars, spy_bars, fast, slow, rsi_period, rsi_buy, rsi_sell):
    close = bars["close"]

    fast_ema = calculate_ema(close, fast)
    slow_ema = calculate_ema(close, slow)
    rsi = calculate_rsi(close, rsi_period)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    macd_line, signal_line = calculate_macd(close)
    atr = calculate_atr(bars)
    vwap = calculate_vwap(bars)

    # Regime detection using SPY
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

        # Regime filter — only buy if SPY is above its 50 EMA
        market_uptrend = spy_price > spy_ema

        # ATR-based stop loss and take profit check
        if position > 0:
            stop_price = buy_price - (ATR_STOP_MULT * buy_atr)
            target_price = buy_price + (ATR_PROFIT_MULT * buy_atr)
            if price <= stop_price:
                sell_cost = calculate_trade_cost(STOCK, price, position, "sell")
                pnl = (price - buy_price) * position - buy_cost_stored - sell_cost
                cash += (price * position) - sell_cost
                trades.append({"action": "STOP", "price": price, "pnl": pnl})
                position = 0
                last_trade_bar = i
                continue
            elif price >= target_price:
                sell_cost = calculate_trade_cost(STOCK, price, position, "sell")
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
            buy_cost = calculate_trade_cost(STOCK, price, QUANTITY, "buy")
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
            sell_cost = calculate_trade_cost(STOCK, price, position, "sell")
            pnl = (price - buy_price) * position - buy_cost_stored - sell_cost
            cash += (price * position) - sell_cost
            last_trade_bar = i
            trades.append({"action": "SELL", "price": price, "pnl": pnl})
            position = 0

    if position > 0:
        final_price = close.iloc[-1]
        sell_cost = calculate_trade_cost(STOCK, final_price, position, "sell")
        pnl = (final_price - buy_price) * position - buy_cost_stored - sell_cost
        cash += (final_price * position) - sell_cost
        trades.append({"action": "SELL", "price": final_price, "pnl": pnl})

    total_return = ((cash - STARTING_CASH) / STARTING_CASH) * 100
    completed = [t for t in trades if t["action"] != "BUY"]
    pnls = [t["pnl"] for t in completed]
    wins = [p for p in pnls if p > 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    sharpe = 0
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252)

    return {
        "total_return": total_return,
        "trades": len(completed),
        "win_rate": win_rate,
        "sharpe": sharpe,
        "cash": cash
    }

# ── Full backtest with chart ──────────────────────────────────────
def run_full_backtest(bars, spy_bars, fast, slow, rsi_period, rsi_buy, rsi_sell):
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
    portfolio_values = []
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
        date = bars.index[i]
        spy_price = spy_close.iloc[i]
        spy_ema = spy_regime_ema.iloc[i]
        macd_bullish = macd_line.iloc[i] > signal_line.iloc[i]
        macd_bearish = macd_line.iloc[i] < signal_line.iloc[i]
        cur_vwap = vwap.iloc[i]

        market_uptrend = spy_price > spy_ema
        portfolio_value = cash + (position * price)
        portfolio_values.append({"date": date, "value": portfolio_value})

        # ATR-based stop loss and take profit check
        if position > 0:
            stop_price = buy_price - (ATR_STOP_MULT * buy_atr)
            target_price = buy_price + (ATR_PROFIT_MULT * buy_atr)
            if price <= stop_price:
                sell_cost = calculate_trade_cost(STOCK, price, position, "sell")
                pnl = (price - buy_price) * position - buy_cost_stored - sell_cost
                cash += (price * position) - sell_cost
                trades.append({
                    "date": date, "action": "STOP",
                    "price": price, "rsi": rsi_val, "pnl": pnl
                })
                position = 0
                last_trade_bar = i
                continue
            elif price >= target_price:
                sell_cost = calculate_trade_cost(STOCK, price, position, "sell")
                pnl = (price - buy_price) * position - buy_cost_stored - sell_cost
                cash += (price * position) - sell_cost
                trades.append({
                    "date": date, "action": "TAKE PROFIT",
                    "price": price, "rsi": rsi_val, "pnl": pnl
                })
                position = 0
                last_trade_bar = i
                continue

        if SIMPLE_SIGNAL_MODE:
            buy_sig = (market_uptrend and
                       fast_val > slow_val and
                       rsi_val < rsi_buy)
        else:
            buy_sig = (market_uptrend and
                       fast_val > slow_val and
                       (rsi_val < rsi_buy or price <= lower) and
                       (macd_bullish or (pd.isna(cur_vwap) or price < cur_vwap)))

        if (buy_sig and
            position == 0 and
            cash >= price * QUANTITY and
            not pd.isna(atr.iloc[i])):
            buy_cost = calculate_trade_cost(STOCK, price, QUANTITY, "buy")
            position = QUANTITY
            cash -= (price * QUANTITY) + buy_cost
            buy_price = price
            buy_atr = atr.iloc[i]
            buy_cost_stored = buy_cost
            last_trade_bar = i
            trades.append({
                "date": date, "action": "BUY",
                "price": price, "rsi": rsi_val
            })

        elif position > 0:
            if SIMPLE_SIGNAL_MODE:
                sell_sig = (fast_val < slow_val and
                            rsi_val > rsi_sell)
            else:
                sell_sig = (fast_val < slow_val and
                            (rsi_val > rsi_sell or price >= upper) and
                            (macd_bearish or (pd.isna(cur_vwap) or price > cur_vwap)))

            if not sell_sig:
                continue
            sell_cost = calculate_trade_cost(STOCK, price, position, "sell")
            pnl = (price - buy_price) * position - buy_cost_stored - sell_cost
            cash += (price * position) - sell_cost
            last_trade_bar = i
            trades.append({
                "date": date, "action": "SELL",
                "price": price, "rsi": rsi_val, "pnl": pnl
            })
            position = 0

    if position > 0:
        final_price = close.iloc[-1]
        sell_cost = calculate_trade_cost(STOCK, final_price, position, "sell")
        pnl = (final_price - buy_price) * position - buy_cost_stored - sell_cost
        cash += (final_price * position) - sell_cost
        trades.append({
            "date": bars.index[-1], "action": "SELL (Close)",
            "price": final_price, "rsi": 0, "pnl": pnl
        })

    return trades, portfolio_values, cash

# ── Print metrics ─────────────────────────────────────────────────
def calculate_metrics(trades, portfolio_values, final_cash, fast, slow, rsi_period, rsi_buy, rsi_sell):
    print(f"\n{'='*60}")
    print("FULL BACKTEST RESULTS — WITH STOP LOSS + REGIME DETECTION")
    print(f"{'='*60}")

    total_return = ((final_cash - STARTING_CASH) / STARTING_CASH) * 100
    print(f"  Stock:            {STOCK}")
    print(f"  EMA:              {fast}/{slow}")
    print(f"  RSI:              Period {rsi_period} | Levels {rsi_buy}/{rsi_sell}")
    print(f"  Stop Loss:        {ATR_STOP_MULT}x ATR")
    print(f"  Take Profit:      {ATR_PROFIT_MULT}x ATR")
    print(f"  Regime Filter:    SPY > {REGIME_EMA} EMA")
    print(f"  Starting Cash:    ${STARTING_CASH:,.2f}")
    print(f"  Final Cash:       ${final_cash:,.2f}")
    print(f"  Total Return:     {total_return:+.2f}%")

    completed = [t for t in trades if t["action"] != "BUY"]
    pnls = [t["pnl"] for t in completed]

    stops = len([t for t in trades if t["action"] == "STOP"])
    take_profits = len([t for t in trades if t["action"] == "TAKE PROFIT"])
    signals = len([t for t in trades if t["action"] == "SELL"])

    if pnls:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100

        print(f"\n  Total Trades:     {len(pnls)}")
        print(f"  Signal Sells:     {signals}")
        print(f"  Take Profits:     {take_profits} 🎯")
        print(f"  Stop Losses:      {stops} 🛑")
        print(f"  Win Rate:         {win_rate:.1f}%")
        if wins:
            print(f"  Avg Win:          ${np.mean(wins):+.2f}")
        if losses:
            print(f"  Avg Loss:         ${np.mean(losses):+.2f}")
        if wins and losses:
            print(f"  Profit Factor:    {abs(sum(wins)/sum(losses)):.2f}")

        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252)
            emoji = "🔥" if sharpe > 2 else "✅" if sharpe > 1 else "⚠️" if sharpe > 0 else "🔴"
            print(f"  Sharpe Ratio:     {sharpe:.2f} {emoji}")

        values = [p["value"] for p in portfolio_values]
        peak = STARTING_CASH
        max_drawdown = 0
        for v in values:
            if v > peak:
                peak = v
            drawdown = (peak - v) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        print(f"  Max Drawdown:     -{max_drawdown:.2f}%")

    print(f"\n{'='*60}")
    print("TRADE LOG (last 20)")
    print(f"{'='*60}")
    for t in trades[-20:]:
        if t["action"] == "BUY":
            print(f"  ✅ BUY          {str(t['date'])[:16]} | ${t['price']:.2f} | RSI: {t['rsi']:.1f}")
        elif t["action"] == "TAKE PROFIT":
            print(f"  🎯 TAKE PROFIT  {str(t['date'])[:16]} | ${t['price']:.2f} | PnL: ${t['pnl']:+.2f}")
        elif t["action"] == "STOP":
            print(f"  🛑 STOP LOSS    {str(t['date'])[:16]} | ${t['price']:.2f} | PnL: ${t['pnl']:+.2f}")
        else:
            pnl = t.get("pnl", 0)
            emoji = "🟢" if pnl > 0 else "🔴"
            print(f"  {emoji} SELL SIGNAL  {str(t['date'])[:16]} | ${t['price']:.2f} | PnL: ${pnl:+.2f}")

    if portfolio_values:
        dates = [p["date"] for p in portfolio_values]
        values = [p["value"] for p in portfolio_values]

        plt.figure(figsize=(14, 6))
        plt.plot(dates, values, label="Bot Portfolio", color="green", linewidth=1)
        plt.axhline(y=STARTING_CASH, color="gray", linestyle="--",
                   label=f"Starting Cash (${STARTING_CASH:,})")

        for t in trades:
            if t["action"] == "BUY":
                plt.axvline(x=t["date"], color="blue", alpha=0.2, linewidth=0.8)
            elif t["action"] == "TAKE PROFIT":
                plt.axvline(x=t["date"], color="green", alpha=0.3, linewidth=0.8)
            elif t["action"] == "STOP":
                plt.axvline(x=t["date"], color="red", alpha=0.3, linewidth=0.8)

        plt.title(f"Backtest: {STOCK} | Stop {ATR_STOP_MULT}x ATR | TP {ATR_PROFIT_MULT}x ATR | Regime Filter ON")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("backtest_minutes.png")
        plt.show()
        print(f"\n  📊 Chart saved as backtest_minutes.png")

# ── Run everything ────────────────────────────────────────────────
print(f"📊 Fetching SPY data for regime detection...")
spy_bars = get_spy_bars(LOOKBACK_DAYS)
bars = get_minute_bars(STOCK, LOOKBACK_DAYS)

if bars is not None and spy_bars is not None:
    print(f"\n{'='*80}")
    print("GRID SEARCH — MINUTE BARS + STOP LOSS + REGIME DETECTION")
    print(f"{'='*80}")
    print(f"  {'EMA':<10} {'RSI':<10} {'Levels':<12} {'Return':<10} {'Trades':<8} {'Win%':<8} {'Sharpe':<8}")
    print(f"  {'-'*70}")

    results = []
    for p in PARAM_GRID:
        result = run_single_backtest(
            bars, spy_bars, p["fast"], p["slow"],
            p["rsi_period"], p["rsi_buy"], p["rsi_sell"]
        )
        result["params"] = p
        results.append(result)
        print(f"  EMA {p['fast']}/{p['slow']:2} | "
              f"RSI {p['rsi_period']:2} | "
              f"{p['rsi_buy']}/{p['rsi_sell']} | "
              f"{result['total_return']:+6.2f}% | "
              f"{result['trades']:4} trades | "
              f"{result['win_rate']:5.1f}% | "
              f"Sharpe: {result['sharpe']:5.2f}")

    best = max(results, key=lambda x: x["total_return"])
    best_p = best["params"]

    print(f"\n{'='*80}")
    print(f"🏆 BEST PARAMETERS:")
    print(f"   EMA:        {best_p['fast']}/{best_p['slow']}")
    print(f"   RSI Period: {best_p['rsi_period']}")
    print(f"   RSI Levels: {best_p['rsi_buy']}/{best_p['rsi_sell']}")
    print(f"   Return:     {best['total_return']:+.2f}%")
    print(f"   Trades:     {best['trades']}")
    print(f"   Win Rate:   {best['win_rate']:.1f}%")
    print(f"   Sharpe:     {best['sharpe']:.2f}")

    print(f"\nRunning full backtest with best parameters...")
    trades, portfolio_values, final_cash = run_full_backtest(
        bars, spy_bars,
        best_p["fast"], best_p["slow"],
        best_p["rsi_period"], best_p["rsi_buy"], best_p["rsi_sell"]
    )
    calculate_metrics(
        trades, portfolio_values, final_cash,
        best_p["fast"], best_p["slow"],
        best_p["rsi_period"], best_p["rsi_buy"], best_p["rsi_sell"]
    )

    # ── Cost Impact Comparison ────────────────────────────────────────
    print(f"\n{'='*80}")
    print("COST IMPACT COMPARISON")
    print(f"{'='*80}")

    # Run best params WITHOUT costs
    COST_MODEL_ENABLED = False
    no_cost_trades, _, no_cost_cash = run_full_backtest(
        bars, spy_bars,
        best_p["fast"], best_p["slow"],
        best_p["rsi_period"], best_p["rsi_buy"], best_p["rsi_sell"]
    )

    # Run best params WITH costs
    COST_MODEL_ENABLED = True
    cost_trades, _, cost_cash = run_full_backtest(
        bars, spy_bars,
        best_p["fast"], best_p["slow"],
        best_p["rsi_period"], best_p["rsi_buy"], best_p["rsi_sell"]
    )

    no_cost_return = ((no_cost_cash - STARTING_CASH) / STARTING_CASH) * 100
    cost_return = ((cost_cash - STARTING_CASH) / STARTING_CASH) * 100

    no_cost_pnls = [t["pnl"] for t in no_cost_trades if t["action"] != "BUY"]
    cost_pnls = [t["pnl"] for t in cost_trades if t["action"] != "BUY"]

    no_cost_wins = len([p for p in no_cost_pnls if p > 0]) / len(no_cost_pnls) * 100 if no_cost_pnls else 0
    cost_wins = len([p for p in cost_pnls if p > 0]) / len(cost_pnls) * 100 if cost_pnls else 0

    no_cost_sharpe = (np.mean(no_cost_pnls) / np.std(no_cost_pnls)) * np.sqrt(252) if len(no_cost_pnls) > 1 and np.std(no_cost_pnls) > 0 else 0
    cost_sharpe = (np.mean(cost_pnls) / np.std(cost_pnls)) * np.sqrt(252) if len(cost_pnls) > 1 and np.std(cost_pnls) > 0 else 0

    print(f"  {'Metric':<20} {'No Costs':<15} {'With Costs':<15} {'Impact':<15}")
    print(f"  {'-'*60}")
    print(f"  {'Return':<20} {no_cost_return:+.4f}%{'':<7} {cost_return:+.4f}%{'':<7} {cost_return - no_cost_return:+.4f}%")
    print(f"  {'Win Rate':<20} {no_cost_wins:.1f}%{'':<10} {cost_wins:.1f}%{'':<10} {cost_wins - no_cost_wins:+.1f}%")
    print(f"  {'Sharpe':<20} {no_cost_sharpe:.2f}{'':<11} {cost_sharpe:.2f}{'':<11} {cost_sharpe - no_cost_sharpe:+.2f}")
    print(f"  {'Trades':<20} {len(no_cost_pnls):<15} {len(cost_pnls):<15}")

    if cost_return > 0 and cost_sharpe > 0:
        print(f"\n  ✅ EDGE SURVIVES TRANSACTION COSTS")
    else:
        print(f"\n  🚨 EDGE DOES NOT SURVIVE TRANSACTION COSTS")
        print(f"  Strategy needs wider profit targets or fewer trades")