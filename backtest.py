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

STOCK = "TSLA"
STOCK = "NVDA" 
STOCK = "MSFT"
STARTING_CASH = 100000
QUANTITY = 10
LOOKBACK_DAYS = 730

# ── Parameter grid to test ────────────────────────────────────────
PARAM_GRID = [
    {"fast": 3,  "slow": 9,  "rsi_period": 7,  "rsi_buy": 35, "rsi_sell": 65},
    {"fast": 3,  "slow": 9,  "rsi_period": 7,  "rsi_buy": 40, "rsi_sell": 60},
    {"fast": 3,  "slow": 9,  "rsi_period": 7,  "rsi_buy": 45, "rsi_sell": 55},
    {"fast": 5,  "slow": 15, "rsi_period": 7,  "rsi_buy": 35, "rsi_sell": 65},
    {"fast": 5,  "slow": 15, "rsi_period": 7,  "rsi_buy": 40, "rsi_sell": 60},
    {"fast": 5,  "slow": 15, "rsi_period": 14, "rsi_buy": 35, "rsi_sell": 65},
    {"fast": 8,  "slow": 21, "rsi_period": 7,  "rsi_buy": 35, "rsi_sell": 65},
    {"fast": 8,  "slow": 21, "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70},
    {"fast": 9,  "slow": 21, "rsi_period": 7,  "rsi_buy": 40, "rsi_sell": 60},
    {"fast": 9,  "slow": 21, "rsi_period": 14, "rsi_buy": 35, "rsi_sell": 65},
]

BB_PERIOD = 20
BB_STD = 2

# ── Fetch historical bars ─────────────────────────────────────────
def get_historical_bars(stock, days):
    end = datetime.now()
    start = end - timedelta(days=days)

    print(f"Fetching {days} days of data for {stock}...")

    bars = api.get_bars(
        stock,
        tradeapi.rest.TimeFrame.Day,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        feed="iex"
    ).df

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(stock, level="symbol")

    print(f"Got {len(bars)} daily bars")
    return bars

# ── Calculate EMA ─────────────────────────────────────────────────
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

# ── Calculate RSI ─────────────────────────────────────────────────
def calculate_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ── Calculate Bollinger Bands ─────────────────────────────────────
def calculate_bollinger_bands(series, period, std):
    middle = series.rolling(window=period).mean()
    deviation = series.rolling(window=period).std()
    upper = middle + (std * deviation)
    lower = middle - (std * deviation)
    return upper, middle, lower

# ── Single backtest run ───────────────────────────────────────────
def run_single_backtest(bars, fast, slow, rsi_period, rsi_buy, rsi_sell):
    close = bars["close"]

    fast_ema = calculate_ema(close, fast)
    slow_ema = calculate_ema(close, slow)
    rsi = calculate_rsi(close, rsi_period)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)

    cash = STARTING_CASH
    position = 0
    trades = []
    buy_price = 0

    for i in range(max(slow, BB_PERIOD, rsi_period) + 1, len(bars)):
        price = close.iloc[i]
        fast_val = fast_ema.iloc[i]
        slow_val = slow_ema.iloc[i]
        rsi_val = rsi.iloc[i]
        upper = bb_upper.iloc[i]
        lower = bb_lower.iloc[i]

        if (fast_val > slow_val and
            (rsi_val < rsi_buy or price <= lower) and
            position == 0 and
            cash >= price * QUANTITY):
            position = QUANTITY
            cash -= price * QUANTITY
            buy_price = price
            trades.append({"action": "BUY", "price": price})

        elif (fast_val < slow_val and
              (rsi_val > rsi_sell or price >= upper) and
              position > 0):
            pnl = (price - buy_price) * position
            cash += price * position
            trades.append({"action": "SELL", "price": price, "pnl": pnl})
            position = 0

    if position > 0:
        final_price = close.iloc[-1]
        pnl = (final_price - buy_price) * position
        cash += final_price * position
        trades.append({"action": "SELL", "price": final_price, "pnl": pnl})

    total_return = ((cash - STARTING_CASH) / STARTING_CASH) * 100
    completed = [t for t in trades if t["action"] == "SELL"]
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
def run_backtest(bars, fast, slow, rsi_period, rsi_buy, rsi_sell):
    close = bars["close"]

    fast_ema = calculate_ema(close, fast)
    slow_ema = calculate_ema(close, slow)
    rsi = calculate_rsi(close, rsi_period)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)

    cash = STARTING_CASH
    position = 0
    trades = []
    portfolio_values = []
    buy_price = 0

    for i in range(max(slow, BB_PERIOD, rsi_period) + 1, len(bars)):
        price = close.iloc[i]
        fast_val = fast_ema.iloc[i]
        slow_val = slow_ema.iloc[i]
        rsi_val = rsi.iloc[i]
        upper = bb_upper.iloc[i]
        lower = bb_lower.iloc[i]
        date = bars.index[i]

        portfolio_value = cash + (position * price)
        portfolio_values.append({"date": date, "value": portfolio_value})

        if (fast_val > slow_val and
            (rsi_val < rsi_buy or price <= lower) and
            position == 0 and
            cash >= price * QUANTITY):
            position = QUANTITY
            cash -= price * QUANTITY
            buy_price = price
            trades.append({
                "date": date,
                "action": "BUY",
                "price": price,
                "rsi": rsi_val
            })

        elif (fast_val < slow_val and
              (rsi_val > rsi_sell or price >= upper) and
              position > 0):
            pnl = (price - buy_price) * position
            cash += price * position
            trades.append({
                "date": date,
                "action": "SELL",
                "price": price,
                "rsi": rsi_val,
                "pnl": pnl
            })
            position = 0

    if position > 0:
        final_price = close.iloc[-1]
        pnl = (final_price - buy_price) * position
        cash += final_price * position
        trades.append({
            "date": bars.index[-1],
            "action": "SELL (Close)",
            "price": final_price,
            "rsi": 0,
            "pnl": pnl
        })

    return trades, portfolio_values, cash

# ── Print metrics and chart ───────────────────────────────────────
def calculate_metrics(trades, portfolio_values, final_cash):
    print(f"\n{'='*60}")
    print("FULL BACKTEST RESULTS")
    print(f"{'='*60}")

    total_return = ((final_cash - STARTING_CASH) / STARTING_CASH) * 100
    print(f"  Starting Cash:    ${STARTING_CASH:,.2f}")
    print(f"  Final Cash:       ${final_cash:,.2f}")
    print(f"  Total Return:     {total_return:+.2f}%")

    completed = [t for t in trades if "SELL" in t["action"]]
    pnls = [t["pnl"] for t in completed]

    if pnls:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100

        print(f"  Total Trades:     {len(pnls)}")
        print(f"  Win Rate:         {win_rate:.1f}%")
        print(f"  Avg Win:          ${np.mean(wins):+.2f}" if wins else "")
        print(f"  Avg Loss:         ${np.mean(losses):+.2f}" if losses else "")

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
    print("TRADE LOG")
    print(f"{'='*60}")
    for t in trades:
        if t["action"] == "BUY":
            print(f"  ✅ BUY  {str(t['date'])[:10]} | ${t['price']:.2f} | RSI: {t['rsi']:.1f}")
        else:
            pnl = t.get("pnl", 0)
            emoji = "🟢" if pnl > 0 else "🔴"
            print(f"  {emoji} SELL {str(t['date'])[:10]} | ${t['price']:.2f} | PnL: ${pnl:+.2f}")

    if portfolio_values:
        dates = [p["date"] for p in portfolio_values]
        values = [p["value"] for p in portfolio_values]

        plt.figure(figsize=(14, 6))
        plt.plot(dates, values, label="Bot Portfolio", color="green", linewidth=2)
        plt.axhline(y=STARTING_CASH, color="gray", linestyle="--",
                   label=f"Starting Cash (${STARTING_CASH:,})")

        for t in trades:
            if t["action"] == "BUY":
                plt.axvline(x=t["date"], color="blue", alpha=0.3, linewidth=1)
            elif "SELL" in t["action"]:
                color = "green" if t.get("pnl", 0) > 0 else "red"
                plt.axvline(x=t["date"], color=color, alpha=0.3, linewidth=1)

        plt.title(f"Backtest: {STOCK} | Last {LOOKBACK_DAYS} Days")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("backtest_results.png")
        plt.show()
        print(f"\n  📊 Chart saved as backtest_results.png")

# ── Run grid search ───────────────────────────────────────────────
bars = get_historical_bars(STOCK, LOOKBACK_DAYS)

print(f"\n{'='*80}")
print("GRID SEARCH RESULTS")
print(f"{'='*80}")
print(f"  {'EMA':<10} {'RSI':<10} {'Levels':<12} {'Return':<10} {'Trades':<8} {'Win%':<8} {'Sharpe':<8}")
print(f"  {'-'*70}")

results = []
for p in PARAM_GRID:
    result = run_single_backtest(
        bars, p["fast"], p["slow"],
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
bars2 = get_historical_bars(STOCK, LOOKBACK_DAYS)
trades, portfolio_values, final_cash = run_backtest(
    bars2,
    best_p["fast"], best_p["slow"],
    best_p["rsi_period"], best_p["rsi_buy"], best_p["rsi_sell"]
)
calculate_metrics(trades, portfolio_values, final_cash)