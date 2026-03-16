import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
import schedule
import csv
import os
import json

from datetime import datetime, timedelta
from dotenv import load_dotenv
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

API_KEY = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
BASE_URL = os.environ["ALPACA_BASE_URL"]
NEWS_API_KEY = os.environ["NEWS_API_KEY"]

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()

STOCKS = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "TSLA", "META", "AMD", "NFLX", "CRM",
    # Finance
    "JPM", "BAC", "GS", "V",
    # ETFs
    "SPY", "QQQ",
    # Energy
    "XOM", "CVX",
    # Healthcare
    "JNJ", "PFE", "UNH",
    # Consumer
    "WMT", "COST", "NKE",
    # Semiconductor
    "INTC", "QCOM"
]
WATCHLIST = []

STOCK_NAMES = {
    # Tech
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "META": "Meta",
    "AMD": "AMD",
    "NFLX": "Netflix",
    "CRM": "Salesforce",
    # Finance
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "V": "Visa",
    # ETFs
    "SPY": "S&P 500",
    "QQQ": "Nasdaq",
    # Energy
    "XOM": "Exxon",
    "CVX": "Chevron",
    # Healthcare
    "JNJ": "Johnson and Johnson",
    "PFE": "Pfizer",
    "UNH": "UnitedHealth",
    # Consumer
    "WMT": "Walmart",
    "COST": "Costco",
    "NKE": "Nike",
    # Semiconductor
    "INTC": "Intel",
    "QCOM": "Qualcomm"
}

# ── Strategy parameters (defaults, overridden by walk-forward results) ──
FAST_MA = 6
SLOW_MA = 10
RSI_PERIOD = 7
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 55
BB_PERIOD = 15
BB_STD = 1.5

# ── Load per-stock optimized parameters from walk-forward results ──
STOCK_PARAMS = {}
for _stock in STOCKS:
    _path = f"walk_forward_results/{_stock}.json"
    if os.path.exists(_path):
        with open(_path) as _f:
            _data = json.load(_f)
            if _data.get("status") in ("PASS", "WEAK"):
                STOCK_PARAMS[_stock] = _data["best_params"]

def get_params(stock):
    """Return per-stock params if available, otherwise defaults."""
    if stock in STOCK_PARAMS:
        p = STOCK_PARAMS[stock]
        return p["fast"], p["slow"], p["rsi_buy"], p["rsi_sell"], p["bb_period"], p["bb_std"]
    return FAST_MA, SLOW_MA, RSI_OVERSOLD, RSI_OVERBOUGHT, BB_PERIOD, BB_STD

# ── MACD parameters ───────────────────────────────────────────────
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ── ATR parameters ────────────────────────────────────────────────
ATR_PERIOD = 14
ATR_RISK_PER_TRADE = 0.01   # Risk 1% of portfolio per trade

# ── Sentiment thresholds ──────────────────────────────────────────
SENTIMENT_THRESHOLD_BUY = 0.15
SENTIMENT_THRESHOLD_SELL = -0.15

# ── Risk management ───────────────────────────────────────────────
ATR_STOP_MULT = 1.5           # Stop loss = entry - 1.5 * ATR
ATR_PROFIT_MULT = 3.0         # Take profit = entry + 3.0 * ATR
REGIME_EMA = 50
DAILY_LOSS_LIMIT = -500       # Stop buying if daily P&L drops below this
MAX_POSITIONS = 5             # Maximum simultaneous open positions

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

# ── Account settings ──────────────────────────────────────────────
STARTING_CASH = 100000
LOG_FILE = "trades_log.csv"
MIN_QTY = 1
MAX_QTY = 50

# ── Track entry prices ────────────────────────────────────────────
ENTRY_PRICES_FILE = "entry_prices.json"

def load_entry_prices():
    if os.path.exists(ENTRY_PRICES_FILE):
        try:
            with open(ENTRY_PRICES_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_entry_prices():
    with open(ENTRY_PRICES_FILE, "w") as f:
        json.dump(ENTRY_PRICES, f)

ENTRY_PRICES = load_entry_prices()
PENDING_ORDERS = set()   # Prevent duplicate buys during order fill delay

# ── Create log file ───────────────────────────────────────────────
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "stock", "action", "price", "rsi",
                        "macd", "fast_ma", "slow_ma", "bb_upper",
                        "bb_lower", "atr", "qty", "sentiment", "est_cost"])

def log_trade(stock, action, price, rsi, macd, fast, slow,
              bb_upper, bb_lower, atr, qty, sentiment, est_cost=0):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), stock, action,
                        round(price, 2), round(rsi, 2), round(macd, 4),
                        round(fast, 2), round(slow, 2), round(bb_upper, 2),
                        round(bb_lower, 2), round(atr, 4), qty,
                        round(sentiment, 3), round(est_cost, 4)])

# ── Market checks ─────────────────────────────────────────────────
def market_is_open():
    clock = api.get_clock()
    return clock.is_open

def is_safe_trading_window():
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    if now < market_open + timedelta(minutes=15):
        print(f"  ⏰ Too close to market open — skipping")
        return False
    if now > market_close - timedelta(minutes=15):
        print(f"  ⏰ Too close to market close — skipping")
        return False
    return True

# ── Get minute bars ───────────────────────────────────────────────
def get_bars(stock, lookback_hours=2):
    end = datetime.now()
    start = end - timedelta(hours=lookback_hours)
    bars = api.get_bars(
        stock,
        tradeapi.rest.TimeFrame.Minute,
        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        feed="iex"
    ).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(stock, level="symbol")
    return bars

# ── Regime detection ──────────────────────────────────────────────
def market_is_uptrend():
    try:
        bars = get_bars("SPY", lookback_hours=4)
        if len(bars) < REGIME_EMA:
            return True
        close = bars["close"]
        regime_ema = close.ewm(span=REGIME_EMA, adjust=False).mean().iloc[-1]
        current_price = close.iloc[-1]
        is_uptrend = current_price > regime_ema
        trend = "📈 UPTREND" if is_uptrend else "📉 DOWNTREND"
        print(f"  Market Regime: {trend} | SPY: ${current_price:.2f} vs 50 EMA: ${regime_ema:.2f}")
        return is_uptrend
    except Exception as e:
        print(f"  Regime check error: {e}")
        return True

# ── EMA Crossover ─────────────────────────────────────────────────
def get_moving_averages(bars, fast_span=None, slow_span=None):
    fast_span = fast_span or FAST_MA
    slow_span = slow_span or SLOW_MA
    if len(bars) < slow_span:
        return None, None
    close = bars["close"]
    fast = close.ewm(span=fast_span, adjust=False).mean().iloc[-1]
    slow = close.ewm(span=slow_span, adjust=False).mean().iloc[-1]
    return fast, slow

# ── RSI ───────────────────────────────────────────────────────────
def get_rsi(bars):
    if len(bars) < RSI_PERIOD + 1:
        return None
    close = bars["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=RSI_PERIOD).mean().iloc[-1]
    avg_loss = loss.rolling(window=RSI_PERIOD).mean().iloc[-1]
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ── Bollinger Bands ───────────────────────────────────────────────
def get_bollinger_bands(bars, bb_period=None, bb_std=None):
    bb_period = bb_period or BB_PERIOD
    bb_std = bb_std or BB_STD
    if len(bars) < bb_period:
        return None, None, None
    close = bars["close"]
    middle = close.rolling(window=bb_period).mean().iloc[-1]
    std = close.rolling(window=bb_period).std().iloc[-1]
    upper = middle + (bb_std * std)
    lower = middle - (bb_std * std)
    return upper, middle, lower

# ── MACD ──────────────────────────────────────────────────────────
def get_macd(bars):
    if len(bars) < MACD_SLOW + MACD_SIGNAL:
        return None, None, None
    close = bars["close"]

    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal_line

    return (macd_line.iloc[-1],
            signal_line.iloc[-1],
            histogram.iloc[-1])

# ── ATR (Average True Range) ──────────────────────────────────────
def get_atr(bars):
    if len(bars) < ATR_PERIOD + 1:
        return None
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]

    # True Range = max of:
    # High - Low
    # |High - Previous Close|
    # |Low - Previous Close|
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=ATR_PERIOD).mean().iloc[-1]
    return atr

# ── VWAP (Volume Weighted Average Price) ─────────────────────────
def get_vwap(stock):
    try:
        now = datetime.now()
        today_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        bars = api.get_bars(
            stock,
            tradeapi.rest.TimeFrame.Minute,
            start=today_open.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            feed="iex"
        ).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(stock, level="symbol")
        if len(bars) < 1:
            return None
        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
        cum_tp_vol = (typical_price * bars["volume"]).cumsum()
        cum_vol = bars["volume"].cumsum()
        vwap = cum_tp_vol / cum_vol
        return vwap.iloc[-1]
    except Exception as e:
        print(f"  VWAP error for {stock}: {e}")
        return None

# ── ATR-based position sizing ─────────────────────────────────────
def get_atr_qty(price, atr):
    try:
        account = api.get_account()
        portfolio_value = float(account.portfolio_value)

        # Dollar risk per trade = 1% of portfolio
        dollar_risk = portfolio_value * ATR_RISK_PER_TRADE

        # Stop distance = 1x ATR (how much the stock typically moves)
        stop_distance = atr

        if stop_distance <= 0:
            return MIN_QTY

        # Qty = Dollar Risk / Stop Distance
        qty = int(dollar_risk / stop_distance)
        qty = max(MIN_QTY, min(qty, MAX_QTY))
        return qty

    except Exception as e:
        print(f"  ATR sizing error: {e}")
        return MIN_QTY

# ── Sentiment (time weighted, cached) ────────────────────────────
SENTIMENT_CACHE = {}         # {stock: (score, timestamp)}
SENTIMENT_TTL = 1800         # 30 minutes in seconds

def get_sentiment(stock):
    # Return cached value if fresh
    if stock in SENTIMENT_CACHE:
        cached_score, cached_time = SENTIMENT_CACHE[stock]
        if (datetime.now() - cached_time).total_seconds() < SENTIMENT_TTL:
            return cached_score
    try:
        company = STOCK_NAMES.get(stock, stock)
        articles = newsapi.get_everything(
            q=company,
            language="en",
            sort_by="publishedAt",
            page_size=10,
            from_param=(datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        )
        if not articles["articles"]:
            return 0
        scores = []
        weights = []
        now = datetime.now()
        for article in articles["articles"]:
            text = f"{article['title']} {article['description'] or ''}"
            score = analyzer.polarity_scores(text)["compound"]
            try:
                published = pd.to_datetime(article["publishedAt"]).replace(tzinfo=None)
                hours_old = max((now - published).total_seconds() / 3600, 0.1)
                weight = 1 / hours_old
            except:
                weight = 1.0
            scores.append(score)
            weights.append(weight)
        result = np.average(scores, weights=weights)
        SENTIMENT_CACHE[stock] = (result, datetime.now())
        return result
    except Exception as e:
        print(f"  Sentiment error for {stock}: {e}")
        return 0

# ── Pre-market scan ───────────────────────────────────────────────
def run_premarket_scan():
    global WATCHLIST
    print(f"\n{'='*50}")
    print(f"🌅 PRE-MARKET SCAN at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")
    WATCHLIST = []
    for stock in STOCKS:
        sentiment = get_sentiment(stock)
        if sentiment > SENTIMENT_THRESHOLD_BUY:
            WATCHLIST.append(stock)
            print(f"  ✅ {stock} added | Sentiment: {sentiment:.3f}")
        else:
            print(f"  ❌ {stock} excluded | Sentiment: {sentiment:.3f}")
    if not WATCHLIST:
        print(f"\n  ⚠️ No stocks passed filter — trading ALL stocks today")
        WATCHLIST = STOCKS.copy()
    else:
        print(f"\n  📋 Today's watchlist: {WATCHLIST}")
    with open("watchlist.txt", "w") as f:
        f.write("\n".join(WATCHLIST))
        f.write(f"\nLast updated: {datetime.now().strftime('%b %d %Y, %I:%M %p')}")

# ── Get current position ──────────────────────────────────────────
def get_position(stock):
    try:
        position = api.get_position(stock)
        return int(position.qty)
    except:
        return 0

# ── Get current price ─────────────────────────────────────────────
def get_price(stock):
    try:
        bars = get_bars(stock, lookback_hours=0.1)
        return bars["close"].iloc[-1]
    except:
        return 0

# ── Stop loss / take profit check ─────────────────────────────────
def check_exit_conditions(stock, current_price, position):
    if stock not in ENTRY_PRICES or position == 0:
        return None
    data = ENTRY_PRICES[stock]
    # Support both old format (float) and new format (dict)
    if isinstance(data, dict):
        entry = data["price"]
        entry_atr = data["atr"]
    else:
        entry = data
        entry_atr = abs(entry * 0.005)  # Fallback: ~0.5% if no ATR saved
    stop_price = entry - (ATR_STOP_MULT * entry_atr)
    target_price = entry + (ATR_PROFIT_MULT * entry_atr)
    if current_price <= stop_price:
        return "STOP LOSS"
    elif current_price >= target_price:
        return "TAKE PROFIT"
    return None

# ── Daily loss check ──────────────────────────────────────────────
def get_daily_pnl():
    try:
        account = api.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)  # Previous day's close
        return equity - last_equity
    except Exception as e:
        print(f"  Daily P&L check error: {e}")
        return 0

# ── Positions summary ─────────────────────────────────────────────
def show_positions():
    print(f"\n{'='*50}")
    print("OPEN POSITIONS")
    print(f"{'='*50}")
    try:
        positions = api.list_positions()
        if not positions:
            print("  No open positions.")
        else:
            total_pnl = 0
            for p in positions:
                pnl = float(p.unrealized_pl)
                pnl_pct = float(p.unrealized_plpc) * 100
                total_pnl += pnl
                emoji = "🟢" if pnl >= 0 else "🔴"
                data = ENTRY_PRICES.get(p.symbol)
                if isinstance(data, dict):
                    entry = data["price"]
                    entry_atr = data["atr"]
                else:
                    entry = data if data else float(p.avg_entry_price)
                    entry_atr = abs(entry * 0.005)
                stop = entry - (ATR_STOP_MULT * entry_atr)
                target = entry + (ATR_PROFIT_MULT * entry_atr)
                print(f"  {emoji} {p.symbol:6} | "
                      f"Qty: {p.qty:>4} | "
                      f"Avg: ${float(p.avg_entry_price):>8.2f} | "
                      f"Current: ${float(p.current_price):>8.2f} | "
                      f"PnL: ${pnl:>+8.2f} ({pnl_pct:+.2f}%) | "
                      f"Stop: ${stop:.2f} | "
                      f"Target: ${target:.2f}")
            print(f"\n  Total Unrealized PnL: ${total_pnl:+,.2f}")
    except Exception as e:
        print(f"  Error fetching positions: {e}")

# ── Performance summary ───────────────────────────────────────────
def show_performance():
    print(f"\n{'='*50}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    try:
        account = api.get_account()
        portfolio_value = float(account.portfolio_value)
        cash = float(account.cash)
        total_return = ((portfolio_value - STARTING_CASH) / STARTING_CASH) * 100
        print(f"  Portfolio Value:  ${portfolio_value:,.2f}")
        print(f"  Cash:             ${cash:,.2f}")
        print(f"  Total Return:     {total_return:+.2f}%")

        if os.path.exists(LOG_FILE):
            trades = pd.read_csv(LOG_FILE)
            if "est_cost" in trades.columns:
                total_est_costs = trades["est_cost"].sum()
                print(f"  Est. Costs:       ${total_est_costs:,.2f}")
            buys = trades[trades["action"] == "BUY"]
            sells = trades[trades["action"].isin(
                ["SELL", "STOP LOSS", "TAKE PROFIT"])]
            pairs = min(len(buys), len(sells))
            if pairs >= 3:
                returns = []
                for i in range(pairs):
                    buy_price = buys.iloc[i]["price"]
                    sell_price = sells.iloc[i]["price"]
                    ret = (sell_price - buy_price) / buy_price
                    returns.append(ret)
                returns = np.array(returns)
                if np.std(returns) > 0:
                    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
                    emoji = "🔥" if sharpe > 2 else "✅" if sharpe > 1 else "⚠️" if sharpe > 0 else "🔴"
                    wins = [r for r in returns if r > 0]
                    print(f"  Sharpe Ratio:     {sharpe:.2f} {emoji}")
                    print(f"  Win Rate:         {len(wins)/len(returns)*100:.1f}%")
            else:
                print(f"  Sharpe Ratio:     Need {3-pairs} more completed trades")
    except Exception as e:
        print(f"  Error generating performance summary: {e}")

# ── Main bot logic ────────────────────────────────────────────────
def run_bot():
    if not market_is_open():
        print(f"  Market is closed. Waiting... ({datetime.now().strftime('%H:%M:%S')})")
        return

    print(f"\n{'='*50}")
    print(f"Running bot at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")

    uptrend = market_is_uptrend()

    if not is_safe_trading_window():
        return

    # ── Risk guardrails ──────────────────────────────────────
    daily_pnl = get_daily_pnl()
    buys_blocked = daily_pnl <= DAILY_LOSS_LIMIT
    if buys_blocked:
        print(f"  🚫 Daily loss limit hit (${daily_pnl:+,.2f}) — no new buys today")

    open_positions = api.list_positions()
    position_count = len(open_positions)
    at_max_positions = position_count >= MAX_POSITIONS
    if at_max_positions:
        print(f"  🚫 Max positions reached ({position_count}/{MAX_POSITIONS}) — no new buys")

    active_stocks = WATCHLIST if WATCHLIST else STOCKS

    for stock in active_stocks:
        try:
            bars = get_bars(stock)

            # Get per-stock optimized parameters (or defaults)
            s_fast, s_slow, s_rsi_buy, s_rsi_sell, s_bb_period, s_bb_std = get_params(stock)

            # Calculate all indicators
            fast, slow = get_moving_averages(bars, s_fast, s_slow)
            rsi = get_rsi(bars)
            bb_upper, bb_middle, bb_lower = get_bollinger_bands(bars, s_bb_period, s_bb_std)
            macd_line, signal_line, histogram = get_macd(bars)
            atr = get_atr(bars)

            if any(v is None for v in [fast, slow, rsi, bb_upper,
                                        macd_line, atr]):
                print(f"  {stock}: Not enough data yet, skipping...")
                continue

            position = get_position(stock)
            if position > 0:
                PENDING_ORDERS.discard(stock)  # Order filled, clear pending
            price = get_price(stock)
            qty = get_atr_qty(price, atr)
            sentiment = get_sentiment(stock)
            vwap = get_vwap(stock)

            # MACD signal direction
            macd_bullish = macd_line > signal_line
            macd_bearish = macd_line < signal_line

            sentiment_emoji = "😊" if sentiment > 0.15 else "😐" if sentiment > -0.15 else "😟"
            macd_emoji = "📈" if macd_bullish else "📉"
            vwap_str = f"${vwap:.2f}" if vwap else "N/A"

            print(f"  {stock} | RSI: {rsi:.1f} | "
                  f"MACD: {macd_emoji} {macd_line:.3f}/{signal_line:.3f} | "
                  f"ATR: {atr:.2f} | VWAP: {vwap_str} | "
                  f"Sentiment: {sentiment:.3f} {sentiment_emoji} | "
                  f"Qty: {qty}")

            # ── Stop loss / take profit check ─────────────────
            if position > 0:
                exit_signal = check_exit_conditions(stock, price, position)
                if exit_signal:
                    api.submit_order(
                        symbol=stock,
                        qty=position,
                        side="sell",
                        type="market",
                        time_in_force="day"
                    )
                    est_cost = calculate_trade_cost(stock, price, position, "sell")
                    log_trade(stock, exit_signal, price, rsi,
                             macd_line, fast, slow, bb_upper,
                             bb_lower, atr, position, sentiment, est_cost)
                    entry_data = ENTRY_PRICES[stock]
                    entry_price = entry_data["price"] if isinstance(entry_data, dict) else entry_data
                    if exit_signal == "STOP LOSS":
                        print(f"  🛑 {stock} STOP LOSS | "
                              f"${price:.2f} | "
                              f"Entry: ${entry_price:.2f}")
                    else:
                        print(f"  🎯 {stock} TAKE PROFIT | "
                              f"${price:.2f} | "
                              f"Entry: ${entry_price:.2f}")
                    ENTRY_PRICES.pop(stock, None)
                    save_entry_prices()
                    PENDING_ORDERS.discard(stock)
                    continue

            # ── Buy: regime + EMA + (RSI OR BB) + (MACD OR VWAP) ─
            if (uptrend and
                not buys_blocked and
                not at_max_positions and
                fast > slow and
                (rsi < s_rsi_buy or price <= bb_lower) and
                (macd_bullish or (vwap is None or price < vwap)) and
                position == 0 and
                stock not in PENDING_ORDERS):

                api.submit_order(
                    symbol=stock,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                PENDING_ORDERS.add(stock)
                ENTRY_PRICES[stock] = {"price": price, "atr": atr}
                save_entry_prices()
                est_cost = calculate_trade_cost(stock, price, qty, "buy")
                log_trade(stock, "BUY", price, rsi, macd_line,
                         fast, slow, bb_upper, bb_lower,
                         atr, qty, sentiment, est_cost)
                stop = price - (ATR_STOP_MULT * atr)
                target = price + (ATR_PROFIT_MULT * atr)
                print(f"  ✅ {stock} BUY | "
                      f"Qty: {qty} (ATR sized) | "
                      f"${price:.2f} | "
                      f"Stop: ${stop:.2f} | "
                      f"Target: ${target:.2f}")

            # ── Sell: EMA + (RSI OR BB) + (MACD OR VWAP) ────
            elif (fast < slow and
                  (rsi > s_rsi_sell or price >= bb_upper) and
                  (macd_bearish or (vwap is None or price > vwap)) and
                  position > 0):

                api.submit_order(
                    symbol=stock,
                    qty=position,
                    side="sell",
                    type="market",
                    time_in_force="day"
                )
                est_cost = calculate_trade_cost(stock, price, position, "sell")
                log_trade(stock, "SELL", price, rsi, macd_line,
                         fast, slow, bb_upper, bb_lower,
                         atr, position, sentiment, est_cost)
                print(f"  🔴 {stock} SELL SIGNAL | ${price:.2f}")
                ENTRY_PRICES.pop(stock, None)
                save_entry_prices()
                PENDING_ORDERS.discard(stock)

            else:
                if position > 0 and stock in ENTRY_PRICES:
                    data = ENTRY_PRICES[stock]
                    if isinstance(data, dict):
                        entry = data["price"]
                        entry_atr = data["atr"]
                    else:
                        entry = data
                        entry_atr = abs(entry * 0.005)
                    pnl_pct = (price - entry) / entry * 100
                    stop = entry - (ATR_STOP_MULT * entry_atr)
                    target = entry + (ATR_PROFIT_MULT * entry_atr)
                    print(f"  ⏸ {stock} HOLD | "
                          f"PnL: {pnl_pct:+.2f}% | "
                          f"Stop: ${stop:.2f} | "
                          f"Target: ${target:.2f}")
                else:
                    print(f"  ⏸ {stock} HOLD")

        except Exception as e:
            print(f"  Error with {stock}: {e}")

    show_positions()
    show_performance()

# ── Schedule ──────────────────────────────────────────────────────
schedule.every().monday.at("04:00").do(run_premarket_scan)
schedule.every().tuesday.at("04:00").do(run_premarket_scan)
schedule.every().wednesday.at("04:00").do(run_premarket_scan)
schedule.every().thursday.at("04:00").do(run_premarket_scan)
schedule.every().friday.at("04:00").do(run_premarket_scan)

schedule.every().monday.at("15:55").do(show_positions)
schedule.every().tuesday.at("15:55").do(show_positions)
schedule.every().wednesday.at("15:55").do(show_positions)
schedule.every().thursday.at("15:55").do(show_positions)
schedule.every().friday.at("15:55").do(show_positions)

schedule.every(1).minutes.do(run_bot)

print("🚀 Bot: EMA + RSI + BB + MACD + ATR Sizing + Sentiment + Kelly + Stop Loss + Regime!")
print(f"   Default EMA: {FAST_MA}/{SLOW_MA} | RSI: {RSI_OVERSOLD}/{RSI_OVERBOUGHT} | "
      f"BB: {BB_PERIOD}/{BB_STD} | MACD: {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}")
if STOCK_PARAMS:
    print(f"   📊 Per-stock optimized params loaded for {len(STOCK_PARAMS)} stocks: {', '.join(STOCK_PARAMS.keys())}")
else:
    print("   ⚠️ No walk-forward results found, using default params for all stocks")
print("Checks every minute during market hours.")
print("Press Ctrl+C to stop.")

run_premarket_scan()
run_bot()

while True:
    schedule.run_pending()
    time.sleep(1)