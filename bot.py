import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
import schedule
import csv
import os
import json
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import signals

load_dotenv()

API_KEY = "PK22XEELBFYNU7QMJHJOGRJ6V6"
SECRET_KEY = "3arXWSeJW69nWfZHKW9nABMWwMkK1Ct964VakJdT7PXV"
BASE_URL = "https://paper-api.alpaca.markets"
NEWS_API_KEY = "801fe14f0cdc4eac8344f9b7ae242e66"
SENTIMENT_CACHE_FILE = "sentiment_cache.json"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()

# ── Tiered Stock Universe ───────────────────────────────────────────
# Tier 1 — Full size (p < 0.05 + PF >= 1.3 + positive BT return — cross-validated both tests)
TIER1_STOCKS = ["MSFT", "AAPL", "MS", "BLK"]
# Tier 2 — Half size (p < 0.15 + PF >= 1.2 + positive BT return)
TIER2_STOCKS = ["V", "WMT", "GS", "AMZN"]
# Tier 3 — Quarter size (pipeline T3 or borderline perm — monitor only)
TIER3_STOCKS = ["QQQ", "COST", "MA", "NET", "AMD"]
# Pending validation — passed permutation but negative backtest return; need better backtest
PENDING_VALIDATION = ["SHOP", "PLTR"]

STOCKS = TIER1_STOCKS + TIER2_STOCKS + TIER3_STOCKS
WATCHLIST = []

TIER1_SIZE_FACTOR = 1.0
TIER2_SIZE_FACTOR = 0.5
TIER3_SIZE_FACTOR = 0.25

# ── Short selling configuration ─────────────────────────────────────
# High-liquidity, easy-to-borrow; WMT/COST excluded (defensive — may rise in bear markets)
SHORT_ELIGIBLE = ["NVDA", "AMD", "AMZN", "AAPL", "MSFT", "META", "NFLX", "COIN", "PLTR"]
MAX_SHORT_POSITIONS = 3
SHORT_DAILY_LOSS_LIMIT = -300
SHORT_MAX_LOSS_PCT = 0.05  # Force close if stock rises 5% above short entry

# ── Pairs tracking (diagnostic only, no auto-trading) ───────────────
PAIRS = {"SEMI": ("AMD", "NVDA"), "MEGACAP": ("MSFT", "AAPL")}

# ── Opening Range Breakout data ──────────────────────────────────────
ORB_DATA = {}  # {stock: {"high": float, "low": float, "date": str, "finalized": bool}}

# ── Sector diversification ──────────────────────────────────────────
SECTOR_MAP = {
    "tech":     ["MSFT", "META", "AAPL", "AMZN", "NFLX", "NET", "CRM", "UBER", "SPOT", "SHOP", "PLTR"],
    "semi":     ["AMD", "NVDA", "QCOM"],
    "finance":  ["GS", "MS", "BLK", "V", "MA", "COIN", "HOOD", "PYPL"],
    "consumer": ["WMT", "COST", "TSLA", "UBER", "ABNB"],
    "etf":      ["SPY", "QQQ"],
}
# Note: finance sector has 5 active stocks (GS, MS, BLK, V, MA).
# MAX_POSITIONS_PER_SECTOR = 2 caps portfolio heat — correct behavior.
MAX_POSITIONS_PER_SECTOR = 2

STOCK_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia",
    "GOOGL": "Google", "AMZN": "Amazon", "TSLA": "Tesla",
    "META": "Meta", "AMD": "AMD", "NFLX": "Netflix",
    "CRM": "Salesforce", "JPM": "JPMorgan", "BAC": "Bank of America",
    "GS": "Goldman Sachs", "V": "Visa", "SPY": "S&P 500",
    "QQQ": "Nasdaq ETF", "XOM": "Exxon", "CVX": "Chevron",
    "JNJ": "Johnson and Johnson", "PFE": "Pfizer",
    "UNH": "UnitedHealth", "WMT": "Walmart", "COST": "Costco",
    "NKE": "Nike", "INTC": "Intel", "QCOM": "Qualcomm",
    "UBER": "Uber", "COIN": "Coinbase", "HOOD": "Robinhood",
    "SPOT": "Spotify", "ROKU": "Roku", "ABNB": "Airbnb",
    "PYPL": "PayPal", "SHOP": "Shopify", "PLTR": "Palantir",
    "AVGO": "Broadcom", "LLY": "Eli Lilly",
    "MA": "Mastercard", "PANW": "Palo Alto Networks",
    "CRWD": "CrowdStrike", "SNOW": "Snowflake",
    "DDOG": "Datadog", "NET": "Cloudflare",
    "ADBE": "Adobe", "NOW": "ServiceNow",
    "ORCL": "Oracle", "MS": "Morgan Stanley",
    "BLK": "BlackRock", "SCHW": "Charles Schwab",
    "HD": "Home Depot", "MCD": "McDonalds",
    "SBUX": "Starbucks",
}

# ── Strategy parameters (defaults, overridden by walk-forward results) ──
FAST_MA = 4
SLOW_MA = 10
RSI_PERIOD = 7
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 65
BB_PERIOD = 15
BB_STD = 1.5
DEFAULT_BUY_THRESHOLD = 4.0
DEFAULT_SELL_THRESHOLD = 3.5

# ── Diagnostic and test modes ─────────────────────────────────────
DIAGNOSTIC_MODE = True   # Print full score breakdowns each minute per stock
TEST_MODE = True         # Subtract 1.0 from all thresholds (calibration only — NOT for live trading)

# ── Load per-stock optimized parameters from walk-forward results ──
STOCK_PARAMS = {}
for _stock in STOCKS:
    _path = f"walk_forward_results/{_stock}.json"
    if os.path.exists(_path):
        with open(_path) as _f:
            _data = json.load(_f)
            if _data.get("status") in ("PASS", "WEAK"):
                _params = _data["best_params"]
                # Also store optimal_threshold from top-level or from best_params
                if "optimal_threshold" in _data:
                    _params["optimal_threshold"] = _data["optimal_threshold"]
                STOCK_PARAMS[_stock] = _params

# ── Load overnight backtest results for additional filtering ──
BACKTEST_RESULTS = {}
for _stock in STOCKS:
    _path = f"backtest_results/{_stock}.json"
    if os.path.exists(_path):
        with open(_path) as _f:
            BACKTEST_RESULTS[_stock] = json.load(_f)

def get_params(stock):
    """Return per-stock params if available, otherwise defaults."""
    if stock in STOCK_PARAMS:
        p = STOCK_PARAMS[stock]
        threshold = p.get("optimal_threshold", p.get("score_threshold", DEFAULT_BUY_THRESHOLD))
        return (p["fast"], p["slow"], p["rsi_buy"], p["rsi_sell"],
                p["bb_period"], p["bb_std"], threshold)
    return FAST_MA, SLOW_MA, RSI_OVERSOLD, RSI_OVERBOUGHT, BB_PERIOD, BB_STD, DEFAULT_BUY_THRESHOLD

# ── Minimum hold time ────────────────────────────────────────────
MIN_HOLD_BARS = 5  # Hold at least 5 minutes — no whipsaw exits
ENTRY_BARS = {}    # Track entry bar index per stock for min hold

# ── ATR parameters ────────────────────────────────────────────────
ATR_PERIOD = 14
ATR_RISK_PER_TRADE = 0.01   # Risk 1% of portfolio per trade

# ── Risk management ───────────────────────────────────────────────
ATR_STOP_MULT = 1.5           # Stop loss = entry - 1.5 * ATR
ATR_PROFIT_MULT = 3.0         # Take profit = entry + 3.0 * ATR
DAILY_LOSS_LIMIT = -500       # Stop buying if daily P&L drops below this
MAX_POSITIONS = 5             # Maximum simultaneous open positions (longs)

# ── Transaction cost model ───────────────────────────────────────
COST_MODEL_ENABLED = True

def calculate_trade_cost(stock, price, quantity, side):
    """Calculate total transaction cost for a single trade."""
    return signals.calculate_trade_cost(stock, price, quantity, side, COST_MODEL_ENABLED)

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
    # Market opens at 6:30 AM Pacific Time (9:30 AM ET)
    market_open = now.replace(hour=6, minute=30, second=0, microsecond=0)
    # Market closes at 1:00 PM Pacific Time (4:00 PM ET)
    market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)
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

# ── Opening Range Breakout tracking ──────────────────────────────
def calculate_orb(stock, bars):
    """Track the 9:30–9:45 AM ET opening range. Returns dict with orb_high/low and
    whether the current price has broken above/below the range."""
    global ORB_DATA
    today_str = pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d')

    existing = ORB_DATA.get(stock, {})
    if existing.get("date") == today_str and existing.get("finalized"):
        last_price = float(bars["close"].iloc[-1])
        return {
            "orb_high": existing["high"],
            "orb_low":  existing["low"],
            "is_above_high": last_price > existing["high"],
            "is_below_low":  last_price < existing["low"],
            "finalized": True,
        }

    # Convert bar index to ET timezone
    if bars.index.tz is None:
        bars_et = bars.copy()
        bars_et.index = bars_et.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        bars_et = bars.copy()
        bars_et.index = bars_et.index.tz_convert('America/New_York')

    orb_start = pd.Timestamp(today_str + ' 09:30', tz='America/New_York')
    orb_end   = pd.Timestamp(today_str + ' 09:45', tz='America/New_York')
    orb_bars  = bars_et[(bars_et.index >= orb_start) & (bars_et.index < orb_end)]

    if len(orb_bars) < 3:
        return {"orb_high": None, "orb_low": None, "is_above_high": False,
                "is_below_low": False, "finalized": False}

    orb_high   = float(orb_bars["high"].max())
    orb_low    = float(orb_bars["low"].min())
    last_price = float(bars["close"].iloc[-1])
    now_et     = pd.Timestamp.now(tz='America/New_York')
    finalized  = now_et >= orb_end

    if finalized and existing.get("date") != today_str:
        ORB_DATA[stock] = {"high": orb_high, "low": orb_low,
                           "date": today_str, "finalized": True}
        print(f"  📊 {stock} ORB finalized: High ${orb_high:.2f} / Low ${orb_low:.2f}")

    return {
        "orb_high":      orb_high,
        "orb_low":       orb_low,
        "is_above_high": last_price > orb_high,
        "is_below_low":  last_price < orb_low,
        "finalized":     finalized,
    }

# ── Pairs signal check (diagnostic only) ─────────────────────────
def check_pairs_signal(pair_name):
    """Compute z-score of price ratio for a stock pair. Prints alert when |z| >= 2.0."""
    if pair_name not in PAIRS:
        return
    stock_a, stock_b = PAIRS[pair_name]
    try:
        bars_a = get_bars(stock_a, lookback_hours=1.5)
        bars_b = get_bars(stock_b, lookback_hours=1.5)
        if len(bars_a) < 10 or len(bars_b) < 10:
            return
        close_a = bars_a["close"].reindex(bars_b["close"].index, method="nearest").dropna()
        close_b = bars_b["close"].reindex(close_a.index, method="nearest").dropna()
        if len(close_a) < 10:
            return
        spread = close_a / close_b
        zscore = (spread.iloc[-1] - spread.mean()) / spread.std()
        if abs(zscore) >= 2.0:
            direction = "LONG A / SHORT B" if zscore > 0 else "SHORT A / LONG B"
            print(f"  🔗 PAIRS {pair_name} ({stock_a}/{stock_b}): z={zscore:.2f} → {direction}")
        elif DIAGNOSTIC_MODE:
            print(f"  🔗 PAIRS {pair_name} ({stock_a}/{stock_b}): z={zscore:.2f} (neutral)")
    except Exception as e:
        print(f"  Pairs check error ({pair_name}): {e}")

# ── Diagnostic label helpers ──────────────────────────────────────
def _rs_label(rs):
    if rs is None:  return "N/A"
    if rs >  0.5:   return f"strong +{rs:.2f}"
    if rs < -0.5:   return f"weak {rs:.2f}"
    return f"neutral {rs:.2f}"

def _orb_label(orb):
    if orb is None or not orb.get("finalized"): return "not_set"
    if orb["is_above_high"]: return f"above_high(${orb['orb_high']:.2f})"
    if orb["is_below_low"]:  return f"below_low(${orb['orb_low']:.2f})"
    return "inside"

def _vwap_label(price, vd):
    if vd is None: return "N/A"
    if price > vd.get("upper_2std", 1e9): return ">+2std"
    if price > vd.get("upper_1std", 1e9): return ">+1std"
    if price < vd.get("lower_2std", 0):   return "<-2std"
    if price < vd.get("lower_1std", 0):   return "<-1std"
    return "inside_bands"

# ── Regime detection (daily 20-day EMA) ──────────────────────────
DAILY_REGIME_EMA = 20

def market_is_uptrend():
    """Check if SPY is above its 20-day EMA (daily timeframe).
    Flips once per day at most, not dozens of times like intraday EMA.
    """
    try:
        end = datetime.now()
        start = end - timedelta(days=45)  # 45 calendar days for warmup
        bars = api.get_bars(
            "SPY",
            tradeapi.rest.TimeFrame.Day,
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            feed="iex"
        ).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs("SPY", level="symbol")
        if len(bars) < DAILY_REGIME_EMA:
            return True
        close = bars["close"]
        ema_20 = close.ewm(span=DAILY_REGIME_EMA, adjust=False).mean()
        current_price = close.iloc[-1]
        regime_ema = ema_20.iloc[-1]
        is_uptrend = current_price > regime_ema
        trend = "UPTREND" if is_uptrend else "DOWNTREND"
        print(f"  Market Regime: {trend} | SPY: ${current_price:.2f} vs {DAILY_REGIME_EMA}d EMA: ${regime_ema:.2f}")
        return is_uptrend
    except Exception as e:
        print(f"  Regime check error: {e}")
        return True

# ── ATR-based position sizing ─────────────────────────────────────
def get_atr_qty(stock, price, atr):
    try:
        account = api.get_account()
        portfolio_value = float(account.portfolio_value)

        # Dollar risk per trade = 1% of portfolio
        dollar_risk = portfolio_value * ATR_RISK_PER_TRADE

        # Stop distance = 1x ATR
        stop_distance = atr

        if stop_distance <= 0:
            return MIN_QTY

        qty = int(dollar_risk / stop_distance)

        # Tier-based sizing: T3 = 25%, T2 = 50%, T1 = full
        if stock in TIER3_STOCKS:
            qty = max(MIN_QTY, int(qty * 0.25))
        elif stock in TIER2_STOCKS:
            qty = max(MIN_QTY, int(qty * 0.50))

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

# ── Sentiment thresholds (for pre-market scan only) ──────────────
SENTIMENT_THRESHOLD_BUY = 0.15

# ── Pre-market scan ───────────────────────────────────────────────
def run_premarket_scan():
    global WATCHLIST
    print(f"\n{'='*50}")
    print(f"🌅 PRE-MARKET SCAN at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")

    # Check if today's sentiment cache already exists (avoids burning API calls on restarts)
    today_str = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(SENTIMENT_CACHE_FILE):
        try:
            with open(SENTIMENT_CACHE_FILE, "r") as f:
                cache_data = json.load(f)
            if cache_data.get("date") == today_str:
                print(f"  📋 Loading sentiment from today's cache ({SENTIMENT_CACHE_FILE})")
                scores = cache_data.get("scores", {})
                WATCHLIST = []
                for stock in STOCKS:
                    sentiment = scores.get(stock, 0)
                    if sentiment > SENTIMENT_THRESHOLD_BUY:
                        WATCHLIST.append(stock)
                        print(f"  ✅ {stock} added | Sentiment: {sentiment:.3f} (cached)")
                    else:
                        print(f"  ❌ {stock} excluded | Sentiment: {sentiment:.3f} (cached)")
                if not WATCHLIST:
                    print(f"\n  ⚠️ No stocks passed filter — trading ALL stocks today")
                    WATCHLIST = STOCKS.copy()
                else:
                    print(f"\n  📋 Today's watchlist: {WATCHLIST}")
                with open("watchlist.txt", "w") as f:
                    f.write("\n".join(WATCHLIST))
                    f.write(f"\nLast updated: {datetime.now().strftime('%b %d %Y, %I:%M %p')}")
                return
        except Exception as e:
            print(f"  Cache load error: {e} — fetching fresh sentiment")

    # No valid cache — fetch fresh from NewsAPI and save for the day
    WATCHLIST = []
    fresh_scores = {}
    for stock in STOCKS:
        sentiment = get_sentiment(stock)
        fresh_scores[stock] = sentiment
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
    # Save cache so bot restarts today reuse the same scores
    try:
        with open(SENTIMENT_CACHE_FILE, "w") as f:
            json.dump({"date": today_str, "scores": fresh_scores}, f, indent=2)
        print(f"  💾 Sentiment cached to {SENTIMENT_CACHE_FILE}")
    except Exception as e:
        print(f"  Cache save error: {e}")

# ── Get current position ──────────────────────────────────────────
def get_position(stock):
    """Returns positive for long, negative for short, 0 for flat."""
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
    """Check ATR-based stops for both long and short positions."""
    if stock not in ENTRY_PRICES or position == 0:
        return None
    data = ENTRY_PRICES[stock]
    if isinstance(data, dict):
        entry = data["price"]
        entry_atr = data["atr"]
        direction = data.get("direction", "long")
    else:
        entry = data
        entry_atr = abs(entry * 0.005)
        direction = "long"

    if direction == "long":
        stop_price = entry - (ATR_STOP_MULT * entry_atr)
        target_price = entry + (ATR_PROFIT_MULT * entry_atr)
        if current_price <= stop_price:
            return "STOP LOSS"
        elif current_price >= target_price:
            return "TAKE PROFIT"
    else:  # short
        stop_price = entry + (ATR_STOP_MULT * entry_atr)
        target_price = entry - (ATR_PROFIT_MULT * entry_atr)
        max_loss_price = entry * (1 + SHORT_MAX_LOSS_PCT)
        if current_price >= stop_price or current_price >= max_loss_price:
            return "SHORT STOP LOSS"
        elif current_price <= target_price:
            return "SHORT TAKE PROFIT"
    return None

# ── Daily loss check ──────────────────────────────────────────────
def get_daily_pnl():
    try:
        account = api.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        return equity - last_equity
    except Exception as e:
        print(f"  Daily P&L check error: {e}")
        return 0

# ── Sector check ──────────────────────────────────────────────────
def get_stock_sector(stock):
    for sector, stocks in SECTOR_MAP.items():
        if stock in stocks:
            return sector
    return "other"

def count_sector_positions(sector, open_positions):
    """Count how many open positions are in the given sector."""
    count = 0
    for p in open_positions:
        if get_stock_sector(p.symbol) == sector:
            count += 1
    return count

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
            long_count = 0
            short_count = 0
            for p in positions:
                qty = int(p.qty)
                pnl = float(p.unrealized_pl)
                pnl_pct = float(p.unrealized_plpc) * 100
                total_pnl += pnl
                is_short = qty < 0
                if is_short:
                    short_count += 1
                else:
                    long_count += 1
                emoji = "🟢" if pnl >= 0 else "🔴"
                direction_label = "SHORT" if is_short else "LONG"
                data = ENTRY_PRICES.get(p.symbol)
                if isinstance(data, dict):
                    entry = data["price"]
                    entry_atr = data["atr"]
                    direction = data.get("direction", "long")
                else:
                    entry = data if data else float(p.avg_entry_price)
                    entry_atr = abs(entry * 0.005)
                    direction = "long"
                if direction == "long":
                    stop = entry - (ATR_STOP_MULT * entry_atr)
                    target = entry + (ATR_PROFIT_MULT * entry_atr)
                else:
                    stop = entry + (ATR_STOP_MULT * entry_atr)
                    target = entry - (ATR_PROFIT_MULT * entry_atr)
                print(f"  {emoji} {p.symbol:6} {direction_label:5} | "
                      f"Qty: {p.qty:>4} | "
                      f"Avg: ${float(p.avg_entry_price):>8.2f} | "
                      f"Current: ${float(p.current_price):>8.2f} | "
                      f"PnL: ${pnl:>+8.2f} ({pnl_pct:+.2f}%) | "
                      f"Stop: ${stop:.2f} | "
                      f"Target: ${target:.2f}")
            print(f"\n  Longs: {long_count} | Shorts: {short_count} | "
                  f"Total Unrealized PnL: ${total_pnl:+,.2f}")
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
                ["SELL", "STOP LOSS", "TAKE PROFIT",
                 "SHORT STOP LOSS", "SHORT TAKE PROFIT", "SHORT COVER",
                 "REGIME COVER"])]
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

    # ── Regime switch: force-close all shorts if uptrend ──────
    if uptrend:
        try:
            for pos in api.list_positions():
                if int(pos.qty) < 0:
                    cover_qty = abs(int(pos.qty))
                    api.submit_order(
                        symbol=pos.symbol,
                        qty=cover_qty,
                        side="buy",
                        type="market",
                        time_in_force="day"
                    )
                    print(f"  📈 REGIME SWITCH: Covering short {pos.symbol} ({cover_qty} shares)")
                    est_cost = calculate_trade_cost(pos.symbol, float(pos.current_price), cover_qty, "buy")
                    rsi_val = 50  # Approximate for logging
                    log_trade(pos.symbol, "REGIME COVER", float(pos.current_price),
                             rsi_val, 0, 0, 0, 0, 0, 0, cover_qty, 0, est_cost)
                    ENTRY_PRICES.pop(pos.symbol, None)
                    save_entry_prices()
                    PENDING_ORDERS.discard(pos.symbol)
        except Exception as e:
            print(f"  Regime switch cover error: {e}")

    # ── Risk guardrails ──────────────────────────────────
    daily_pnl = get_daily_pnl()
    buys_blocked = daily_pnl <= DAILY_LOSS_LIMIT
    shorts_blocked = daily_pnl <= SHORT_DAILY_LOSS_LIMIT
    if buys_blocked:
        print(f"  🚫 Daily loss limit hit (${daily_pnl:+,.2f}) — no new buys today")
    if shorts_blocked:
        print(f"  🚫 Short loss limit hit (${daily_pnl:+,.2f}) — no new shorts today")

    open_positions = api.list_positions()
    long_count = sum(1 for p in open_positions if int(p.qty) > 0)
    short_count = sum(1 for p in open_positions if int(p.qty) < 0)
    at_max_longs = long_count >= MAX_POSITIONS
    at_max_shorts = short_count >= MAX_SHORT_POSITIONS
    if at_max_longs:
        print(f"  🚫 Max long positions ({long_count}/{MAX_POSITIONS})")
    if at_max_shorts:
        print(f"  🚫 Max short positions ({short_count}/{MAX_SHORT_POSITIONS})")

    active_stocks = WATCHLIST if WATCHLIST else STOCKS

    # Fetch SPY bars once for relative strength calculation
    spy_bars_for_rs = None
    try:
        spy_bars_for_rs = get_bars("SPY")
    except Exception as e:
        print(f"  SPY bars fetch error (RS disabled): {e}")

    for stock in active_stocks:
        try:
            bars = get_bars(stock)

            # Get per-stock optimized parameters (or defaults)
            s_fast, s_slow, s_rsi_buy, s_rsi_sell, s_bb_period, s_bb_std, s_threshold = get_params(stock)

            # Calculate indicators using signals module
            fast, slow = signals.get_moving_averages(bars, s_fast, s_slow)
            rsi = signals.get_rsi(bars)
            bb_upper, bb_middle, bb_lower = signals.get_bollinger_bands(bars, s_bb_period, s_bb_std)
            atr = signals.get_atr(bars)

            if any(v is None for v in [fast, slow, rsi, bb_upper, atr]):
                print(f"  {stock}: Not enough data yet, skipping...")
                continue

            # ── v3 signals: ORB, VWAP bands, Relative Strength ────────
            orb_data  = calculate_orb(stock, bars)
            vwap_data = signals.get_vwap_bands(bars)
            rel_strength = None
            if stock != "SPY" and spy_bars_for_rs is not None:
                try:
                    rel_strength = signals.get_relative_strength(bars, spy_bars_for_rs)
                except Exception:
                    rel_strength = None

            position = get_position(stock)
            if position != 0:
                PENDING_ORDERS.discard(stock)
            price = get_price(stock)
            qty = get_atr_qty(stock, price, atr)

            # Volume for scoring
            cur_volume = None
            cur_avg_volume = None
            if len(bars) >= 20:
                cur_volume = bars["volume"].iloc[-1]
                cur_avg_volume = bars["volume"].rolling(window=20).mean().iloc[-1]

            # Determine tier label
            if stock in TIER1_STOCKS:
                tier_label = "T1"
            elif stock in TIER2_STOCKS:
                tier_label = "T2"
            else:
                tier_label = "T3"

            ema_sep = abs(fast - slow) / slow * 100 if slow > 0 else 0
            print(f"  {stock} [{tier_label}] | RSI: {rsi:.1f} | "
                  f"EMA sep: {ema_sep:.3f}% | "
                  f"ATR: {atr:.2f} | "
                  f"Qty: {qty}")

            # ═══════════════════════════════════════════════════
            # 1. EXIT CHECKS (stop loss / take profit) — always first
            # ═══════════════════════════════════════════════════
            if position != 0:
                exit_signal = check_exit_conditions(stock, price, position)
                if exit_signal:
                    exit_side = "sell" if position > 0 else "buy"
                    exit_qty = abs(position)
                    api.submit_order(
                        symbol=stock,
                        qty=exit_qty,
                        side=exit_side,
                        type="market",
                        time_in_force="day"
                    )
                    est_cost = calculate_trade_cost(stock, price, exit_qty, exit_side)
                    log_trade(stock, exit_signal, price, rsi,
                             0, fast, slow, bb_upper,
                             bb_lower, atr, exit_qty, 0, est_cost)
                    entry_data = ENTRY_PRICES.get(stock, {})
                    entry_price = entry_data.get("price", price) if isinstance(entry_data, dict) else entry_data
                    direction = entry_data.get("direction", "long") if isinstance(entry_data, dict) else "long"
                    if "STOP LOSS" in exit_signal:
                        print(f"  🛑 {stock} {exit_signal} | "
                              f"${price:.2f} | Entry: ${entry_price:.2f} | "
                              f"Dir: {direction}")
                    else:
                        print(f"  🎯 {stock} {exit_signal} | "
                              f"${price:.2f} | Entry: ${entry_price:.2f} | "
                              f"Dir: {direction}")
                    ENTRY_PRICES.pop(stock, None)
                    save_entry_prices()
                    PENDING_ORDERS.discard(stock)
                    continue

            # ═══════════════════════════════════════════════════
            # 2. SELL SIGNAL (close long via scoring)
            # ═══════════════════════════════════════════════════
            if position > 0:
                # Min hold time check
                entry_bar = ENTRY_BARS.get(stock, 0)
                bars_held = len(bars) - entry_bar if entry_bar > 0 else MIN_HOLD_BARS
                if bars_held < MIN_HOLD_BARS:
                    print(f"  ⏸ {stock} HOLD (min hold: {bars_held}/{MIN_HOLD_BARS} bars)")
                    continue

                sell_score, sell_bd = signals.calculate_sell_score(
                    fast, slow, rsi, s_rsi_sell, price, bb_upper,
                    bb_lower=bb_lower
                )
                sell_threshold = DEFAULT_SELL_THRESHOLD if stock not in STOCK_PARAMS else s_threshold - 0.5
                if TEST_MODE:
                    sell_threshold = max(0.5, sell_threshold - 1.0)

                if sell_score >= sell_threshold:
                    api.submit_order(
                        symbol=stock,
                        qty=position,
                        side="sell",
                        type="market",
                        time_in_force="day"
                    )
                    est_cost = calculate_trade_cost(stock, price, position, "sell")
                    log_trade(stock, "SELL", price, rsi, 0,
                             fast, slow, bb_upper, bb_lower,
                             atr, position, 0, est_cost)
                    print(f"  🔴 {stock} SELL score: {sell_score:.1f}/{sell_threshold} | "
                          f"${price:.2f} | {sell_bd}")
                    ENTRY_PRICES.pop(stock, None)
                    save_entry_prices()
                    PENDING_ORDERS.discard(stock)
                    continue

            # ═══════════════════════════════════════════════════
            # 3. COVER SIGNAL (close short via scoring)
            # ═══════════════════════════════════════════════════
            if position < 0:
                cover_score, cover_bd = signals.calculate_cover_score(
                    fast, slow, rsi, s_rsi_buy, price, bb_lower,
                    bb_upper=bb_upper
                )
                cover_threshold = s_threshold - 0.5
                if TEST_MODE:
                    cover_threshold = max(0.5, cover_threshold - 1.0)

                if cover_score >= cover_threshold:
                    cover_qty = abs(position)
                    api.submit_order(
                        symbol=stock,
                        qty=cover_qty,
                        side="buy",
                        type="market",
                        time_in_force="day"
                    )
                    est_cost = calculate_trade_cost(stock, price, cover_qty, "buy")
                    log_trade(stock, "SHORT COVER", price, rsi, 0,
                             fast, slow, bb_upper, bb_lower,
                             atr, cover_qty, 0, est_cost)
                    print(f"  📗 {stock} COVER score: {cover_score:.1f}/{cover_threshold} | "
                          f"${price:.2f} | {cover_bd}")
                    ENTRY_PRICES.pop(stock, None)
                    save_entry_prices()
                    PENDING_ORDERS.discard(stock)
                    continue

            # ═══════════════════════════════════════════════════
            # 4. NEW ENTRY SIGNALS (only when flat)
            # ═══════════════════════════════════════════════════
            if position == 0 and stock not in PENDING_ORDERS:
                # Backtest quality check
                bt = BACKTEST_RESULTS.get(stock, {})
                bt_ok = bt.get("profit_factor", 999) > 1.0
                if not bt_ok:
                    print(f"  ⚠️ {stock} skipped — backtest profit factor < 1.0")
                    continue

                # Sector check
                stock_sector = get_stock_sector(stock)
                sector_count = count_sector_positions(stock_sector, open_positions)

                # Calculate buy score (gradient scoring v3, per-stock threshold)
                buy_score, buy_bd = signals.calculate_buy_score(
                    fast, slow, rsi, s_rsi_buy, price, bb_lower,
                    bb_upper=bb_upper,
                    regime_uptrend=uptrend,
                    current_volume=cur_volume, avg_volume=cur_avg_volume,
                    rel_strength=rel_strength, orb_data=orb_data, vwap_data=vwap_data
                )
                buy_threshold = s_threshold

                # Calculate short score (gradient scoring v3; no +1.0 penalty — regime gate provides conservatism)
                short_score, short_bd = signals.calculate_short_score(
                    fast, slow, rsi, s_rsi_sell, price, bb_upper,
                    bb_lower=bb_lower,
                    regime_downtrend=not uptrend,
                    current_volume=cur_volume, avg_volume=cur_avg_volume,
                    rel_strength=rel_strength, orb_data=orb_data, vwap_data=vwap_data
                )
                short_threshold = s_threshold

                # TEST_MODE: lower all entry thresholds by 1.0 to verify end-to-end flow
                if TEST_MODE:
                    buy_threshold = max(0.5, buy_threshold - 1.0)
                    short_threshold = max(0.5, short_threshold - 1.0)

                # DIAGNOSTIC_MODE: full per-stock breakdown every cycle
                if DIAGNOSTIC_MODE:
                    regime_str = "UPTREND" if uptrend else "DOWNTREND"
                    bt_pf = BACKTEST_RESULTS.get(stock, {}).get("profit_factor", None)
                    pf_str = f"PF:{bt_pf:.2f}" if bt_pf is not None else "PF:?"
                    short_ok = stock in SHORT_ELIGIBLE
                    short_tag = "short:✓" if short_ok else "short:✗"
                    hdr = f"  DIAG {stock} [{tier_label} | {pf_str} | {short_tag}]  [{regime_str}]"

                    if uptrend:
                        r_gate = buy_bd.get("regime")
                        v_gate = buy_bd.get("volume")
                        if r_gate == "blocked":
                            print(f"{hdr}  BUY: regime blocked | SHORT: N/A (uptrend)")
                        elif v_gate == "too_low":
                            vol_ratio = (cur_volume / cur_avg_volume) if cur_avg_volume else 0
                            print(f"{hdr}  BUY: volume too low ({vol_ratio:.2f}x avg) | SHORT: N/A (uptrend)")
                        else:
                            ema_s  = buy_bd.get("ema",  0.0)
                            rsi_s  = buy_bd.get("rsi",  0.0)
                            bb_s   = buy_bd.get("bb",   0.0)
                            rs_s   = buy_bd.get("rs",   0.0)
                            orb_s  = buy_bd.get("orb",  0.0)
                            vwap_s = buy_bd.get("vwap", 0.0)
                            buy_gap = buy_score - buy_threshold
                            fire = ">>> FIRE <<<" if buy_gap >= 0 else ""
                            print(f"{hdr}")
                            print(f"    BUY   EMA:{ema_s:.1f}/3 RSI:{rsi_s:.1f}/2 BB:{bb_s:.1f}/2 "
                                  f"RS:{rs_s:.1f}[{_rs_label(rel_strength)}] "
                                  f"ORB:{orb_s:.1f}[{_orb_label(orb_data)}] "
                                  f"VWAP:{vwap_s:.1f}[{_vwap_label(price, vwap_data)}]")
                            print(f"          Score:{buy_score:.1f}/{signals.MAX_SCORE}  "
                                  f"thresh:{buy_threshold:.1f}  gap:{buy_gap:+.1f}  {fire}")
                            print(f"    SHORT N/A (regime: uptrend)")
                    else:
                        r_gate = short_bd.get("regime")
                        v_gate = short_bd.get("volume")
                        if r_gate == "blocked":
                            print(f"{hdr}  SHORT: regime blocked | BUY: N/A (downtrend)")
                        elif v_gate == "too_low":
                            vol_ratio = (cur_volume / cur_avg_volume) if cur_avg_volume else 0
                            print(f"{hdr}  SHORT: volume too low ({vol_ratio:.2f}x avg) | BUY: N/A (downtrend)")
                        else:
                            ema_s  = short_bd.get("ema",  0.0)
                            rsi_s  = short_bd.get("rsi",  0.0)
                            bb_s   = short_bd.get("bb",   0.0)
                            rs_s   = short_bd.get("rs",   0.0)
                            orb_s  = short_bd.get("orb",  0.0)
                            vwap_s = short_bd.get("vwap", 0.0)
                            short_gap = short_score - short_threshold
                            if short_gap >= 0 and short_ok:
                                fire = ">>> FIRE <<<"
                            elif short_gap >= 0 and not short_ok:
                                fire = "score OK — not short-eligible"
                            else:
                                fire = ""
                            print(f"{hdr}")
                            print(f"    SHORT EMA:{ema_s:.1f}/3 RSI:{rsi_s:.1f}/2 BB:{bb_s:.1f}/2 "
                                  f"RS:{rs_s:.1f}[{_rs_label(rel_strength)}] "
                                  f"ORB:{orb_s:.1f}[{_orb_label(orb_data)}] "
                                  f"VWAP:{vwap_s:.1f}[{_vwap_label(price, vwap_data)}]")
                            print(f"          Score:{short_score:.1f}/{signals.MAX_SCORE}  "
                                  f"thresh:{short_threshold:.1f}  gap:{short_gap:+.1f}  {fire}")
                            print(f"    BUY   N/A (regime: downtrend)")

                can_buy = (buy_score >= buy_threshold and
                          not buys_blocked and
                          not at_max_longs and
                          sector_count < MAX_POSITIONS_PER_SECTOR)

                can_short = (short_score >= short_threshold and
                            not shorts_blocked and
                            not at_max_shorts and
                            not uptrend and
                            stock in SHORT_ELIGIBLE and
                            sector_count < MAX_POSITIONS_PER_SECTOR)

                # If both qualify, pick the stronger signal
                if can_buy and can_short:
                    if buy_score - buy_threshold > short_score - short_threshold:
                        can_short = False
                    else:
                        can_buy = False

                if can_buy:
                    api.submit_order(
                        symbol=stock,
                        qty=qty,
                        side="buy",
                        type="market",
                        time_in_force="day"
                    )
                    PENDING_ORDERS.add(stock)
                    ENTRY_PRICES[stock] = {"price": price, "atr": atr, "direction": "long"}
                    ENTRY_BARS[stock] = len(bars)
                    save_entry_prices()
                    est_cost = calculate_trade_cost(stock, price, qty, "buy")
                    log_trade(stock, "BUY", price, rsi, 0,
                             fast, slow, bb_upper, bb_lower,
                             atr, qty, 0, est_cost)
                    stop = price - (ATR_STOP_MULT * atr)
                    target = price + (ATR_PROFIT_MULT * atr)
                    print(f"  ✅ {stock} [{tier_label}] BUY score: {buy_score:.1f}/{buy_threshold} | "
                          f"Qty: {qty} | ${price:.2f} | "
                          f"Stop: ${stop:.2f} | Target: ${target:.2f} | {buy_bd}")

                elif can_short:
                    api.submit_order(
                        symbol=stock,
                        qty=qty,
                        side="sell",
                        type="market",
                        time_in_force="day"
                    )
                    PENDING_ORDERS.add(stock)
                    ENTRY_PRICES[stock] = {"price": price, "atr": atr, "direction": "short"}
                    ENTRY_BARS[stock] = len(bars)
                    save_entry_prices()
                    est_cost = calculate_trade_cost(stock, price, qty, "sell")
                    log_trade(stock, "SHORT", price, rsi, 0,
                             fast, slow, bb_upper, bb_lower,
                             atr, qty, 0, est_cost)
                    stop = price + (ATR_STOP_MULT * atr)
                    target = price - (ATR_PROFIT_MULT * atr)
                    print(f"  🔻 {stock} [{tier_label}] SHORT score: {short_score:.1f}/{short_threshold} | "
                          f"Qty: {qty} | ${price:.2f} | "
                          f"Stop: ${stop:.2f} | Target: ${target:.2f} | {short_bd}")

                else:
                    print(f"  ⏸ {stock} HOLD")

            # ═══════════════════════════════════════════════════
            # 5. HOLD — show position info
            # ═══════════════════════════════════════════════════
            elif position != 0 and stock in ENTRY_PRICES:
                data = ENTRY_PRICES[stock]
                if isinstance(data, dict):
                    entry = data["price"]
                    entry_atr = data["atr"]
                    direction = data.get("direction", "long")
                else:
                    entry = data
                    entry_atr = abs(entry * 0.005)
                    direction = "long"
                if direction == "long":
                    pnl_pct = (price - entry) / entry * 100
                    stop = entry - (ATR_STOP_MULT * entry_atr)
                    target = entry + (ATR_PROFIT_MULT * entry_atr)
                else:
                    pnl_pct = (entry - price) / entry * 100
                    stop = entry + (ATR_STOP_MULT * entry_atr)
                    target = entry - (ATR_PROFIT_MULT * entry_atr)
                dir_label = "LONG" if direction == "long" else "SHORT"
                print(f"  ⏸ {stock} HOLD {dir_label} | "
                      f"PnL: {pnl_pct:+.2f}% | "
                      f"Stop: ${stop:.2f} | "
                      f"Target: ${target:.2f}")
            else:
                print(f"  ⏸ {stock} HOLD")

        except Exception as e:
            print(f"  Error with {stock}: {e}")

    show_positions()
    show_performance()

    # ── Pairs diagnostic ──────────────────────────────────────────
    if DIAGNOSTIC_MODE:
        for pair_name in PAIRS:
            check_pairs_signal(pair_name)

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

# ── Startup summary ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("🚀 TRADING BOT — STARTUP SUMMARY")
print(f"{'='*60}")
if TEST_MODE:
    print("  ⚠️  TEST MODE ACTIVE — thresholds reduced by 1.0 (calibration only, NOT for live trading)")
if DIAGNOSTIC_MODE:
    print("  🔍 DIAGNOSTIC MODE ACTIVE — full score breakdowns printed each cycle")
print(f"\n  Signal Mode: GRADIENT SCORING v3 (signals.py, max score {signals.MAX_SCORE})")
print(f"  Defaults: EMA {FAST_MA}/{SLOW_MA} | RSI {RSI_OVERSOLD}/{RSI_OVERBOUGHT} | "
      f"BB {BB_PERIOD}/{BB_STD}")
print(f"  Default thresholds: Buy {DEFAULT_BUY_THRESHOLD} | Sell {DEFAULT_SELL_THRESHOLD}")
print(f"  Risk: Stop {ATR_STOP_MULT}x ATR | Target {ATR_PROFIT_MULT}x ATR | "
      f"Max {MAX_POSITIONS} longs, {MAX_SHORT_POSITIONS} shorts | "
      f"Daily limit ${DAILY_LOSS_LIMIT}")

print(f"\n  Tier 1 — Full size ({len(TIER1_STOCKS)} stocks):")
print(f"    {', '.join(TIER1_STOCKS)}")
print(f"\n  Tier 2 — 50% size ({len(TIER2_STOCKS)} stocks):")
print(f"    {', '.join(TIER2_STOCKS)}")
print(f"\n  Tier 3 — 25% size, monitor ({len(TIER3_STOCKS)} stocks):")
print(f"    {', '.join(TIER3_STOCKS)}")

print(f"\n  Total stocks monitored: {len(STOCKS)}")
print(f"  Short eligible: {', '.join(SHORT_ELIGIBLE)}")
print(f"\n  Awaiting backtest confirmation (NOT trading): {', '.join(PENDING_VALIDATION)}")

print(f"\n  Walk-forward params loaded: {len(STOCK_PARAMS)} stocks")
if STOCK_PARAMS:
    for _s, _p in STOCK_PARAMS.items():
        _t = _p.get("optimal_threshold", _p.get("score_threshold", DEFAULT_BUY_THRESHOLD))
        print(f"    {_s}: EMA {_p['fast']}/{_p['slow']} | Thresh: {_t:.1f}")
else:
    print("    ⚠️ None — using defaults for all stocks")

print(f"\n  Backtest results loaded: {len(BACKTEST_RESULTS)} stocks")
if BACKTEST_RESULTS:
    for s, r in BACKTEST_RESULTS.items():
        pf = r.get("profit_factor", 0)
        pf_str = f"PF={pf:.2f}" if pf < 100 else "PF=N/A"
        ok = "✅" if pf > 1.0 else "❌"
        print(f"    {ok} {s}: {pf_str}")
else:
    print("    ⚠️ None — no backtest filtering active")

print(f"\n{'='*60}")
print("  Checks every minute during market hours.")
print("  Press Ctrl+C to stop.")
print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan",    action="store_true", help="Run the pre-market scan")
    parser.add_argument("--trade",   action="store_true", help="Run the trading loop continuously")
    parser.add_argument("--summary", action="store_true", help="Show positions and performance")
    args = parser.parse_args()

    if args.scan:
        run_premarket_scan()
    elif args.trade:
        run_premarket_scan()
        while True:
            schedule.run_pending()
            run_bot()
            time.sleep(60)
    elif args.summary:
        show_positions()
        show_performance()
    else:
        # Default: run scan + one bot cycle (same as before)
        run_premarket_scan()
        run_bot()
