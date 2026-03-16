"""
Shared Signal Module
====================
Single source of truth for all indicator calculations, signal scoring,
and transaction cost modeling. Imported by bot.py, walk_forward.py,
backtest_minutes.py, overnight_pipeline.py, and permutation_test.py.
"""

import numpy as np
import pandas as pd

# ── Signal Weights ──────────────────────────────────────────────────
# Buy signal weights (long entry)
W_EMA       = 3.0   # EMA trend (fast > slow)
W_RSI       = 2.0   # RSI oversold
W_BB        = 2.0   # Price <= BB lower band
W_MACD      = 1.5   # MACD bullish crossover
W_VWAP      = 1.0   # Price below VWAP
W_REGIME    = 2.5   # SPY regime (uptrend for longs, downtrend for shorts)
W_SENTIMENT = 0.5   # News sentiment
W_VOLUME    = 1.0   # Volume above 20-period average
W_VOLUME_PENALTY = -0.5  # Volume below 50% of average (low conviction)

# Max possible: 13.5 (13.0 without sentiment in backtests)

# ── Tier Thresholds ─────────────────────────────────────────────────
BUY_THRESHOLD_T1  = 6.0    # Tier 1: proven edge
BUY_THRESHOLD_T2  = 7.0    # Tier 2: promising
BUY_THRESHOLD_T3  = 8.0    # Tier 3: monitoring

SELL_THRESHOLD_T1 = 5.5
SELL_THRESHOLD_T2 = 6.0
SELL_THRESHOLD_T3 = 7.0

SHORT_THRESHOLD_T1 = 7.0   # Shorts are more conservative
SHORT_THRESHOLD_T2 = 8.0

COVER_THRESHOLD_T1 = 5.5
COVER_THRESHOLD_T2 = 6.0

# ── Indicator Defaults ──────────────────────────────────────────────
DEFAULT_FAST_MA = 4
DEFAULT_SLOW_MA = 10
DEFAULT_RSI_PERIOD = 7
DEFAULT_RSI_OVERSOLD = 30
DEFAULT_RSI_OVERBOUGHT = 65
DEFAULT_BB_PERIOD = 15
DEFAULT_BB_STD = 1.5
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_ATR_PERIOD = 14
DEFAULT_REGIME_EMA = 50

# ── Transaction Cost Model ──────────────────────────────────────────
SPREAD_ESTIMATES = {
    "AAPL": 0.02, "MSFT": 0.02, "NVDA": 0.03, "GOOGL": 0.05,
    "AMZN": 0.04, "TSLA": 0.05, "META": 0.03, "AMD": 0.03,
    "NFLX": 0.05, "CRM": 0.04, "JPM": 0.02, "BAC": 0.01,
    "GS": 0.03, "V": 0.03, "SPY": 0.01, "QQQ": 0.01,
    "XOM": 0.02, "CVX": 0.02, "JNJ": 0.02, "PFE": 0.01,
    "UNH": 0.05, "WMT": 0.02, "COST": 0.05, "NKE": 0.03,
    "INTC": 0.02, "QCOM": 0.03,
    # New stocks
    "UBER": 0.02, "PLTR": 0.02, "COIN": 0.05, "SHOP": 0.04,
    "SQ": 0.03, "ROKU": 0.04, "ABNB": 0.04, "PYPL": 0.03,
    "SPOT": 0.03, "ZM": 0.04, "HOOD": 0.02,
    # Pipeline candidates
    "AVGO": 0.05, "LLY": 0.08, "MA": 0.03,
    "PANW": 0.05, "CRWD": 0.05, "SNOW": 0.05,
    "DDOG": 0.04, "NET": 0.03, "ADBE": 0.05,
    "NOW": 0.05, "ORCL": 0.03, "MS": 0.03,
    "BLK": 0.08, "SCHW": 0.02, "HD": 0.03,
    "MCD": 0.03, "SBUX": 0.02,
}
DEFAULT_SPREAD = 0.03
SLIPPAGE_MULT = 0.5
SEC_FEE_RATE = 0.0000278


# ═══════════════════════════════════════════════════════════════════
#  INDICATOR FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def calculate_ema(close, span):
    """Calculate EMA series for the given span."""
    return close.ewm(span=span, adjust=False).mean()


def get_moving_averages(bars, fast_span=None, slow_span=None):
    """Return (fast_ema, slow_ema) current values."""
    fast_span = fast_span or DEFAULT_FAST_MA
    slow_span = slow_span or DEFAULT_SLOW_MA
    if len(bars) < slow_span:
        return None, None
    close = bars["close"]
    fast = close.ewm(span=fast_span, adjust=False).mean().iloc[-1]
    slow = close.ewm(span=slow_span, adjust=False).mean().iloc[-1]
    return fast, slow


def get_rsi(bars, period=None):
    """Return current RSI value."""
    period = period or DEFAULT_RSI_PERIOD
    if len(bars) < period + 1:
        return None
    close = bars["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_bollinger_bands(bars, bb_period=None, bb_std=None):
    """Return (upper, middle, lower) Bollinger Band values."""
    bb_period = bb_period or DEFAULT_BB_PERIOD
    bb_std = bb_std or DEFAULT_BB_STD
    if len(bars) < bb_period:
        return None, None, None
    close = bars["close"]
    middle = close.rolling(window=bb_period).mean().iloc[-1]
    std = close.rolling(window=bb_period).std().iloc[-1]
    upper = middle + (bb_std * std)
    lower = middle - (bb_std * std)
    return upper, middle, lower


def get_macd(bars, fast=None, slow=None, signal=None):
    """Return (macd_line, signal_line, histogram) current values."""
    fast = fast or DEFAULT_MACD_FAST
    slow = slow or DEFAULT_MACD_SLOW
    signal = signal or DEFAULT_MACD_SIGNAL
    if len(bars) < slow + signal:
        return None, None, None
    close = bars["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]


def get_atr(bars, period=None):
    """Return current ATR value."""
    period = period or DEFAULT_ATR_PERIOD
    if len(bars) < period + 1:
        return None
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean().iloc[-1]
    return atr


def calculate_vwap(bars):
    """Calculate VWAP from bar data. Returns current VWAP value or None."""
    if len(bars) < 1:
        return None
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_tp_vol = (typical_price * bars["volume"]).cumsum()
    cum_vol = bars["volume"].cumsum()
    if cum_vol.iloc[-1] == 0:
        return None
    vwap = cum_tp_vol / cum_vol
    return vwap.iloc[-1]


# ── Backtest indicator series (full series, not just current value) ──

def calculate_rsi_series(close, period=None):
    """Return full RSI series for backtesting."""
    period = period or DEFAULT_RSI_PERIOD
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_bb_series(close, period=None, std_dev=None):
    """Return (upper, middle, lower) Bollinger Band series."""
    period = period or DEFAULT_BB_PERIOD
    std_dev = std_dev or DEFAULT_BB_STD
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return upper, middle, lower


def calculate_macd_series(close, fast=None, slow=None, signal=None):
    """Return (macd_line, signal_line, histogram) series."""
    fast = fast or DEFAULT_MACD_FAST
    slow = slow or DEFAULT_MACD_SLOW
    signal = signal or DEFAULT_MACD_SIGNAL
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_atr_series(bars, period=None):
    """Return full ATR series for backtesting."""
    period = period or DEFAULT_ATR_PERIOD
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def calculate_vwap_series(bars):
    """Return full VWAP series for backtesting. Resets each day."""
    if "volume" not in bars.columns or len(bars) < 1:
        return pd.Series(np.nan, index=bars.index)
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    dates = bars.index.date if hasattr(bars.index, 'date') else pd.Series(range(len(bars)))
    vwap = pd.Series(np.nan, index=bars.index)
    for date in pd.unique(dates):
        mask = bars.index.date == date if hasattr(bars.index, 'date') else True
        day_tp = typical_price[mask]
        day_vol = bars["volume"][mask]
        cum_tp_vol = (day_tp * day_vol).cumsum()
        cum_vol = day_vol.cumsum()
        day_vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        vwap[mask] = day_vwap
    return vwap


# ═══════════════════════════════════════════════════════════════════
#  SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def calculate_buy_score(fast_val, slow_val, rsi_val, rsi_buy_level,
                        price, bb_lower, macd_line, signal_line,
                        vwap, regime_uptrend, sentiment=0,
                        current_volume=None, avg_volume=None):
    """
    Calculate weighted buy score for a long entry.
    Returns (score, breakdown_dict).
    """
    score = 0.0
    breakdown = {}

    # EMA trend
    if fast_val > slow_val:
        score += W_EMA
        breakdown["ema"] = W_EMA

    # RSI oversold
    if rsi_val < rsi_buy_level:
        score += W_RSI
        breakdown["rsi"] = W_RSI

    # Bollinger Band lower
    if bb_lower is not None and price <= bb_lower:
        score += W_BB
        breakdown["bb"] = W_BB

    # MACD bullish
    if macd_line is not None and signal_line is not None and macd_line > signal_line:
        score += W_MACD
        breakdown["macd"] = W_MACD

    # VWAP
    if vwap is not None and not (isinstance(vwap, float) and np.isnan(vwap)) and price < vwap:
        score += W_VWAP
        breakdown["vwap"] = W_VWAP

    # Regime
    if regime_uptrend:
        score += W_REGIME
        breakdown["regime"] = W_REGIME

    # Sentiment
    if sentiment > 0.15:
        score += W_SENTIMENT
        breakdown["sentiment"] = W_SENTIMENT

    # Volume confirmation
    if current_volume is not None and avg_volume is not None and avg_volume > 0:
        if current_volume > avg_volume:
            score += W_VOLUME
            breakdown["volume"] = W_VOLUME
        elif current_volume < avg_volume * 0.5:
            score += W_VOLUME_PENALTY
            breakdown["volume"] = W_VOLUME_PENALTY

    return score, breakdown


def calculate_sell_score(fast_val, slow_val, rsi_val, rsi_sell_level,
                         price, bb_upper, macd_line, signal_line, vwap):
    """
    Calculate weighted sell score for closing a long position.
    No regime in sell scoring — always exit deteriorating positions.
    Returns (score, breakdown_dict).
    """
    score = 0.0
    breakdown = {}

    if fast_val < slow_val:
        score += W_EMA
        breakdown["ema"] = W_EMA

    if rsi_val > rsi_sell_level:
        score += W_RSI
        breakdown["rsi"] = W_RSI

    if bb_upper is not None and price >= bb_upper:
        score += W_BB
        breakdown["bb"] = W_BB

    if macd_line is not None and signal_line is not None and macd_line < signal_line:
        score += W_MACD
        breakdown["macd"] = W_MACD

    if vwap is not None and not (isinstance(vwap, float) and np.isnan(vwap)) and price > vwap:
        score += W_VWAP
        breakdown["vwap"] = W_VWAP

    return score, breakdown


def calculate_short_score(fast_val, slow_val, rsi_val, rsi_sell_level,
                          price, bb_upper, macd_line, signal_line,
                          vwap, regime_downtrend, sentiment=0,
                          current_volume=None, avg_volume=None):
    """
    Calculate weighted short entry score (mirror of buy score, inverted).
    Returns (score, breakdown_dict).
    """
    score = 0.0
    breakdown = {}

    # EMA bearish
    if fast_val < slow_val:
        score += W_EMA
        breakdown["ema"] = W_EMA

    # RSI overbought
    if rsi_val > rsi_sell_level:
        score += W_RSI
        breakdown["rsi"] = W_RSI

    # Price at/above BB upper (overextended)
    if bb_upper is not None and price >= bb_upper:
        score += W_BB
        breakdown["bb"] = W_BB

    # MACD bearish
    if macd_line is not None and signal_line is not None and macd_line < signal_line:
        score += W_MACD
        breakdown["macd"] = W_MACD

    # Above VWAP (overvalued)
    if vwap is not None and not (isinstance(vwap, float) and np.isnan(vwap)) and price > vwap:
        score += W_VWAP
        breakdown["vwap"] = W_VWAP

    # Regime downtrend (tailwind for shorts)
    if regime_downtrend:
        score += W_REGIME
        breakdown["regime"] = W_REGIME

    # Negative sentiment
    if sentiment < -0.15:
        score += W_SENTIMENT
        breakdown["sentiment"] = W_SENTIMENT

    # Volume confirmation
    if current_volume is not None and avg_volume is not None and avg_volume > 0:
        if current_volume > avg_volume:
            score += W_VOLUME
            breakdown["volume"] = W_VOLUME
        elif current_volume < avg_volume * 0.5:
            score += W_VOLUME_PENALTY
            breakdown["volume"] = W_VOLUME_PENALTY

    return score, breakdown


def calculate_cover_score(fast_val, slow_val, rsi_val, rsi_buy_level,
                          price, bb_lower, macd_line, signal_line, vwap):
    """
    Calculate score for closing a short position (covering).
    Returns (score, breakdown_dict).
    """
    score = 0.0
    breakdown = {}

    if fast_val > slow_val:
        score += W_EMA
        breakdown["ema"] = W_EMA

    if rsi_val < rsi_buy_level:
        score += W_RSI
        breakdown["rsi"] = W_RSI

    if bb_lower is not None and price <= bb_lower:
        score += W_BB
        breakdown["bb"] = W_BB

    if macd_line is not None and signal_line is not None and macd_line > signal_line:
        score += W_MACD
        breakdown["macd"] = W_MACD

    if vwap is not None and not (isinstance(vwap, float) and np.isnan(vwap)) and price < vwap:
        score += W_VWAP
        breakdown["vwap"] = W_VWAP

    return score, breakdown


# ── Threshold helpers ───────────────────────────────────────────────

def get_buy_threshold(stock, tier1_stocks, tier2_stocks):
    if stock in tier1_stocks:
        return BUY_THRESHOLD_T1
    elif stock in tier2_stocks:
        return BUY_THRESHOLD_T2
    else:
        return BUY_THRESHOLD_T3


def get_sell_threshold(stock, tier1_stocks, tier2_stocks):
    if stock in tier1_stocks:
        return SELL_THRESHOLD_T1
    elif stock in tier2_stocks:
        return SELL_THRESHOLD_T2
    else:
        return SELL_THRESHOLD_T3


def get_short_threshold(stock, tier1_stocks, tier2_stocks):
    if stock in tier1_stocks:
        return SHORT_THRESHOLD_T1
    elif stock in tier2_stocks:
        return SHORT_THRESHOLD_T2
    else:
        return 99.0  # Tier 3 cannot short


def get_cover_threshold(stock, tier1_stocks, tier2_stocks):
    if stock in tier1_stocks:
        return COVER_THRESHOLD_T1
    elif stock in tier2_stocks:
        return COVER_THRESHOLD_T2
    else:
        return 99.0


# ═══════════════════════════════════════════════════════════════════
#  TRANSACTION COST MODEL
# ═══════════════════════════════════════════════════════════════════

def calculate_trade_cost(stock, price, quantity, side, cost_model_enabled=True):
    """Calculate total transaction cost for a single trade."""
    if not cost_model_enabled:
        return 0.0
    half_spread = SPREAD_ESTIMATES.get(stock, DEFAULT_SPREAD)
    spread_cost = half_spread * quantity
    slippage_cost = spread_cost * SLIPPAGE_MULT
    sec_fee = 0.0
    if side == "sell":
        sec_fee = price * quantity * SEC_FEE_RATE
    return spread_cost + slippage_cost + sec_fee
