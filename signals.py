"""
Shared Signal Module
====================
Single source of truth for all indicator calculations, signal scoring,
and transaction cost modeling. Imported by bot.py, walk_forward.py,
backtest_minutes.py, overnight_pipeline.py, and permutation_test.py.

v2: Gradient scoring with 3 scored indicators + 2 hard gates.
    Dropped MACD (too slow on 1-min), VWAP (weak), Sentiment (unreliable).
    Regime is now a hard gate, not a scored component.
    Volume is a hard gate (blocks thin-volume entries).
    Max possible score: 7.0 (EMA 3.0 + RSI 2.0 + BB 2.0)
"""

import numpy as np
import pandas as pd

# ── Signal Weight Caps ────────────────────────────────────────────
W_EMA_MAX = 3.0    # EMA trend gradient (0 to 3.0 based on separation)
W_RSI_MAX = 2.0    # RSI gradient (0 to 2.0 based on how oversold/overbought)
W_BB_MAX  = 2.0    # Bollinger %B gradient (0 to 2.0 based on band position)

# Max possible score: 7.0

# ── Tier Thresholds ───────────────────────────────────────────────
# Lower than v1 because max score is 7.0 (was 13.5)
BUY_THRESHOLD_T1  = 3.5    # Tier 1: proven edge
BUY_THRESHOLD_T2  = 4.0    # Tier 2: promising
BUY_THRESHOLD_T3  = 4.5    # Tier 3: monitoring

SELL_THRESHOLD_T1 = 3.0
SELL_THRESHOLD_T2 = 3.5
SELL_THRESHOLD_T3 = 4.0

SHORT_THRESHOLD_T1 = 4.5   # Shorts are more conservative
SHORT_THRESHOLD_T2 = 5.0

COVER_THRESHOLD_T1 = 3.0
COVER_THRESHOLD_T2 = 3.5

# ── Indicator Defaults ────────────────────────────────────────────
DEFAULT_FAST_MA = 4
DEFAULT_SLOW_MA = 10
DEFAULT_RSI_PERIOD = 7
DEFAULT_RSI_OVERSOLD = 30
DEFAULT_RSI_OVERBOUGHT = 65
DEFAULT_BB_PERIOD = 15
DEFAULT_BB_STD = 1.5
DEFAULT_ATR_PERIOD = 14

# ── Transaction Cost Model ────────────────────────────────────────
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
#  INDICATOR FUNCTIONS (current value for live trading)
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
#  SCORING FUNCTIONS (v2 — gradient scoring)
# ═══════════════════════════════════════════════════════════════════

def _ema_buy_gradient(fast_val, slow_val):
    """EMA trend gradient: 0 to 3.0 based on percentage separation."""
    if fast_val <= slow_val or slow_val == 0:
        return 0.0
    sep = (fast_val - slow_val) / slow_val
    return min(sep / 0.001, W_EMA_MAX)


def _ema_sell_gradient(fast_val, slow_val):
    """EMA bearish gradient: 0 to 3.0 based on how far fast is below slow."""
    if fast_val >= slow_val or slow_val == 0:
        return 0.0
    sep = (slow_val - fast_val) / slow_val
    return min(sep / 0.001, W_EMA_MAX)


def _rsi_oversold_gradient(rsi_val, rsi_buy_level):
    """RSI oversold gradient: 0 to 2.0 based on depth below buy level."""
    if rsi_val >= rsi_buy_level:
        return 0.0
    depth = (rsi_buy_level - rsi_val) / 15.0  # Normalize: 15 pts below = full score
    return min(depth, 1.0) * W_RSI_MAX


def _rsi_overbought_gradient(rsi_val, rsi_sell_level):
    """RSI overbought gradient: 0 to 2.0 based on how far above sell level."""
    if rsi_val <= rsi_sell_level:
        return 0.0
    excess = (rsi_val - rsi_sell_level) / 15.0
    return min(excess, 1.0) * W_RSI_MAX


def _bb_lower_gradient(price, bb_lower, bb_upper):
    """BB buy gradient: 0 to 2.0 based on %B position (lower = higher score)."""
    if bb_lower is None or bb_upper is None:
        return 0.0
    bb_range = bb_upper - bb_lower
    if bb_range <= 0:
        return 0.0
    pct_b = (price - bb_lower) / bb_range  # 0 = at lower, 1 = at upper
    if pct_b >= 0.3:
        return 0.0
    return (0.3 - pct_b) / 0.3 * W_BB_MAX


def _bb_upper_gradient(price, bb_lower, bb_upper):
    """BB sell gradient: 0 to 2.0 based on %B position (higher = higher score)."""
    if bb_lower is None or bb_upper is None:
        return 0.0
    bb_range = bb_upper - bb_lower
    if bb_range <= 0:
        return 0.0
    pct_b = (price - bb_lower) / bb_range
    if pct_b <= 0.7:
        return 0.0
    return (pct_b - 0.7) / 0.3 * W_BB_MAX


def calculate_buy_score(fast_val, slow_val, rsi_val, rsi_buy_level,
                        price, bb_lower, bb_upper=None,
                        regime_uptrend=True,
                        current_volume=None, avg_volume=None,
                        **kwargs):
    """
    Calculate gradient buy score for a long entry.
    Returns (score, breakdown_dict).

    Hard gates: regime must be uptrend, volume must not be thin.
    Scored: EMA separation (0-3), RSI depth (0-2), BB %B position (0-2).
    Max possible: 7.0
    """
    score = 0.0
    breakdown = {}

    # HARD GATE: Regime — no longs in downtrend
    if not regime_uptrend:
        return 0.0, {"regime": "blocked"}

    # HARD GATE: Volume — no entries on thin volume
    if current_volume is not None and avg_volume is not None and avg_volume > 0:
        if current_volume < avg_volume * 0.5:
            return 0.0, {"volume": "too_low"}

    # 1. EMA trend gradient (0 to 3.0)
    ema_score = _ema_buy_gradient(fast_val, slow_val)
    if ema_score > 0:
        score += ema_score
        breakdown["ema"] = round(ema_score, 2)

    # 2. RSI oversold gradient (0 to 2.0)
    rsi_score = _rsi_oversold_gradient(rsi_val, rsi_buy_level)
    if rsi_score > 0:
        score += rsi_score
        breakdown["rsi"] = round(rsi_score, 2)

    # 3. Bollinger Band %B gradient (0 to 2.0)
    bb_score = _bb_lower_gradient(price, bb_lower, bb_upper)
    if bb_score > 0:
        score += bb_score
        breakdown["bb"] = round(bb_score, 2)

    return score, breakdown


def calculate_sell_score(fast_val, slow_val, rsi_val, rsi_sell_level,
                         price, bb_upper, bb_lower=None, **kwargs):
    """
    Calculate gradient sell score for closing a long.
    No regime gate — always exit deteriorating positions.
    Max possible: 7.0
    """
    score = 0.0
    breakdown = {}

    ema_score = _ema_sell_gradient(fast_val, slow_val)
    if ema_score > 0:
        score += ema_score
        breakdown["ema"] = round(ema_score, 2)

    rsi_score = _rsi_overbought_gradient(rsi_val, rsi_sell_level)
    if rsi_score > 0:
        score += rsi_score
        breakdown["rsi"] = round(rsi_score, 2)

    bb_score = _bb_upper_gradient(price, bb_lower, bb_upper)
    if bb_score > 0:
        score += bb_score
        breakdown["bb"] = round(bb_score, 2)

    return score, breakdown


def calculate_short_score(fast_val, slow_val, rsi_val, rsi_sell_level,
                          price, bb_upper, bb_lower=None,
                          regime_downtrend=False,
                          current_volume=None, avg_volume=None,
                          **kwargs):
    """
    Calculate gradient short entry score (bearish mirror of buy).
    Hard gate: regime must be downtrend for shorts.
    Max possible: 7.0
    """
    score = 0.0
    breakdown = {}

    # HARD GATE: Regime — no shorts in uptrend
    if not regime_downtrend:
        return 0.0, {"regime": "blocked"}

    # HARD GATE: Volume
    if current_volume is not None and avg_volume is not None and avg_volume > 0:
        if current_volume < avg_volume * 0.5:
            return 0.0, {"volume": "too_low"}

    # 1. EMA bearish gradient (0 to 3.0)
    ema_score = _ema_sell_gradient(fast_val, slow_val)
    if ema_score > 0:
        score += ema_score
        breakdown["ema"] = round(ema_score, 2)

    # 2. RSI overbought gradient (0 to 2.0)
    rsi_score = _rsi_overbought_gradient(rsi_val, rsi_sell_level)
    if rsi_score > 0:
        score += rsi_score
        breakdown["rsi"] = round(rsi_score, 2)

    # 3. BB upper gradient (0 to 2.0)
    bb_score = _bb_upper_gradient(price, bb_lower, bb_upper)
    if bb_score > 0:
        score += bb_score
        breakdown["bb"] = round(bb_score, 2)

    return score, breakdown


def calculate_cover_score(fast_val, slow_val, rsi_val, rsi_buy_level,
                          price, bb_lower, bb_upper=None, **kwargs):
    """
    Calculate gradient score for closing a short (covering).
    No regime gate — always exit deteriorating shorts.
    Max possible: 7.0
    """
    score = 0.0
    breakdown = {}

    ema_score = _ema_buy_gradient(fast_val, slow_val)
    if ema_score > 0:
        score += ema_score
        breakdown["ema"] = round(ema_score, 2)

    rsi_score = _rsi_oversold_gradient(rsi_val, rsi_buy_level)
    if rsi_score > 0:
        score += rsi_score
        breakdown["rsi"] = round(rsi_score, 2)

    bb_score = _bb_lower_gradient(price, bb_lower, bb_upper)
    if bb_score > 0:
        score += bb_score
        breakdown["bb"] = round(bb_score, 2)

    return score, breakdown


# ── Threshold helpers ─────────────────────────────────────────────

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
