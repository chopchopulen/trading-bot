"""
Shared Signal Module
====================
Single source of truth for all indicator calculations, signal scoring,
and transaction cost modeling. Imported by bot.py, walk_forward.py,
backtest_minutes.py, overnight_pipeline.py, and permutation_test.py.

v2: Gradient scoring — 3 scored indicators + 2 hard gates.
    Dropped MACD (too slow on 1-min), VWAP (weak), Sentiment (unreliable).
    Regime is a hard gate, not scored. Volume is a hard gate.

v3 additions:
    - Relative Strength vs SPY  (±1.5 pts buy, ±1.5 pts short)
    - Opening Range Breakout    (±2.0 pts buy, ±2.0 pts short)
    - VWAP band mean-reversion  (±1.5 pts buy, ±2.0 pts short)
    - get_relative_strength()   — 5-bar rolling return vs SPY
    - get_vwap_bands()          — VWAP + rolling std dev bands

    Max possible buy score:   12.0  (EMA 3 + RSI 2 + BB 2 + RS 1.5 + ORB 2 + VWAP 1.5)
    Max possible short score: 12.5  (EMA 3 + RSI 2 + BB 2 + RS 1.5 + ORB 2 + VWAP 2.0)
"""

import numpy as np
import pandas as pd

# ── Signal Weight Caps ────────────────────────────────────────────
W_EMA_MAX      = 3.0    # EMA trend gradient
W_RSI_MAX      = 2.0    # RSI oversold/overbought gradient
W_BB_MAX       = 2.0    # Bollinger %B gradient
W_RS_MAX       = 1.5    # Relative strength vs SPY bonus (penalty is -1.0)
W_ORB_MAX      = 2.0    # Opening Range Breakout bonus (penalty is -1.0)
W_VWAP_BUY_MAX = 1.5    # VWAP mean-reversion buy bonus (no penalty for basic)
W_VWAP_SHT_MAX = 2.0    # VWAP rejection short bonus (penalty is -1.0 below VWAP)

MAX_SCORE = 12.5  # Theoretical max (short score is highest)

# ── Tier Thresholds ───────────────────────────────────────────────
BUY_THRESHOLD_T1  = 3.5
BUY_THRESHOLD_T2  = 4.0
BUY_THRESHOLD_T3  = 4.5

SELL_THRESHOLD_T1 = 3.0
SELL_THRESHOLD_T2 = 3.5
SELL_THRESHOLD_T3 = 4.0

SHORT_THRESHOLD_T1 = 4.5
SHORT_THRESHOLD_T2 = 5.0

COVER_THRESHOLD_T1 = 3.0
COVER_THRESHOLD_T2 = 3.5

# ── Indicator Defaults ────────────────────────────────────────────
DEFAULT_FAST_MA        = 4
DEFAULT_SLOW_MA        = 10
DEFAULT_RSI_PERIOD     = 7
DEFAULT_RSI_OVERSOLD   = 30
DEFAULT_RSI_OVERBOUGHT = 65
DEFAULT_BB_PERIOD      = 15
DEFAULT_BB_STD         = 1.5
DEFAULT_ATR_PERIOD     = 14

# ── Transaction Cost Model ────────────────────────────────────────
SPREAD_ESTIMATES = {
    "AAPL": 0.02, "MSFT": 0.02, "NVDA": 0.03, "GOOGL": 0.05,
    "AMZN": 0.04, "TSLA": 0.05, "META": 0.03, "AMD": 0.03,
    "NFLX": 0.05, "CRM": 0.04, "JPM": 0.02, "BAC": 0.01,
    "GS": 0.03, "V": 0.03, "SPY": 0.01, "QQQ": 0.01,
    "XOM": 0.02, "CVX": 0.02, "JNJ": 0.02, "PFE": 0.01,
    "UNH": 0.05, "WMT": 0.02, "COST": 0.05, "NKE": 0.03,
    "INTC": 0.02, "QCOM": 0.03,
    "UBER": 0.02, "PLTR": 0.02, "COIN": 0.05, "SHOP": 0.04,
    "SQ": 0.03, "ROKU": 0.04, "ABNB": 0.04, "PYPL": 0.03,
    "SPOT": 0.03, "ZM": 0.04, "HOOD": 0.02,
    "AVGO": 0.05, "LLY": 0.08, "MA": 0.03,
    "PANW": 0.05, "CRWD": 0.05, "SNOW": 0.05,
    "DDOG": 0.04, "NET": 0.03, "ADBE": 0.05,
    "NOW": 0.05, "ORCL": 0.03, "MS": 0.03,
    "BLK": 0.08, "SCHW": 0.02, "HD": 0.03,
    "MCD": 0.03, "SBUX": 0.02,
}
DEFAULT_SPREAD  = 0.03
SLIPPAGE_MULT   = 0.5
SEC_FEE_RATE    = 0.0000278


# ═══════════════════════════════════════════════════════════════════
#  INDICATOR FUNCTIONS — current value for live trading
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
    close    = bars["close"]
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_bollinger_bands(bars, bb_period=None, bb_std=None):
    """Return (upper, middle, lower) Bollinger Band values."""
    bb_period = bb_period or DEFAULT_BB_PERIOD
    bb_std    = bb_std    or DEFAULT_BB_STD
    if len(bars) < bb_period:
        return None, None, None
    close  = bars["close"]
    middle = close.rolling(window=bb_period).mean().iloc[-1]
    std    = close.rolling(window=bb_period).std().iloc[-1]
    upper  = middle + (bb_std * std)
    lower  = middle - (bb_std * std)
    return upper, middle, lower


def get_atr(bars, period=None):
    """Return current ATR value."""
    period = period or DEFAULT_ATR_PERIOD
    if len(bars) < period + 1:
        return None
    high  = bars["high"]
    low   = bars["low"]
    close = bars["close"]
    tr1   = high - low
    tr2   = abs(high - close.shift(1))
    tr3   = abs(low  - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean().iloc[-1]


def calculate_vwap(bars):
    """Calculate VWAP from bar data. Returns current VWAP value or None."""
    if len(bars) < 1:
        return None
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_tp_vol    = (typical_price * bars["volume"]).cumsum()
    cum_vol       = bars["volume"].cumsum()
    if cum_vol.iloc[-1] == 0:
        return None
    return (cum_tp_vol / cum_vol).iloc[-1]


def get_relative_strength(stock_bars, spy_bars, window=5):
    """
    Compare rolling 5-bar returns of stock vs SPY.

    Returns score in [-2.0, +2.0]:
      +2.0  stock strongly outperforming SPY  → long signal
      +0.5  threshold above which stock is "relatively strong"
      -0.5  threshold below which stock is "relatively weak"
      -2.0  stock strongly underperforming SPY → short signal

    Normalization: 0.2% relative difference = 1.0 score unit.
    Returns None if insufficient data.
    """
    try:
        if spy_bars is None or len(stock_bars) < window + 1 or len(spy_bars) < window + 1:
            return None
        stock_close = stock_bars["close"]
        spy_close   = spy_bars["close"]
        stock_ret   = (float(stock_close.iloc[-1]) - float(stock_close.iloc[-window])) / float(stock_close.iloc[-window])
        spy_ret     = (float(spy_close.iloc[-1])   - float(spy_close.iloc[-window]))   / float(spy_close.iloc[-window])
        rel_diff    = stock_ret - spy_ret
        score       = rel_diff / 0.002        # 0.2% diff → 1.0 score
        return max(-2.0, min(2.0, score))
    except Exception:
        return None


def get_vwap_bands(bars, window=20):
    """
    Calculate VWAP and rolling std dev bands from bar data.

    Returns dict or None:
    {
        "vwap":             float,
        "upper_1std":       float | None,
        "upper_2std":       float | None,
        "lower_1std":       float | None,
        "lower_2std":       float | None,
        "last_candle_red":  bool,   # close < open
        "last_candle_green": bool,  # close > open
    }
    """
    try:
        if len(bars) < window + 1 or "volume" not in bars.columns:
            return None
        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
        cum_tp_vol    = (typical_price * bars["volume"]).cumsum()
        cum_vol       = bars["volume"].cumsum()
        if cum_vol.iloc[-1] == 0:
            return None
        vwap_series = cum_tp_vol / cum_vol
        vwap        = float(vwap_series.iloc[-1])
        deviation   = typical_price - vwap_series
        std_val     = float(deviation.rolling(window=window).std().iloc[-1])
        last_open   = float(bars["open"].iloc[-1])
        last_close  = float(bars["close"].iloc[-1])
        result = {
            "vwap":              vwap,
            "last_candle_red":   last_close < last_open,
            "last_candle_green": last_close > last_open,
        }
        if pd.isna(std_val) or std_val <= 0:
            result.update({"upper_1std": None, "upper_2std": None,
                           "lower_1std": None, "lower_2std": None})
        else:
            result.update({
                "upper_1std": vwap + std_val,
                "upper_2std": vwap + 2 * std_val,
                "lower_1std": vwap - std_val,
                "lower_2std": vwap - 2 * std_val,
            })
        return result
    except Exception:
        return None


# ── Backtest series helpers (full series, not just current value) ──

def calculate_rsi_series(close, period=None):
    """Return full RSI series for backtesting."""
    period   = period or DEFAULT_RSI_PERIOD
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_bb_series(close, period=None, std_dev=None):
    """Return (upper, middle, lower) Bollinger Band series."""
    period  = period  or DEFAULT_BB_PERIOD
    std_dev = std_dev or DEFAULT_BB_STD
    middle  = close.rolling(window=period).mean()
    std     = close.rolling(window=period).std()
    upper   = middle + (std_dev * std)
    lower   = middle - (std_dev * std)
    return upper, middle, lower


def calculate_atr_series(bars, period=None):
    """Return full ATR series for backtesting."""
    period = period or DEFAULT_ATR_PERIOD
    high   = bars["high"]
    low    = bars["low"]
    close  = bars["close"]
    tr1    = high - low
    tr2    = abs(high - close.shift(1))
    tr3    = abs(low  - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def calculate_vwap_series(bars):
    """Return full VWAP series for backtesting. Resets each trading day."""
    if "volume" not in bars.columns or len(bars) < 1:
        return pd.Series(np.nan, index=bars.index)
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    dates = bars.index.date if hasattr(bars.index, 'date') else pd.Series(range(len(bars)))
    vwap  = pd.Series(np.nan, index=bars.index)
    for date in pd.unique(dates):
        mask       = bars.index.date == date if hasattr(bars.index, 'date') else True
        day_tp     = typical_price[mask]
        day_vol    = bars["volume"][mask]
        cum_tp_vol = (day_tp * day_vol).cumsum()
        cum_vol    = day_vol.cumsum()
        day_vwap   = cum_tp_vol / cum_vol.replace(0, np.nan)
        vwap[mask] = day_vwap
    return vwap


# ═══════════════════════════════════════════════════════════════════
#  GRADIENT HELPERS
# ═══════════════════════════════════════════════════════════════════

def _ema_buy_gradient(fast_val, slow_val):
    """EMA trend gradient: 0 to 3.0 based on % separation (fast > slow)."""
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
    depth = (rsi_buy_level - rsi_val) / 15.0
    return min(depth, 1.0) * W_RSI_MAX


def _rsi_overbought_gradient(rsi_val, rsi_sell_level):
    """RSI overbought gradient: 0 to 2.0 based on how far above sell level."""
    if rsi_val <= rsi_sell_level:
        return 0.0
    excess = (rsi_val - rsi_sell_level) / 15.0
    return min(excess, 1.0) * W_RSI_MAX


def _bb_lower_gradient(price, bb_lower, bb_upper):
    """BB buy gradient: 0 to 2.0. Scores when price is in lower 30% of band."""
    if bb_lower is None or bb_upper is None:
        return 0.0
    bb_range = bb_upper - bb_lower
    if bb_range <= 0:
        return 0.0
    pct_b = (price - bb_lower) / bb_range
    if pct_b >= 0.3:
        return 0.0
    return (0.3 - pct_b) / 0.3 * W_BB_MAX


def _bb_upper_gradient(price, bb_lower, bb_upper):
    """BB sell gradient: 0 to 2.0. Scores when price is in upper 30% of band."""
    if bb_lower is None or bb_upper is None:
        return 0.0
    bb_range = bb_upper - bb_lower
    if bb_range <= 0:
        return 0.0
    pct_b = (price - bb_lower) / bb_range
    if pct_b <= 0.7:
        return 0.0
    return (pct_b - 0.7) / 0.3 * W_BB_MAX


# ═══════════════════════════════════════════════════════════════════
#  SCORING FUNCTIONS (v3 — gradient + contextual signals)
# ═══════════════════════════════════════════════════════════════════

def calculate_buy_score(fast_val, slow_val, rsi_val, rsi_buy_level,
                        price, bb_lower, bb_upper=None,
                        regime_uptrend=True,
                        current_volume=None, avg_volume=None,
                        rel_strength=None,
                        orb_data=None,
                        vwap_data=None,
                        **kwargs):
    """
    Calculate gradient buy score for a long entry.
    Returns (score, breakdown_dict).

    Hard gates:  regime must be uptrend, volume must not be thin.
    Base:        EMA (0-3), RSI (0-2), BB (0-2)
    Contextual:  Relative strength vs SPY (±1.5),
                 ORB breakout (±2.0),
                 VWAP lower-band bounce (±1.5)
    Max possible: 12.0
    """
    score    = 0.0
    breakdown = {}

    # ── HARD GATE: Regime ────────────────────────────────────────
    if not regime_uptrend:
        return 0.0, {"regime": "blocked"}

    # ── HARD GATE: Volume ────────────────────────────────────────
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

    # 4. Relative strength vs SPY (±1.5 / -1.0)
    if rel_strength is not None:
        if rel_strength > 0.5:
            score += W_RS_MAX
            breakdown["rel_strength"] = W_RS_MAX
        elif rel_strength < -0.5:
            score -= 1.0
            breakdown["rel_strength"] = -1.0

    # 5. ORB breakout bonus (±2.0 / -1.0)
    if orb_data is not None and orb_data.get("set"):
        orb_high = orb_data.get("high")
        orb_low  = orb_data.get("low")
        vol_ok   = (current_volume is not None and avg_volume is not None
                    and avg_volume > 0 and current_volume >= avg_volume * 1.5)
        if orb_high is not None and price > orb_high and vol_ok:
            score += W_ORB_MAX
            breakdown["orb"] = W_ORB_MAX
        elif orb_low is not None and price < orb_low:
            score -= 1.0
            breakdown["orb"] = -1.0

    # 6. VWAP lower-band mean-reversion (±1.5 / +0.5)
    if vwap_data is not None and vwap_data.get("lower_1std") is not None:
        lower_2std = vwap_data.get("lower_2std")
        lower_1std = vwap_data.get("lower_1std")
        last_green = vwap_data.get("last_candle_green", False)
        if lower_2std is not None and price < lower_2std and last_green:
            score += W_VWAP_BUY_MAX
            breakdown["vwap"] = W_VWAP_BUY_MAX
        elif price < lower_1std:
            score += 0.5
            breakdown["vwap"] = 0.5

    return score, breakdown


def calculate_sell_score(fast_val, slow_val, rsi_val, rsi_sell_level,
                         price, bb_upper, bb_lower=None, **kwargs):
    """
    Calculate gradient sell score for closing a long.
    No regime gate — always allow exits.
    Uses base indicators only (max 7.0). Contextual signals not applied to exits.
    """
    score     = 0.0
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
                          rel_strength=None,
                          orb_data=None,
                          vwap_data=None,
                          **kwargs):
    """
    Calculate gradient short entry score (bearish mirror of buy).
    Hard gate: regime must be downtrend.
    Base:       EMA (0-3), RSI (0-2), BB (0-2)
    Contextual: Relative weakness vs SPY (±1.5 / -1.0),
                ORB breakdown (±2.0 / -1.0),
                VWAP upper-band rejection (±2.0 / -1.0)
    Max possible: 12.5
    """
    score     = 0.0
    breakdown = {}

    # ── HARD GATE: Regime ────────────────────────────────────────
    if not regime_downtrend:
        return 0.0, {"regime": "blocked"}

    # ── HARD GATE: Volume ────────────────────────────────────────
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

    # 4. Relative weakness vs SPY (±1.5 / -1.0)
    # Negative rel_strength means stock underperforming SPY — short signal
    if rel_strength is not None:
        if rel_strength < -0.5:
            score += W_RS_MAX
            breakdown["rel_strength"] = W_RS_MAX
        elif rel_strength > 0.5:
            score -= 1.0
            breakdown["rel_strength"] = -1.0

    # 5. ORB breakdown bonus (±2.0 / -1.0)
    if orb_data is not None and orb_data.get("set"):
        orb_high = orb_data.get("high")
        orb_low  = orb_data.get("low")
        vol_ok   = (current_volume is not None and avg_volume is not None
                    and avg_volume > 0 and current_volume >= avg_volume * 1.5)
        if orb_low is not None and price < orb_low and vol_ok:
            score += W_ORB_MAX
            breakdown["orb"] = W_ORB_MAX
        elif orb_high is not None and price > orb_high:
            score -= 1.0
            breakdown["orb"] = -1.0

    # 6. VWAP upper-band rejection (±2.0 / +1.0 / -1.0)
    if vwap_data is not None:
        vwap       = vwap_data.get("vwap")
        upper_2std = vwap_data.get("upper_2std")
        upper_1std = vwap_data.get("upper_1std")
        last_red   = vwap_data.get("last_candle_red", False)
        if upper_2std is not None and price > upper_2std and last_red:
            score += W_VWAP_SHT_MAX
            breakdown["vwap"] = W_VWAP_SHT_MAX
        elif upper_1std is not None and price > upper_1std:
            score += 1.0
            breakdown["vwap"] = 1.0
        elif vwap is not None and price < vwap:
            score -= 1.0
            breakdown["vwap"] = -1.0

    return score, breakdown


def calculate_cover_score(fast_val, slow_val, rsi_val, rsi_buy_level,
                          price, bb_lower, bb_upper=None, **kwargs):
    """
    Calculate gradient score for closing a short (covering).
    No regime gate — always allow exits.
    Uses base indicators only (max 7.0).
    """
    score     = 0.0
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
    half_spread   = SPREAD_ESTIMATES.get(stock, DEFAULT_SPREAD)
    spread_cost   = half_spread * quantity
    slippage_cost = spread_cost * SLIPPAGE_MULT
    sec_fee       = 0.0
    if side == "sell":
        sec_fee = price * quantity * SEC_FEE_RATE
    return spread_cost + slippage_cost + sec_fee
