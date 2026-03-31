"""
Avellaneda-Stoikov Pairs Market Maker
=====================================
Standalone market making strategy on cointegrated equity pairs.
Runs as a separate process from bot.py on the same Alpaca account.

Pipeline:
  1. Systematic pair selection via Engle-Granger cointegration test
  2. Kalman Filter with EM calibration for time-varying hedge ratios
  3. Avellaneda-Stoikov optimal quoting with broker-aware fee floors
  4. Phantom fill prevention in backtesting

Usage:
  python3 pairs_market_maker.py              # Live trading (paper)
  python3 pairs_market_maker.py --backtest   # Run backtest on historical data
  python3 pairs_market_maker.py --scan       # Scan for cointegrated pairs only
"""

import os
import math
import json
import argparse
import time as time_module
import numpy as np
import pandas as pd
import schedule
from datetime import datetime, time, timedelta
from dotenv import load_dotenv
import pytz
import alpaca_trade_api as tradeapi
from statsmodels.tsa.stattools import coint
from itertools import combinations

import signals as sig

load_dotenv()

API_KEY = "PK22XEELBFYNU7QMJHJOGRJ6V6"
SECRET_KEY = "3arXWSeJW69nWfZHKW9nABMWwMkK1Ct964VakJdT7PXV"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

ET = pytz.timezone("America/New_York")


def fetch_bars(symbol, timeframe, limit=100):
    """Fetch bars with explicit start/end dates (Alpaca requires them)."""
    end = datetime.now()
    if timeframe == "1Day":
        start = end - timedelta(days=int(limit * 1.5) + 10)
    else:
        # Minute bars: limit bars / 390 bars per day, plus buffer
        start = end - timedelta(days=int(limit / 390 * 1.5) + 5)
    df = api.get_bars(
        symbol, timeframe,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        limit=limit,
    ).df
    return df


# ── Pair Candidates (within-sector to avoid spurious cointegration) ──
SECTOR_PAIRS = {
    "tech":     ["MSFT", "AAPL"],
    "semi":     ["AMD", "NVDA"],
    "consumer": ["WMT", "COST"],
    "ibank":    ["MS", "GS"],
    "payments": ["V", "MA"],
    "finance":  ["WFC", "AXP"],
}
CANDIDATE_TICKERS = list(set(t for tickers in SECTOR_PAIRS.values() for t in tickers))

# ── Trading Config ───────────────────────────────────────────────
PAIR = ("V", "MA")           # Default — overridden by cointegration scan
PAIR_QTY = 5                 # Shares per leg per spread unit
MAX_INVENTORY = 10           # Max |q| before forced unwind
GAMMA = 0.1                  # Risk aversion parameter
SIGMA_WINDOW = 20            # Rolling bars for spread volatility
FLATTEN_BY = time(15, 45)    # Flatten all inventory by 3:45 PM ET
COINTEGRATION_P_THRESHOLD = 0.05
MIN_HALF_LIFE = 30           # Bars (minutes) — too fast = noise
MAX_HALF_LIFE = 1500         # Bars — too slow = won't revert in session
DAILY_LOSS_LIMIT = -200      # Stop trading if daily PnL drops below this


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — PAIR SELECTION (Engle-Granger Cointegration)
# ══════════════════════════════════════════════════════════════════

def find_cointegrated_pairs(bars_dict, max_pairs=3):
    """
    Test all within-sector pairs for cointegration using Engle-Granger.

    Engle-Granger test (statsmodels.tsa.stattools.coint):
      - Runs OLS regression: price_A = beta * price_B + epsilon
      - Tests if residuals epsilon are stationary (ADF test)
      - Returns (test_stat, p_value, crit_values)
      - p < 0.05 means the pair IS cointegrated (residuals mean-revert)

    Also estimates half-life: how many bars for the spread to revert halfway.
    """
    results = []

    for sector, tickers in SECTOR_PAIRS.items():
        for a, b in combinations(tickers, 2):
            if a not in bars_dict or b not in bars_dict:
                continue

            prices_a = bars_dict[a]
            prices_b = bars_dict[b]

            # Align on common index
            aligned = pd.concat([prices_a, prices_b], axis=1, keys=[a, b]).dropna()
            if len(aligned) < 50:  # Need at least 50 daily bars
                continue

            pa, pb = aligned[a], aligned[b]

            # Engle-Granger cointegration test
            test_stat, p_value, crit_values = coint(pa, pb)

            if p_value >= COINTEGRATION_P_THRESHOLD:
                print(f"  {a}/{b} ({sector}): p={p_value:.4f} — NOT cointegrated")
                continue

            # Estimate static hedge ratio via OLS
            beta = np.polyfit(pb, pa, 1)[0]
            spread = pa - beta * pb

            # Estimate half-life of mean reversion
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            common = spread_diff.index.intersection(spread_lag.index)
            if len(common) < 10:
                continue
            phi = np.polyfit(spread_lag.loc[common], spread_diff.loc[common], 1)[0]

            if phi < 0:
                half_life = -np.log(2) / np.log(1 + phi)
            else:
                half_life = float('inf')

            results.append({
                "stock_a": a, "stock_b": b,
                "sector": sector,
                "p_value": p_value,
                "half_life_bars": half_life,
                "static_beta": beta,
                "test_stat": test_stat,
            })

            print(f"  {a}/{b} ({sector}): p={p_value:.4f} beta={beta:.4f} "
                  f"half_life={half_life:.0f} bars — COINTEGRATED")

    # Filter by half-life and sort by p-value
    # half_life_bars is in DAILY bars (since we use daily data for cointegration).
    # Convert to minutes: 1 day = 390 minute bars.
    # Accept half-life between ~0.08 days (30 min) and ~3.8 days (1500 min).
    min_hl_days = MIN_HALF_LIFE / 390.0   # ~0.08 days
    max_hl_days = MAX_HALF_LIFE / 390.0   # ~3.85 days
    results = [r for r in results if min_hl_days < r["half_life_bars"] < max_hl_days]
    results.sort(key=lambda x: x["p_value"])

    return results[:max_pairs]


def check_pair_still_valid(stock_a, stock_b, lookback_days=60):
    """Re-run cointegration test on rolling window. Returns (valid, p_value)."""
    try:
        bars_a = fetch_bars(stock_a, "1Day", limit=lookback_days)["close"]
        bars_b = fetch_bars(stock_b, "1Day", limit=lookback_days)["close"]
        aligned = pd.concat([bars_a, bars_b], axis=1).dropna()
        if len(aligned) < 30:
            return False, 1.0
        _, p_value, _ = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return p_value < 0.10, p_value  # Allow drift up to 0.10 before stopping
    except Exception as e:
        print(f"  Pair validation error: {e}")
        return True, 0.0  # Assume valid on error


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — KALMAN FILTER FOR DYNAMIC HEDGE RATIO
# ══════════════════════════════════════════════════════════════════

class KalmanHedgeRatio:
    """
    Tracks the hedge ratio beta as a hidden state that evolves over time.

    State space model:
      State:       beta(t) = beta(t-1) + w,  w ~ N(0, Q)    [random walk]
      Observation: price_A(t) = beta(t) * price_B(t) + v,  v ~ N(0, R)

    Each new minute bar:
      1. PREDICT: beta stays the same, uncertainty P grows by Q
      2. UPDATE:  observe actual prices, adjust beta toward reality

    Q = process noise (how much beta can drift per bar)
    R = measurement noise (how noisy the price relationship is)
    """

    def __init__(self, initial_beta=1.0, Q=1e-5, R=1e-3):
        self.beta = initial_beta
        self.P = 1.0           # Uncertainty in beta

        self.Q = Q             # Process noise
        self.R = R             # Measurement noise

        # History for EM calibration and diagnostics
        self.residuals = []
        self.betas = []

        # Internal predict state
        self._beta_prior = initial_beta
        self._P_prior = 1.0

    def predict(self):
        """Predict step: beta doesn't change, but uncertainty grows."""
        self._beta_prior = self.beta
        self._P_prior = self.P + self.Q

    def update(self, price_a, price_b):
        """
        Update step: adjust beta based on new price observation.

        Returns: (updated_beta, current_spread)
        """
        self.predict()

        H = price_b  # Observation matrix (scalar)

        # Innovation: how far off was our prediction?
        y = price_a - self._beta_prior * price_b

        # Innovation covariance
        S = H * self._P_prior * H + self.R

        # Kalman gain: how much to trust observation vs prediction
        K = self._P_prior * H / S

        # Update state
        self.beta = self._beta_prior + K * y

        # Update uncertainty (shrinks after observation)
        self.P = (1 - K * H) * self._P_prior

        # Track history
        self.residuals.append(y)
        self.betas.append(self.beta)

        # Current spread
        spread = price_a - self.beta * price_b
        return self.beta, spread

    def current_beta(self):
        return self.beta

    def em_calibrate(self, window=5000):
        """Auto-tune Q and R from recent history.
        Q = variance of beta changes, R = variance of residuals."""
        if len(self.betas) < window:
            return

        recent_betas = self.betas[-window:]
        recent_resid = self.residuals[-window:]

        beta_diffs = np.diff(recent_betas)
        self.Q = max(np.var(beta_diffs), 1e-8)
        self.R = max(np.var(recent_resid), 1e-6)

        print(f"  EM calibrated: Q={self.Q:.2e}, R={self.R:.2e}")

    def drift_detected(self, threshold=3.0):
        """Alert if beta has moved >3 std devs from its 500-bar mean."""
        if len(self.betas) < 500:
            return False
        recent = self.betas[-500:]
        mu = np.mean(recent)
        sigma = np.std(recent)
        if sigma < 1e-8:
            return False
        z = abs(self.beta - mu) / sigma
        return z > threshold


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — AVELLANEDA-STOIKOV OPTIMAL QUOTING
# ══════════════════════════════════════════════════════════════════

def get_optimal_quotes(s, q, sigma, kappa, gamma, T):
    """
    Avellaneda-Stoikov optimal market making quotes.

    s     = Current spread mid-price (price_A - beta * price_B)
    q     = Current inventory in spread units (signed: + = long, - = short)
    sigma = Spread volatility (rolling std dev of spread changes)
    kappa = Order arrival intensity (fills per bar, estimated)
    gamma = Risk aversion (0.01 = aggressive, 1.0 = conservative)
    T     = Time remaining in session (fraction: 1.0 at open, 0.0 at close)

    Returns: (bid_price, ask_price, reservation_price, full_spread)
    """
    sigma_sq = sigma ** 2

    # Reservation price: shifted away from inventory
    # Long inventory (q>0) → r < s (want to sell, quote below mid)
    # Short inventory (q<0) → r > s (want to buy, quote above mid)
    r = s - q * gamma * sigma_sq * T

    # Optimal half-spread: compensation for adverse selection + inventory risk
    if kappa < 1e-6:
        kappa = 0.01
    delta = (1.0 / gamma) * math.log(1.0 + gamma / kappa) + 0.5 * gamma * sigma_sq * T

    bid = r - delta
    ask = r + delta

    return bid, ask, r, delta * 2


def apply_fee_floor(bid, ask, cost_per_unit):
    """Never quote narrower than 1.5x transaction costs (50% profit margin)."""
    mid = (bid + ask) / 2.0
    min_half_spread = cost_per_unit * 0.75

    if (ask - bid) < 2 * min_half_spread:
        bid = mid - min_half_spread
        ask = mid + min_half_spread

    return bid, ask


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — ORDER MANAGEMENT & MAIN LOOP
# ══════════════════════════════════════════════════════════════════

# ── State ────────────────────────────────────────────────────────
inventory_q = 0
kalman = KalmanHedgeRatio(initial_beta=1.0)
active_orders = {}       # {"bid_a": id, "bid_b": id, "ask_a": id, "ask_b": id}
spread_history = []
daily_pnl = 0.0
fills_today = 0
session_trades = []


def get_session_T():
    """Fraction of trading day remaining. 1.0 at 9:30, 0.0 at 4:00."""
    now = datetime.now(ET)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    total = (market_close - market_open).total_seconds()
    remaining = (market_close - now).total_seconds()
    return max(0.0, min(1.0, remaining / total))


def cancel_all_pair_orders():
    """Cancel outstanding limit orders before requoting."""
    for tag, oid in list(active_orders.items()):
        try:
            api.cancel_order(oid)
        except Exception:
            pass
    active_orders.clear()


def check_fills():
    """Check order statuses via API — never assume fills (phantom fill prevention)."""
    global inventory_q, fills_today, daily_pnl

    for tag, oid in list(active_orders.items()):
        try:
            order = api.get_order(oid)
        except Exception:
            continue

        if order.status == "filled":
            filled_price = float(order.filled_avg_price)

            if tag == "bid_a":
                # Bid filled on stock A = bought spread
                inventory_q += 1
                fills_today += 1
                session_trades.append({
                    "time": datetime.now(ET).isoformat(),
                    "side": "buy_spread", "price": filled_price, "q": inventory_q
                })
                print(f"  BID FILLED: bought spread @ {filled_price:.2f}, q={inventory_q}")
            elif tag == "ask_a":
                # Ask filled on stock A = sold spread
                inventory_q -= 1
                fills_today += 1
                session_trades.append({
                    "time": datetime.now(ET).isoformat(),
                    "side": "sell_spread", "price": filled_price, "q": inventory_q
                })
                print(f"  ASK FILLED: sold spread @ {filled_price:.2f}, q={inventory_q}")

            del active_orders[tag]

        elif order.status == "partially_filled":
            print(f"  PARTIAL FILL on {tag}: {order.filled_qty}/{order.qty}")


def submit_pair_orders(bid_price, ask_price, beta):
    """Submit limit orders for both legs of the spread."""
    stock_a, stock_b = PAIR
    qty_b = max(1, round(PAIR_QTY * abs(beta)))

    try:
        quote_b = api.get_latest_quote(stock_b)
        mid_b = (float(quote_b.bp) + float(quote_b.ap)) / 2
    except Exception as e:
        print(f"  Quote fetch error: {e}")
        return

    # Derive stock A limit prices from spread quotes
    # spread = price_A - beta * price_B
    # To buy spread at bid_price: buy A at (bid_price + beta * mid_b)
    # To sell spread at ask_price: sell A at (ask_price + beta * mid_b)
    limit_a_buy = round(bid_price + beta * mid_b, 2)
    limit_a_sell = round(ask_price + beta * mid_b, 2)

    try:
        # BID side: buy A, sell B
        bid_a = api.submit_order(
            symbol=stock_a, qty=PAIR_QTY, side="buy",
            type="limit", time_in_force="day", limit_price=limit_a_buy)
        active_orders["bid_a"] = bid_a.id

        api.submit_order(
            symbol=stock_b, qty=qty_b, side="sell",
            type="limit", time_in_force="day", limit_price=round(mid_b, 2))

        # ASK side: sell A, buy B
        ask_a = api.submit_order(
            symbol=stock_a, qty=PAIR_QTY, side="sell",
            type="limit", time_in_force="day", limit_price=limit_a_sell)
        active_orders["ask_a"] = ask_a.id

        api.submit_order(
            symbol=stock_b, qty=qty_b, side="buy",
            type="limit", time_in_force="day", limit_price=round(mid_b, 2))

    except Exception as e:
        print(f"  Order submission error: {e}")
        cancel_all_pair_orders()


def flatten_inventory(beta):
    """Force-flatten inventory with market orders at end of day."""
    global inventory_q
    stock_a, stock_b = PAIR
    qty_b = max(1, round(PAIR_QTY * abs(beta)))

    if inventory_q > 0:
        for _ in range(abs(inventory_q)):
            api.submit_order(stock_a, PAIR_QTY, "sell", "market", "day")
            api.submit_order(stock_b, qty_b, "buy", "market", "day")
    elif inventory_q < 0:
        for _ in range(abs(inventory_q)):
            api.submit_order(stock_a, PAIR_QTY, "buy", "market", "day")
            api.submit_order(stock_b, qty_b, "sell", "market", "day")

    print(f"  Flattened {abs(inventory_q)} spread units")
    inventory_q = 0


def run_market_maker():
    """Main loop — runs every 1 minute during market hours."""
    global inventory_q

    now = datetime.now(ET)
    if now.time() < time(9, 31) or now.time() > time(15, 55):
        return

    stock_a, stock_b = PAIR

    # Step 1: Check fills from previous quotes
    check_fills()

    # Step 2: Fetch latest bars
    try:
        bars_a = fetch_bars(stock_a, "1Min", limit=SIGMA_WINDOW + 5)
        bars_b = fetch_bars(stock_b, "1Min", limit=SIGMA_WINDOW + 5)
        price_a = float(bars_a["close"].iloc[-1])
        price_b = float(bars_b["close"].iloc[-1])
    except Exception as e:
        print(f"  Bar fetch error: {e}")
        return

    # Step 3: Kalman update
    beta, spread = kalman.update(price_a, price_b)
    spread_history.append(spread)

    # Drift check
    if kalman.drift_detected(threshold=3.0):
        print(f"  PAIR DRIFT DETECTED: beta={beta:.4f}, pausing quotes")
        cancel_all_pair_orders()
        return

    # Step 4: Spread volatility
    if len(spread_history) < SIGMA_WINDOW:
        print(f"  Warming up ({len(spread_history)}/{SIGMA_WINDOW} bars)...")
        return
    sigma = np.std(spread_history[-SIGMA_WINDOW:])
    if sigma < 1e-6:
        sigma = 0.01

    # Step 5: Avellaneda-Stoikov quotes
    T = get_session_T()
    kappa = max(0.01, fills_today / max(1, 390 - int((1 - T) * 390)))
    if fills_today == 0:
        kappa = 0.05  # Default until we observe fills

    bid, ask, r, full_spread = get_optimal_quotes(spread, inventory_q, sigma, kappa, GAMMA, T)

    # Step 6: Fee floor
    cost_a = sig.SPREAD_ESTIMATES.get(stock_a, 0.03)
    cost_b = sig.SPREAD_ESTIMATES.get(stock_b, 0.03)
    cost_per_unit = 2 * (cost_a + cost_b)
    bid, ask = apply_fee_floor(bid, ask, cost_per_unit)

    # Step 7: Cancel old orders and requote
    cancel_all_pair_orders()

    # Step 8: Safety checks
    if now.time() >= FLATTEN_BY:
        if inventory_q != 0:
            print(f"  EOD FLATTEN: q={inventory_q}")
            flatten_inventory(beta)
        return

    if abs(inventory_q) >= MAX_INVENTORY:
        print(f"  MAX INVENTORY: q={inventory_q}, skipping quotes")
        return

    if daily_pnl < DAILY_LOSS_LIMIT:
        print(f"  DAILY LOSS LIMIT: PnL={daily_pnl:.2f}, stopping")
        return

    submit_pair_orders(bid, ask, beta)

    # Diagnostic
    print(f"  MM {stock_a}/{stock_b} beta={beta:.3f} spread={spread:.3f} "
          f"sigma={sigma:.4f} q={inventory_q} T={T:.2f}")
    print(f"     r={r:.3f} delta={full_spread/2:.3f} bid={bid:.3f} ask={ask:.3f} "
          f"fills={fills_today}")


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — PHANTOM FILL PREVENTION (BACKTESTING)
# ══════════════════════════════════════════════════════════════════

def simulate_mm_backtest(spread_series, gamma=0.1, kappa=1.0, sigma_window=120,
                         cost_per_fill=0.12, max_inventory=10,
                         z_entry=1.5, z_exit=0.0, z_stop=3.0,
                         verbose=False):
    """
    Backtest A-S market making on historical spread data.

    Entry logic: z-score based mean-reversion.
      - Enter long spread when z-score < -z_entry  (spread too low, expect rise)
      - Enter short spread when z-score > +z_entry  (spread too high, expect fall)
      - Exit when z-score crosses back through z_exit (mean reversion complete)
      - Stop-loss when |z-score| exceeds z_stop (spread blew out further)

    A-S quotes operate on the z-score series (mean≈0, std≈1), not the raw dollar
    spread. Raw spread has a non-zero mean and large sigma (e.g. $17-26), which
    makes the A-S half-spread blow out to ±10 points — price never reaches the
    quotes. Z-score normalization keeps sigma≈1 so A-S quotes are tight and
    realistic. PnL is tracked in z-score units then converted back to dollars
    at the end using the spread's rolling std.

    Phantom fill prevention:
      - Require price to cross THROUGH the limit by at least 1 tick
      - Apply 50% random fill probability (queue priority simulation)
    """
    TICK_SIZE = 0.01
    FILL_PROB = 0.5

    # ── FIX 1: Pre-loop diagnostics ─────────────────────────────────
    # Compute z-scores using a rolling window
    roll = spread_series.rolling(window=sigma_window)
    roll_mean = roll.mean()
    roll_std = roll.std()
    z_series = (spread_series - roll_mean) / roll_std.clip(lower=1e-6)
    z_valid = z_series.dropna()

    print(f"\n  --- Spread Diagnostics ---")
    print(f"  Spread mean:  {spread_series.mean():.4f}   std: {spread_series.std():.4f}")
    print(f"  Z-score range: [{z_valid.min():.2f}, {z_valid.max():.2f}]")
    print(f"  Z-score percentiles — 5th: {z_valid.quantile(0.05):.2f}  "
          f"25th: {z_valid.quantile(0.25):.2f}  "
          f"75th: {z_valid.quantile(0.75):.2f}  "
          f"95th: {z_valid.quantile(0.95):.2f}")
    print(f"  Entry threshold: |z| > {z_entry:.2f}")
    bars_above = (z_valid.abs() > z_entry).sum()
    print(f"  Bars with |z| > {z_entry:.2f}: {bars_above} of {len(z_valid)} "
          f"({100*bars_above/len(z_valid):.1f}%)")
    print(f"  --------------------------")
    # ────────────────────────────────────────────────────────────────

    rng = np.random.RandomState(42)
    pnl = 0.0
    inventory = 0
    trades = []
    pnl_series = []
    max_inv = 0
    position_entry_price = None  # Track entry price for PnL on exit

    T_total = 390  # Bars in a trading day

    for i in range(sigma_window + 1, len(spread_series)):
        s = spread_series.iloc[i]
        s_prev = spread_series.iloc[i - 1]

        # Rolling spread stats for z-score
        window = spread_series.iloc[i - sigma_window:i]
        mu = window.mean()
        sigma_dollar = window.std()
        if sigma_dollar < 1e-6:
            sigma_dollar = 0.01

        # Z-score of the spread (mean=0, std≈1)
        z      = (s      - mu) / sigma_dollar
        z_prev = (s_prev - mu) / sigma_dollar

        # Time remaining
        bar_in_day = i % T_total
        T = max(0.01, (T_total - bar_in_day) / T_total)

        # A-S optimal half-spread in z-score units, then convert to dollars.
        # Using sigma_z=1 keeps the ln(1+gamma/kappa) term from blowing up.
        sigma_z = 1.0
        _, _, _, delta_z_full = get_optimal_quotes(z_prev, inventory, sigma_z, kappa, gamma, T)
        half_spread_dollar = (delta_z_full / 2.0) * sigma_dollar

        # Reservation price in dollar terms (A-S inventory skew)
        r_dollar = s_prev - inventory * gamma * (sigma_dollar ** 2) * T

        # Bid/ask quotes in dollar spread space
        bid = r_dollar - half_spread_dollar
        ask = r_dollar + half_spread_dollar
        bid, ask = apply_fee_floor(bid, ask, cost_per_fill)

        # Simulate bar high/low in dollar spread space
        change = abs(s - s_prev)
        bar_high = max(s, s_prev) + change * 0.3
        bar_low  = min(s, s_prev) - change * 0.3

        # ── FIX 2: Z-score entry gates ───────────────────────────────
        # Only quote the long side when spread is sufficiently low (z < -z_entry)
        # Only quote the short side when spread is sufficiently high (z > +z_entry)
        # Exit (flip allowed) when z crosses back through z_exit
        # Stop-loss when |z| > z_stop
        want_long  = (z < -z_entry) and (inventory < max_inventory)
        want_short = (z >  z_entry) and (inventory > -max_inventory)

        # If we're already in a position, check exit/stop conditions
        if inventory > 0:
            # Long spread position: exit when z rises back to z_exit, stop if z < -z_stop
            want_long = False
            want_short = (z >= z_exit) or (z < -z_stop)
        elif inventory < 0:
            # Short spread position: exit when z falls back to z_exit, stop if z > z_stop
            want_short = False
            want_long = (z <= z_exit) or (z > z_stop)
        # ─────────────────────────────────────────────────────────────

        # ── FIX 4: Verbose signal logging ───────────────────────────
        if verbose:
            ts = spread_series.index[i]
            ts_str = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(i)
            stock_a, stock_b = PAIR
            if z < -z_entry or z > z_entry:
                signal = "LONG SIGNAL" if z < -z_entry else "SHORT SIGNAL"
                print(f"  [{ts_str}] {stock_a}/{stock_b} z={z:.2f} → {signal}")
            else:
                print(f"  [{ts_str}] {stock_a}/{stock_b} spread z-score: {z:.2f} "
                      f"— below threshold {z_entry:.1f}")
        # ─────────────────────────────────────────────────────────────

        # Phantom fill check (only on the gated sides)
        # bid/ask are now in dollar spread space — compare directly to bar_low/bar_high
        bid_filled = want_long  and bar_low  < (bid - TICK_SIZE) and rng.random() < FILL_PROB
        ask_filled = want_short and bar_high > (ask + TICK_SIZE) and rng.random() < FILL_PROB

        # If both would fill, only allow one
        if bid_filled and ask_filled:
            if rng.random() < 0.5:
                ask_filled = False
            else:
                bid_filled = False

        if bid_filled:
            inventory += 1
            pnl -= bid + cost_per_fill
            trades.append({"bar": i, "side": "buy", "price": bid, "inv": inventory, "z": z})

        if ask_filled:
            inventory -= 1
            pnl += ask - cost_per_fill
            trades.append({"bar": i, "side": "sell", "price": ask, "inv": inventory, "z": z})

        max_inv = max(max_inv, abs(inventory))

        # End-of-day flatten (every 390 bars)
        if bar_in_day == T_total - 1 and inventory != 0:
            pnl += inventory * s
            pnl -= abs(inventory) * cost_per_fill
            inventory = 0

        pnl_series.append(pnl)

    # Final flatten at last dollar spread price
    if inventory != 0:
        pnl += inventory * spread_series.iloc[-1]
        pnl -= abs(inventory) * cost_per_fill

    # Calculate metrics
    pnl_arr = np.array(pnl_series)
    returns = np.diff(pnl_arr) if len(pnl_arr) > 1 else np.array([0])
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252 * 390)) if np.std(returns) > 0 else 0

    peak = np.maximum.accumulate(pnl_arr)
    drawdown = (peak - pnl_arr) / np.maximum(peak, 1)
    max_dd = np.max(drawdown) * 100 if len(drawdown) > 0 else 0

    return {
        "pnl": pnl,
        "trades": len(trades),
        "max_inventory": max_inv,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "fills_per_day": len(trades) / max(1, len(spread_series) / 390),
        "z_entry": z_entry,
    }


def _run_optimize(spread_series, cost_per_fill, sigma_window=120):
    """
    FIX 3: Parameter sweep over z_entry thresholds.
    Tests z_entry from 0.5 to 3.0 in 0.25 steps and prints a results table.
    """
    import math

    thresholds = [round(0.5 + 0.25 * k, 2) for k in range(11)]  # 0.5 → 3.0

    print(f"\n{'='*65}")
    print(f"  Z-ENTRY THRESHOLD SWEEP")
    print(f"{'='*65}")
    print(f"  {'Z-Entry':>8}  {'Trades':>7}  {'PnL':>9}  {'Sharpe':>8}  {'MaxDD':>7}  {'Fills/Day':>10}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*10}")

    best_sharpe = -999
    best_z = None

    for z in thresholds:
        r = simulate_mm_backtest(
            spread_series,
            cost_per_fill=cost_per_fill,
            z_entry=z,
            sigma_window=sigma_window,
            verbose=False,
        )
        marker = " ◄" if r["sharpe"] > best_sharpe else ""
        if r["sharpe"] > best_sharpe:
            best_sharpe = r["sharpe"]
            best_z = z
        print(f"  {z:>8.2f}  {r['trades']:>7d}  "
              f"${r['pnl']:>8.2f}  {r['sharpe']:>8.2f}  "
              f"{r['max_drawdown_pct']:>6.1f}%  {r['fills_per_day']:>10.1f}{marker}")

    print(f"{'='*65}")
    print(f"  Best z_entry: {best_z:.2f} (Sharpe {best_sharpe:.2f})")
    print(f"{'='*65}\n")


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — ENTRY POINTS
# ══════════════════════════════════════════════════════════════════

def scan_pairs():
    """Scan for cointegrated pairs using daily data."""
    print("\n=== COINTEGRATION SCAN ===\n")
    bars_dict = {}
    for t in CANDIDATE_TICKERS:
        try:
            daily = fetch_bars(t, "1Day", limit=90)
            if hasattr(daily.index, 'get_level_values'):
                try:
                    daily = daily.xs(t, level="symbol")
                except (KeyError, TypeError):
                    pass
            bars_dict[t] = daily["close"]
            print(f"  Fetched {t}: {len(daily)} daily bars")
        except Exception as e:
            print(f"  Failed to fetch {t}: {e}")

    pairs = find_cointegrated_pairs(bars_dict)

    if not pairs:
        print("\nNo cointegrated pairs found.")
        return []

    print(f"\n{'='*60}")
    print(f"COINTEGRATED PAIRS (p < {COINTEGRATION_P_THRESHOLD}):")
    print(f"{'='*60}")
    for p in pairs:
        print(f"  {p['stock_a']}/{p['stock_b']} ({p['sector']}): "
              f"p={p['p_value']:.4f} beta={p['static_beta']:.4f} "
              f"half_life={p['half_life_bars']:.0f} bars")

    return pairs


def run_backtest_mode(_z_entry=1.5, _optimize=False, _verbose=False, _sigma_window=120):
    """Backtest the market maker on the best cointegrated pair."""
    pairs = scan_pairs()
    if not pairs:
        return

    best = pairs[0]
    stock_a, stock_b = best["stock_a"], best["stock_b"]
    print(f"\nBacktesting on {stock_a}/{stock_b}...")

    # Fetch minute bars
    bars_a = fetch_bars(stock_a, "1Min", limit=10000)
    bars_b = fetch_bars(stock_b, "1Min", limit=10000)
    if hasattr(bars_a.index, 'get_level_values'):
        try:
            bars_a = bars_a.xs(stock_a, level="symbol")
            bars_b = bars_b.xs(stock_b, level="symbol")
        except (KeyError, TypeError):
            pass

    aligned = pd.concat([bars_a["close"], bars_b["close"]], axis=1, keys=[stock_a, stock_b]).dropna()
    print(f"  Aligned: {len(aligned)} minute bars")

    # Use static OLS beta for the backtest spread.
    # The Kalman filter fits beta bar-by-bar, which makes the spread collapse to
    # near-zero on the same data it trained on (overfitting). Static beta gives
    # a realistic spread with actual variance for backtesting.
    # Kalman is still used in live trading where it updates on fresh, unseen bars.
    static_beta = best["static_beta"]
    spread_series = aligned.iloc[:, 0] - static_beta * aligned.iloc[:, 1]
    print(f"  Static beta: {static_beta:.4f}  "
          f"spread std: {spread_series.std():.4f}  "
          f"spread range: [{spread_series.min():.3f}, {spread_series.max():.3f}]")

    # Still warm up the Kalman so it's ready for live trading (EM calibration)
    kf = KalmanHedgeRatio(initial_beta=static_beta)
    for i in range(len(aligned)):
        kf.update(aligned.iloc[i, 0], aligned.iloc[i, 1])
    kf.em_calibrate()

    # Transaction costs for both legs
    cost_a = sig.SPREAD_ESTIMATES.get(stock_a, 0.03)
    cost_b = sig.SPREAD_ESTIMATES.get(stock_b, 0.03)
    cost_per_fill = 2 * (cost_a + cost_b)

    # FIX 3: Optimize mode — sweep z_entry thresholds
    if _optimize:
        _run_optimize(spread_series, cost_per_fill, sigma_window=_sigma_window)
        return None

    # Single backtest run
    result = simulate_mm_backtest(
        spread_series,
        gamma=GAMMA,
        cost_per_fill=cost_per_fill,
        z_entry=_z_entry,
        verbose=_verbose,
        sigma_window=_sigma_window,
    )

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS — {stock_a}/{stock_b}")
    print(f"{'='*60}")
    print(f"  Z-Entry Threshold: {_z_entry:.2f}")
    print(f"  Net PnL:        ${result['pnl']:.2f}")
    print(f"  Total Trades:   {result['trades']}")
    print(f"  Fills/Day:      {result['fills_per_day']:.1f}")
    print(f"  Sharpe Ratio:   {result['sharpe']:.2f}")
    print(f"  Max Drawdown:   {result['max_drawdown_pct']:.2f}%")
    print(f"  Max Inventory:  {result['max_inventory']}")

    return result


def run_live():
    """Live trading mode."""
    global PAIR

    print("\n=== AVELLANEDA-STOIKOV PAIRS MARKET MAKER ===\n")

    # Step 1: Find best pair
    pairs = scan_pairs()
    if not pairs:
        print("No cointegrated pairs found. Exiting.")
        return

    best = pairs[0]
    PAIR = (best["stock_a"], best["stock_b"])
    print(f"\nSelected pair: {PAIR[0]}/{PAIR[1]} "
          f"(p={best['p_value']:.4f}, half_life={best['half_life_bars']:.0f} bars)")

    # Step 2: Initialize Kalman
    kalman.__init__(initial_beta=best["static_beta"])

    # Step 3: Warm up on historical minute bars
    print(f"Warming up Kalman filter...")
    try:
        bars_a = fetch_bars(PAIR[0], "1Min", limit=5000)
        bars_b = fetch_bars(PAIR[1], "1Min", limit=5000)
        if hasattr(bars_a.index, 'get_level_values'):
            try:
                bars_a = bars_a.xs(PAIR[0], level="symbol")
                bars_b = bars_b.xs(PAIR[1], level="symbol")
            except (KeyError, TypeError):
                pass
        aligned = pd.concat([bars_a["close"], bars_b["close"]], axis=1).dropna()
        for i in range(len(aligned)):
            kalman.update(float(aligned.iloc[i, 0]), float(aligned.iloc[i, 1]))
        kalman.em_calibrate()
        print(f"  Warmed up on {len(aligned)} bars, beta={kalman.current_beta():.4f}")
    except Exception as e:
        print(f"  Warmup error: {e}")
        print(f"  Starting with static beta={best['static_beta']:.4f}")

    # Step 4: Schedule
    schedule.every(1).minutes.do(run_market_maker)

    print(f"\nMarket maker running on {PAIR[0]}/{PAIR[1]}. Ctrl+C to stop.\n")
    try:
        while True:
            schedule.run_pending()
            time_module.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cancel_all_pair_orders()
        if inventory_q != 0:
            print(f"Flattening remaining inventory (q={inventory_q})...")
            flatten_inventory(kalman.current_beta())
        # Save session trades
        if session_trades:
            with open("mm_trades_log.json", "w") as f:
                json.dump(session_trades, f, indent=2)
            print(f"Saved {len(session_trades)} trades to mm_trades_log.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avellaneda-Stoikov Pairs Market Maker")
    parser.add_argument("--backtest",  action="store_true", help="Run backtest on historical data")
    parser.add_argument("--scan",      action="store_true", help="Scan for cointegrated pairs only")
    parser.add_argument("--optimize",  action="store_true", help="Sweep z_entry thresholds 0.5→3.0")
    parser.add_argument("--verbose",   action="store_true", help="Print every bar's z-score signal")
    parser.add_argument("--z-entry",     type=float, default=1.5, help="Z-score entry threshold (default 1.5)")
    parser.add_argument("--sigma-window", type=int,   default=120, help="Rolling window for spread z-score (default 120 bars = 2 hours, optimal for intraday mean reversion)")
    parser.add_argument("--gamma",     type=float, default=0.1,  help="Risk aversion parameter")
    parser.add_argument("--pair",      type=str,   default=None,  help="Force pair (e.g., 'V,MA')")
    args = parser.parse_args()

    GAMMA = args.gamma

    if args.pair:
        parts = args.pair.split(",")
        if len(parts) == 2:
            PAIR = (parts[0].strip(), parts[1].strip())
            print(f"Forced pair: {PAIR[0]}/{PAIR[1]}")

    if args.scan:
        scan_pairs()
    elif args.backtest or args.optimize:
        run_backtest_mode(_z_entry=args.z_entry, _optimize=args.optimize, _verbose=args.verbose,
                         _sigma_window=args.sigma_window)
    else:
        run_live()
