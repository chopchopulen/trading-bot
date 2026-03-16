from flask import Flask, jsonify
import alpaca_trade_api as tradeapi
import pandas as pd
import csv
import os
import json
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────
API_KEY = "PK22XEELBFYNU7QMJHJOGRJ6V6"
SECRET_KEY = "3arXWSeJW69nWfZHKW9nABMWwMkK1Ct964VakJdT7PXV"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)
app = Flask(__name__)

STARTING_CASH = 100000
LOG_FILE = "trades_log.csv"

TIER1_STOCKS = ["MSFT", "META", "AMD", "SPY"]
TIER2_STOCKS = ["AAPL", "TSLA", "NFLX", "WMT", "COST"]

STOCK_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia",
    "GOOGL": "Google", "AMZN": "Amazon", "TSLA": "Tesla",
    "META": "Meta", "AMD": "AMD", "NFLX": "Netflix",
    "CRM": "Salesforce", "JPM": "JPMorgan", "BAC": "Bank of America",
    "GS": "Goldman Sachs", "V": "Visa", "SPY": "S&P 500",
    "QQQ": "Nasdaq", "XOM": "Exxon", "CVX": "Chevron",
    "JNJ": "J&J", "PFE": "Pfizer", "UNH": "UnitedHealth",
    "WMT": "Walmart", "COST": "Costco", "NKE": "Nike",
    "INTC": "Intel", "QCOM": "Qualcomm", "UBER": "Uber",
    "PLTR": "Palantir", "COIN": "Coinbase", "SHOP": "Shopify",
    "SQ": "Block", "ROKU": "Roku", "ABNB": "Airbnb",
    "PYPL": "PayPal", "SPOT": "Spotify", "ZM": "Zoom",
    "HOOD": "Robinhood",
}


# ── Helper functions ──────────────────────────────────────────────

def get_watchlist():
    """Read today's watchlist from watchlist.txt."""
    stocks = []
    if os.path.exists("watchlist.txt"):
        with open("watchlist.txt") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("Last updated"):
                    stocks.append(line)
    return stocks


def get_spy_data():
    """Get SPY price and 50 EMA for regime detection."""
    try:
        bars = api.get_bars(
            "SPY",
            tradeapi.rest.TimeFrame.Day,
            start=(datetime.now() - timedelta(days=80)).strftime("%Y-%m-%d"),
            end=datetime.now().strftime("%Y-%m-%d"),
            feed="iex"
        ).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs("SPY", level="symbol")
        close = bars["close"]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        current = close.iloc[-1]
        start_price = close.iloc[0]
        spy_return = ((current - start_price) / start_price) * 100
        return current, ema50, spy_return
    except Exception:
        return None, None, 0


def get_fisher_pvalue():
    """Load Fisher combined p-value from permutation test results."""
    path = "permutation_test_results/combined.json"
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("fisher_combined_p")
        except Exception:
            return None
    return None


def get_walk_forward_params():
    """Load walk-forward results for signal strength display."""
    results = {}
    wf_dir = "walk_forward_results"
    if os.path.exists(wf_dir):
        for fname in os.listdir(wf_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(wf_dir, fname)) as f:
                        data = json.load(f)
                    stock = data.get("stock", fname.replace(".json", ""))
                    results[stock] = data
                except Exception:
                    pass
    return results


def load_trades():
    """Load all trades from CSV."""
    trades = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            reader = csv.DictReader(f)
            trades = list(reader)
    return trades


def compute_trade_stats(trades):
    """Compute win rate and total wins/losses from trade pairs."""
    wins = 0
    total_closed = 0
    open_buys = {}
    realized_pnl = 0.0
    for t in trades:
        action = t.get("action", "").upper()
        stock = t.get("stock", "")
        price = float(t.get("price", 0))
        if action == "BUY":
            open_buys[stock] = price
        elif action in ("SELL", "STOP LOSS", "TAKE PROFIT") and stock in open_buys:
            entry = open_buys.pop(stock)
            pnl = price - entry
            realized_pnl += pnl
            total_closed += 1
            if pnl > 0:
                wins += 1
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
    return win_rate, wins, total_closed, realized_pnl


# ── API endpoint for AJAX refresh ─────────────────────────────────

@app.route("/api/data")
def api_data():
    """Return all dashboard data as JSON for client-side refresh."""
    try:
        account = api.get_account()
        portfolio_value = float(account.portfolio_value)
        cash = float(account.cash)
    except Exception:
        portfolio_value = STARTING_CASH
        cash = STARTING_CASH

    total_return = ((portfolio_value - STARTING_CASH) / STARTING_CASH) * 100

    try:
        positions = api.list_positions()
    except Exception:
        positions = []

    unrealized_pnl = sum(float(p.unrealized_pl) for p in positions)

    try:
        clock = api.get_clock()
        market_open = clock.is_open
    except Exception:
        market_open = False

    spy_price, spy_ema, spy_return = get_spy_data()
    watchlist = get_watchlist()
    fisher_p = get_fisher_pvalue()
    trades = load_trades()
    win_rate, wins, total_closed, realized_pnl = compute_trade_stats(trades)

    # Determine regime
    if spy_price and spy_ema:
        if spy_price > spy_ema * 1.005:
            regime = "UPTREND"
        elif spy_price < spy_ema * 0.995:
            regime = "DOWNTREND"
        else:
            regime = "NEUTRAL"
    else:
        regime = "NEUTRAL"

    # Today's P&L (unrealized from positions)
    today_pnl = unrealized_pnl

    # Positions data
    pos_data = []
    for p in positions:
        entry = float(p.avg_entry_price)
        current = float(p.current_price)
        pnl = float(p.unrealized_pl)
        pnl_pct = float(p.unrealized_plpc) * 100
        qty = int(p.qty) if float(p.qty) == int(float(p.qty)) else float(p.qty)
        # Estimate stop/target from ATR (1.5x stop, 3x target)
        atr_est = abs(current - entry) * 0.5 if abs(current - entry) > 0 else current * 0.01
        stop = entry - 1.5 * atr_est
        target = entry + 3.0 * atr_est
        # Progress: how far from entry toward target (positive) or stop (negative)
        if current >= entry:
            progress = min((current - entry) / (target - entry) * 100, 100) if target != entry else 0
        else:
            progress = max(-((entry - current) / (entry - stop) * 100), -100) if entry != stop else 0

        pos_data.append({
            "symbol": p.symbol,
            "qty": qty,
            "side": p.side,
            "entry": entry,
            "current": current,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "stop": round(stop, 2),
            "target": round(target, 2),
            "progress": round(progress, 1),
        })

    # Recent trades (last 20)
    recent_trades = []
    for t in reversed(trades[-20:]):
        recent_trades.append({
            "timestamp": t.get("timestamp", "")[:16],
            "stock": t.get("stock", ""),
            "action": t.get("action", ""),
            "price": float(t.get("price", 0)),
        })

    # Equity curve from trade dates
    equity_dates = []
    equity_values = []
    seen_dates = set()
    for t in trades:
        d = t.get("timestamp", "")[:10]
        if d and d not in seen_dates:
            seen_dates.add(d)
            equity_dates.append(d)
    # Linear interpolation from STARTING_CASH to current
    if equity_dates:
        n = len(equity_dates)
        for i in range(n):
            val = STARTING_CASH + (portfolio_value - STARTING_CASH) * ((i + 1) / n)
            equity_values.append(round(val, 2))
    equity_dates.append(datetime.now().strftime("%Y-%m-%d"))
    equity_values.append(portfolio_value)

    # Walk-forward data for signal panel
    wf = get_walk_forward_params()

    # Build signal info for watchlist stocks
    signal_data = []
    for stock in watchlist:
        tier = "T1" if stock in TIER1_STOCKS else "T2" if stock in TIER2_STOCKS else "T3"
        wf_info = wf.get(stock, {})
        sharpe = wf_info.get("test_sharpe", 0)
        wr = wf_info.get("test_win_rate", 0)
        status = wf_info.get("status", "N/A")
        signal_data.append({
            "stock": stock,
            "name": STOCK_NAMES.get(stock, stock),
            "tier": tier,
            "sharpe": round(sharpe, 2) if sharpe else 0,
            "win_rate": round(wr, 1),
            "status": status,
        })

    # Sharpe ratio estimate
    if total_closed > 1 and realized_pnl != 0:
        avg_return = total_return / max(total_closed, 1)
        sharpe = round(total_return / max(abs(avg_return) * (total_closed ** 0.5), 0.01), 2)
    else:
        sharpe = 0.0

    return jsonify({
        "portfolio_value": portfolio_value,
        "cash": cash,
        "total_return": round(total_return, 4),
        "today_pnl": round(today_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "realized_pnl": round(realized_pnl, 2),
        "market_open": market_open,
        "spy_price": round(spy_price, 2) if spy_price else None,
        "spy_ema": round(spy_ema, 2) if spy_ema else None,
        "spy_return": round(spy_return, 2),
        "regime": regime,
        "watchlist": watchlist,
        "fisher_p": round(fisher_p, 4) if fisher_p else None,
        "win_rate": round(win_rate, 1),
        "wins": wins,
        "total_closed": total_closed,
        "sharpe": sharpe,
        "positions": pos_data,
        "recent_trades": recent_trades,
        "equity_dates": equity_dates,
        "equity_values": equity_values,
        "signals": signal_data,
        "updated": datetime.now().strftime("%b %d %Y, %I:%M:%S %p"),
    })


# ── Main dashboard route ──────────────────────────────────────────

@app.route("/")
def dashboard():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Algo Trader — Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
/* ── Reset & Base ─────────────────────────────────────────── */
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  --bg: #0a0a0f;
  --card-bg: #12121a;
  --card-border: #1e1e2e;
  --text-primary: #e2e8f0;
  --text-secondary: #64748b;
  --positive: #10b981;
  --negative: #ef4444;
  --accent: #6366f1;
  --warning: #f59e0b;
  --border-hl: #2d2d44;
  --card-hover: #161622;
}

html { font-size: 14px; }

body {
  background: var(--bg);
  color: var(--text-primary);
  font-family: 'Inter', -apple-system, sans-serif;
  line-height: 1.5;
  min-height: 100vh;
  overflow-x: hidden;
}

.mono { font-family: 'JetBrains Mono', monospace; }
.caps { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; font-weight: 600; color: var(--text-secondary); }

/* ── Animations ───────────────────────────────────────────── */
@keyframes pulse-green {
  0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5); }
  50% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
}
@keyframes pulse-red {
  0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
  50% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
@keyframes fade-in-up {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fade-in-up 0.5s ease-out both; }
.fade-in-d1 { animation-delay: 0.05s; }
.fade-in-d2 { animation-delay: 0.1s; }
.fade-in-d3 { animation-delay: 0.15s; }
.fade-in-d4 { animation-delay: 0.2s; }

.num-transition {
  transition: color 0.4s ease, opacity 0.3s ease;
}
.num-flash {
  animation: num-pop 0.4s ease;
}
@keyframes num-pop {
  0% { opacity: 0.5; transform: scale(0.97); }
  50% { transform: scale(1.02); }
  100% { opacity: 1; transform: scale(1); }
}

/* ── Layout ───────────────────────────────────────────────── */
.container {
  max-width: 1440px;
  margin: 0 auto;
  padding: 0 24px 40px;
}

/* ── Header ───────────────────────────────────────────────── */
.header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(10, 10, 15, 0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--card-border);
  padding: 14px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}
.header-logo {
  font-size: 1.3rem;
}
.header-title {
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--text-secondary);
}
.header-clock {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.1rem;
  font-weight: 500;
  color: var(--text-primary);
  letter-spacing: 0.02em;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}
.market-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 14px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.market-pill.open {
  background: rgba(16, 185, 129, 0.1);
  color: var(--positive);
  border: 1px solid rgba(16, 185, 129, 0.2);
}
.market-pill.closed {
  background: rgba(239, 68, 68, 0.1);
  color: var(--negative);
  border: 1px solid rgba(239, 68, 68, 0.2);
}
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}
.status-dot.open {
  background: var(--positive);
  animation: pulse-green 2s infinite;
}
.status-dot.closed {
  background: var(--negative);
  animation: pulse-red 2s infinite;
}
.est-time {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: var(--text-secondary);
}

/* ── Cards ────────────────────────────────────────────────── */
.stats-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-top: 24px;
}
.card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 24px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s ease, transform 0.2s ease;
}
.card:hover {
  border-color: var(--border-hl);
  transform: translateY(-1px);
}
.card-label {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-secondary);
  margin-bottom: 10px;
}
.card-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 2rem;
  font-weight: 700;
  line-height: 1.1;
  margin-bottom: 8px;
}
.card-sub {
  font-size: 0.78rem;
  color: var(--text-secondary);
}
.card-sub .mono {
  font-size: 0.75rem;
}
.return-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 10px;
  border-radius: 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  font-weight: 600;
}
.return-badge.positive {
  background: rgba(16, 185, 129, 0.12);
  color: var(--positive);
}
.return-badge.negative {
  background: rgba(239, 68, 68, 0.12);
  color: var(--negative);
}

/* ── Win rate ring ────────────────────────────────────────── */
.ring-container {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 16px;
}
.ring-svg {
  width: 56px;
  height: 56px;
  transform: rotate(-90deg);
}
.ring-bg {
  fill: none;
  stroke: var(--card-border);
  stroke-width: 5;
}
.ring-fg {
  fill: none;
  stroke-width: 5;
  stroke-linecap: round;
  transition: stroke-dashoffset 1s ease, stroke 0.4s ease;
}
.ring-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.8rem;
  font-weight: 700;
}

/* ── Regime bar ───────────────────────────────────────────── */
.regime-bar {
  margin-top: 16px;
  padding: 14px 24px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border: 1px solid var(--card-border);
  transition: background 0.5s ease, border-color 0.5s ease;
}
.regime-bar.uptrend {
  background: rgba(16, 185, 129, 0.05);
  border-color: rgba(16, 185, 129, 0.15);
}
.regime-bar.downtrend {
  background: rgba(239, 68, 68, 0.05);
  border-color: rgba(239, 68, 68, 0.15);
}
.regime-bar.neutral {
  background: rgba(245, 158, 11, 0.05);
  border-color: rgba(245, 158, 11, 0.15);
}
.regime-label {
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: 0.05em;
}
.regime-spy {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: var(--text-secondary);
}
.regime-spy span { color: var(--text-primary); font-weight: 600; }
.watchlist-pills {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.wl-pill {
  padding: 3px 10px;
  border-radius: 8px;
  font-size: 0.72rem;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
  background: rgba(99, 102, 241, 0.1);
  color: var(--accent);
  border: 1px solid rgba(99, 102, 241, 0.2);
}

/* ── Two-column layout ────────────────────────────────────── */
.two-col {
  display: grid;
  grid-template-columns: 3fr 2fr;
  gap: 16px;
  margin-top: 16px;
}
.section-card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 24px;
}
.section-title {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-secondary);
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.section-title .badge {
  background: var(--accent);
  color: white;
  font-size: 0.65rem;
  padding: 2px 8px;
  border-radius: 8px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
}

/* ── Chart ────────────────────────────────────────────────── */
.chart-wrap {
  position: relative;
  height: 280px;
}

/* ── Table ────────────────────────────────────────────────── */
.clean-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
}
.clean-table thead th {
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-secondary);
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--card-border);
}
.clean-table tbody td {
  padding: 12px;
  font-size: 0.85rem;
  border-bottom: 1px solid rgba(30, 30, 46, 0.5);
  transition: background 0.2s ease;
}
.clean-table tbody tr:hover td {
  background: rgba(99, 102, 241, 0.03);
}
.clean-table tbody tr:last-child td {
  border-bottom: none;
}
.ticker {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  font-size: 0.9rem;
}
.pnl-positive { color: var(--positive); font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.pnl-negative { color: var(--negative); font-family: 'JetBrains Mono', monospace; font-weight: 600; }

/* ── Progress bar (positions) ─────────────────────────────── */
.progress-bar-wrap {
  width: 80px;
  height: 6px;
  background: var(--card-border);
  border-radius: 3px;
  overflow: hidden;
  position: relative;
}
.progress-bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.6s ease;
  position: absolute;
}
.progress-bar-fill.pos { background: var(--positive); left: 50%; }
.progress-bar-fill.neg { background: var(--negative); right: 50%; }
.progress-center {
  position: absolute;
  left: 50%;
  top: -1px;
  width: 1px;
  height: 8px;
  background: var(--text-secondary);
}

/* ── Signal cards ─────────────────────────────────────────── */
.signal-stack {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 440px;
  overflow-y: auto;
  padding-right: 4px;
}
.signal-stack::-webkit-scrollbar { width: 4px; }
.signal-stack::-webkit-scrollbar-track { background: transparent; }
.signal-stack::-webkit-scrollbar-thumb { background: var(--card-border); border-radius: 2px; }

.signal-card {
  padding: 14px 16px;
  background: rgba(18, 18, 26, 0.6);
  border: 1px solid var(--card-border);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  transition: border-color 0.3s ease;
}
.signal-card:hover {
  border-color: var(--border-hl);
}
.signal-left {
  display: flex;
  align-items: center;
  gap: 12px;
}
.signal-ticker {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  font-size: 0.95rem;
}
.tier-badge {
  font-size: 0.6rem;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 6px;
  letter-spacing: 0.05em;
}
.tier-badge.t1 {
  background: rgba(99, 102, 241, 0.15);
  color: var(--accent);
  border: 1px solid rgba(99, 102, 241, 0.3);
}
.tier-badge.t2 {
  background: rgba(168, 85, 247, 0.15);
  color: #a855f7;
  border: 1px solid rgba(168, 85, 247, 0.3);
}
.tier-badge.t3 {
  background: rgba(100, 116, 139, 0.15);
  color: var(--text-secondary);
  border: 1px solid rgba(100, 116, 139, 0.3);
}
.signal-dots {
  display: flex;
  gap: 4px;
}
.signal-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  border: 1.5px solid var(--text-secondary);
  transition: background 0.3s ease, border-color 0.3s ease;
}
.signal-dot.active {
  background: var(--accent);
  border-color: var(--accent);
}
.signal-dot-label {
  font-size: 0.55rem;
  color: var(--text-secondary);
  text-align: center;
  margin-top: 2px;
  letter-spacing: 0.03em;
}
.signal-score {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.1rem;
  font-weight: 700;
}
.signal-status {
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-top: 2px;
}

/* ── Sentiment bars ───────────────────────────────────────── */
.sentiment-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.sentiment-ticker {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  font-weight: 600;
  width: 45px;
  text-align: right;
}
.sentiment-bar-wrap {
  flex: 1;
  height: 6px;
  background: linear-gradient(to right, rgba(239,68,68,0.2), rgba(100,116,139,0.1) 50%, rgba(16,185,129,0.2));
  border-radius: 3px;
  position: relative;
}
.sentiment-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  position: absolute;
  top: -3px;
  transition: left 0.6s ease, background 0.4s ease;
  border: 2px solid var(--card-bg);
}
.sentiment-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  width: 40px;
  text-align: left;
}

/* ── Trades table ─────────────────────────────────────────── */
.trades-section {
  margin-top: 16px;
}
.trade-row {
  border-left: 3px solid transparent;
}
.trade-row.buy { border-left-color: var(--accent); }
.trade-row.sell { border-left-color: var(--positive); }
.trade-row.stop { border-left-color: var(--negative); }
.trade-row.tp { border-left-color: var(--warning); }

.action-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 8px;
  font-size: 0.72rem;
  font-weight: 600;
}
.action-badge.buy {
  background: rgba(99, 102, 241, 0.12);
  color: var(--accent);
}
.action-badge.sell {
  background: rgba(16, 185, 129, 0.12);
  color: var(--positive);
}
.action-badge.stop {
  background: rgba(239, 68, 68, 0.12);
  color: var(--negative);
}
.action-badge.tp {
  background: rgba(245, 158, 11, 0.12);
  color: var(--warning);
}

/* ── Empty state ──────────────────────────────────────────── */
.empty-state {
  text-align: center;
  padding: 40px 20px;
  color: var(--text-secondary);
}
.empty-state i {
  font-size: 2rem;
  margin-bottom: 12px;
  opacity: 0.3;
  display: block;
}
.empty-state p {
  font-size: 0.85rem;
}

/* ── Footer ───────────────────────────────────────────────── */
.footer {
  margin-top: 32px;
  padding: 20px 0;
  border-top: 1px solid var(--card-border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.75rem;
  color: var(--text-secondary);
}
.footer-left {
  display: flex;
  align-items: center;
  gap: 16px;
}
.refresh-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
}
.refresh-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse-dot 2s infinite;
}
.fisher-badge {
  font-family: 'JetBrains Mono', monospace;
  padding: 3px 10px;
  border-radius: 8px;
  font-size: 0.7rem;
  font-weight: 600;
  background: rgba(99, 102, 241, 0.08);
  border: 1px solid rgba(99, 102, 241, 0.15);
  color: var(--accent);
}

/* ── Responsive ───────────────────────────────────────────── */
@media (max-width: 1100px) {
  .stats-row { grid-template-columns: repeat(2, 1fr); }
  .two-col { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .stats-row { grid-template-columns: 1fr; }
  .header { flex-direction: column; gap: 8px; }
}
</style>
</head>
<body>

<!-- ═══════════════ HEADER ═══════════════ -->
<header class="header">
  <div class="header-left">
    <span class="header-logo">🤖</span>
    <span class="header-title">Algo Trader</span>
  </div>
  <div class="header-clock" id="clock">--:--:--</div>
  <div class="header-right">
    <div class="market-pill closed" id="market-pill">
      <span class="status-dot closed" id="status-dot"></span>
      <span id="market-status-text">CLOSED</span>
    </div>
    <span class="est-time mono" id="est-time">--:-- EST</span>
  </div>
</header>

<div class="container">

  <!-- ═══════════════ HERO STATS ═══════════════ -->
  <div class="stats-row" id="stats-row">
    <!-- Card 1: Portfolio Value -->
    <div class="card fade-in fade-in-d1">
      <div class="card-label">Portfolio Value</div>
      <div class="card-value mono num-transition" id="portfolio-value">$--</div>
      <div class="card-sub">
        <span class="return-badge positive" id="return-badge">
          <i class="fa-solid fa-arrow-up" style="font-size:0.65rem" id="return-icon"></i>
          <span id="return-pct">0.00%</span>
        </span>
        <span style="margin-left:8px;font-size:0.72rem" class="mono" id="cash-display">Cash: $--</span>
      </div>
    </div>

    <!-- Card 2: Today's P&L -->
    <div class="card fade-in fade-in-d2">
      <div class="card-label">Today's P&L</div>
      <div class="card-value mono num-transition" id="today-pnl">$--</div>
      <div class="card-sub">
        <span class="mono" id="pnl-breakdown" style="font-size:0.72rem">Unrealized: $-- &middot; Realized: $--</span>
      </div>
    </div>

    <!-- Card 3: Win Rate -->
    <div class="card fade-in fade-in-d3">
      <div class="card-label">Win Rate</div>
      <div class="ring-container">
        <svg class="ring-svg" viewBox="0 0 40 40">
          <circle class="ring-bg" cx="20" cy="20" r="16"></circle>
          <circle class="ring-fg" id="win-ring" cx="20" cy="20" r="16"
                  stroke="var(--positive)"
                  stroke-dasharray="100.53"
                  stroke-dashoffset="100.53"></circle>
        </svg>
        <div>
          <div class="ring-value num-transition" id="win-rate">0%</div>
          <div class="card-sub mono" id="win-count" style="font-size:0.7rem">0 / 0 trades</div>
        </div>
      </div>
    </div>

    <!-- Card 4: Sharpe Ratio -->
    <div class="card fade-in fade-in-d4">
      <div class="card-label">Sharpe Ratio</div>
      <div class="card-value mono num-transition" id="sharpe-value">--</div>
      <div class="card-sub" id="sharpe-sub"></div>
    </div>
  </div>

  <!-- ═══════════════ REGIME BAR ═══════════════ -->
  <div class="regime-bar neutral fade-in" id="regime-bar">
    <div class="regime-spy">
      SPY <span id="spy-price">--</span>
      <span style="margin:0 4px;color:var(--text-secondary)">vs</span>
      50 EMA <span id="spy-ema">--</span>
      <span id="regime-arrow" style="margin-left:6px"></span>
    </div>
    <div class="regime-label" id="regime-label">&#10145;&#65039; NEUTRAL</div>
    <div class="watchlist-pills" id="watchlist-pills"></div>
  </div>

  <!-- ═══════════════ TWO COLUMN ═══════════════ -->
  <div class="two-col">

    <!-- LEFT COLUMN -->
    <div>
      <!-- Equity Curve -->
      <div class="section-card fade-in">
        <div class="section-title">
          <i class="fa-solid fa-chart-line" style="color:var(--accent)"></i>
          Equity Curve
        </div>
        <div class="chart-wrap">
          <canvas id="equity-chart"></canvas>
        </div>
      </div>

      <!-- Open Positions -->
      <div class="section-card fade-in" style="margin-top:16px">
        <div class="section-title">
          <i class="fa-solid fa-layer-group" style="color:var(--accent)"></i>
          Open Positions
          <span class="badge" id="pos-count">0</span>
        </div>
        <div id="positions-container">
          <div class="empty-state">
            <i class="fa-regular fa-folder-open"></i>
            <p>No open positions</p>
          </div>
        </div>
      </div>
    </div>

    <!-- RIGHT COLUMN -->
    <div>
      <!-- Signal Strength -->
      <div class="section-card fade-in">
        <div class="section-title">
          <i class="fa-solid fa-signal" style="color:var(--accent)"></i>
          Signal Strength
        </div>
        <div class="signal-stack" id="signal-stack">
          <div class="empty-state">
            <i class="fa-solid fa-signal"></i>
            <p>No signals</p>
          </div>
        </div>
      </div>

      <!-- Sentiment Gauge -->
      <div class="section-card fade-in" style="margin-top:16px">
        <div class="section-title">
          <i class="fa-solid fa-gauge-high" style="color:var(--accent)"></i>
          Sentiment Scores
        </div>
        <div id="sentiment-container">
          <div class="empty-state">
            <p style="font-size:0.78rem">Sentiment data updates during market hours</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ═══════════════ TRADE HISTORY ═══════════════ -->
  <div class="section-card trades-section fade-in">
    <div class="section-title">
      <i class="fa-solid fa-clock-rotate-left" style="color:var(--accent)"></i>
      Trade History
    </div>
    <div id="trades-container">
      <div class="empty-state">
        <i class="fa-regular fa-clock"></i>
        <p>No trades yet</p>
      </div>
    </div>
  </div>

  <!-- ═══════════════ FOOTER ═══════════════ -->
  <div class="footer">
    <div class="footer-left">
      <span id="last-updated">Last updated: --</span>
      <div class="refresh-indicator">
        <span class="refresh-dot"></span>
        <span>Auto-refreshing every 15s</span>
      </div>
    </div>
    <div id="fisher-container"></div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════
//  STATE & CHART
// ═══════════════════════════════════════════════════════════════

let equityChart = null;
let previousData = {};

function initChart() {
  const ctx = document.getElementById('equity-chart').getContext('2d');
  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Portfolio Value',
        data: [],
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.08)',
        borderWidth: 2.5,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: '#6366f1',
        pointHoverBorderColor: '#fff',
        pointHoverBorderWidth: 2,
      }, {
        label: 'Starting Cash',
        data: [],
        borderColor: 'rgba(100, 116, 139, 0.3)',
        borderWidth: 1,
        borderDash: [6, 4],
        fill: false,
        pointRadius: 0,
        pointHoverRadius: 0,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: 'index',
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(18, 18, 26, 0.95)',
          titleColor: '#e2e8f0',
          bodyColor: '#e2e8f0',
          borderColor: '#2d2d44',
          borderWidth: 1,
          cornerRadius: 10,
          padding: 12,
          titleFont: { family: 'Inter', size: 12, weight: '600' },
          bodyFont: { family: 'JetBrains Mono', size: 13, weight: '600' },
          callbacks: {
            label: function(ctx) {
              if (ctx.datasetIndex === 0) {
                return ' $' + ctx.parsed.y.toLocaleString('en-US', {minimumFractionDigits: 2});
              }
              return null;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 10 },
            maxTicksLimit: 8,
          },
          border: { display: false },
        },
        y: {
          grid: {
            color: 'rgba(30, 30, 46, 0.4)',
            lineWidth: 1,
          },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: function(v) { return '$' + v.toLocaleString(); },
          },
          border: { display: false },
        }
      }
    }
  });
}

// ═══════════════════════════════════════════════════════════════
//  CLOCK
// ═══════════════════════════════════════════════════════════════

function updateClock() {
  var now = new Date();
  document.getElementById('clock').textContent =
    now.toLocaleTimeString('en-US', { hour12: false });

  var est = now.toLocaleTimeString('en-US', {
    timeZone: 'America/New_York',
    hour: '2-digit', minute: '2-digit', hour12: true
  });
  document.getElementById('est-time').textContent = est + ' EST';
}
setInterval(updateClock, 1000);
updateClock();

// ═══════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════

function fmt(n, decimals) {
  decimals = decimals === undefined ? 2 : decimals;
  return Number(n).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

function fmtSign(n, decimals) {
  decimals = decimals === undefined ? 2 : decimals;
  var prefix = n >= 0 ? '+' : '';
  return prefix + fmt(n, decimals);
}

function flashElement(el) {
  el.classList.remove('num-flash');
  void el.offsetWidth;
  el.classList.add('num-flash');
}

function getActionClass(action) {
  var a = action.toUpperCase();
  if (a.indexOf('STOP') >= 0) return 'stop';
  if (a.indexOf('TAKE') >= 0 || a.indexOf('PROFIT') >= 0) return 'tp';
  if (a.indexOf('SELL') >= 0) return 'sell';
  return 'buy';
}

function getActionIcon(action) {
  var a = action.toUpperCase();
  if (a.indexOf('STOP') >= 0) return '<i class="fa-solid fa-hand"></i>';
  if (a.indexOf('TAKE') >= 0 || a.indexOf('PROFIT') >= 0) return '<i class="fa-solid fa-bullseye"></i>';
  if (a.indexOf('SELL') >= 0) return '<i class="fa-solid fa-arrow-right-from-bracket"></i>';
  return '<i class="fa-solid fa-arrow-right-to-bracket"></i>';
}

// ═══════════════════════════════════════════════════════════════
//  RENDER
// ═══════════════════════════════════════════════════════════════

function render(data) {
  // ── Market status ──
  var pill = document.getElementById('market-pill');
  var dot = document.getElementById('status-dot');
  var statusText = document.getElementById('market-status-text');
  if (data.market_open) {
    pill.className = 'market-pill open';
    dot.className = 'status-dot open';
    statusText.textContent = 'MARKET OPEN';
  } else {
    pill.className = 'market-pill closed';
    dot.className = 'status-dot closed';
    statusText.textContent = 'MARKET CLOSED';
  }

  // ── Portfolio Value ──
  var pvEl = document.getElementById('portfolio-value');
  pvEl.textContent = '$' + fmt(data.portfolio_value);
  if (previousData.portfolio_value !== data.portfolio_value) flashElement(pvEl);

  var retBadge = document.getElementById('return-badge');
  var retIcon = document.getElementById('return-icon');
  var retPct = document.getElementById('return-pct');
  retPct.textContent = fmtSign(data.total_return, 2) + '%';
  if (data.total_return >= 0) {
    retBadge.className = 'return-badge positive';
    retIcon.className = 'fa-solid fa-arrow-up';
  } else {
    retBadge.className = 'return-badge negative';
    retIcon.className = 'fa-solid fa-arrow-down';
  }
  document.getElementById('cash-display').textContent = 'Cash: $' + fmt(data.cash);

  // ── Today's P&L ──
  var pnlEl = document.getElementById('today-pnl');
  var pnlVal = data.today_pnl;
  pnlEl.textContent = '$' + fmtSign(pnlVal);
  pnlEl.style.color = pnlVal >= 0 ? 'var(--positive)' : 'var(--negative)';
  if (previousData.today_pnl !== data.today_pnl) flashElement(pnlEl);

  document.getElementById('pnl-breakdown').innerHTML =
    'Unrealized: <span class="mono" style="color:' + (data.unrealized_pnl >= 0 ? 'var(--positive)' : 'var(--negative)') + '">$' + fmtSign(data.unrealized_pnl) + '</span>' +
    ' &middot; Realized: <span class="mono" style="color:' + (data.realized_pnl >= 0 ? 'var(--positive)' : 'var(--negative)') + '">$' + fmtSign(data.realized_pnl) + '</span>';

  // ── Win Rate ──
  var wr = data.win_rate || 0;
  document.getElementById('win-rate').textContent = fmt(wr, 1) + '%';
  document.getElementById('win-count').textContent = data.wins + ' / ' + data.total_closed + ' trades';
  var ring = document.getElementById('win-ring');
  var circumference = 2 * Math.PI * 16;
  ring.style.strokeDasharray = circumference;
  ring.style.strokeDashoffset = circumference - (wr / 100) * circumference;
  if (wr >= 50) {
    ring.style.stroke = 'var(--positive)';
  } else if (wr >= 30) {
    ring.style.stroke = 'var(--warning)';
  } else {
    ring.style.stroke = 'var(--negative)';
  }

  // ── Sharpe ──
  var sharpeEl = document.getElementById('sharpe-value');
  sharpeEl.textContent = fmt(data.sharpe, 2);
  if (data.sharpe > 1.0) {
    sharpeEl.style.color = 'var(--positive)';
  } else if (data.sharpe >= 0) {
    sharpeEl.style.color = 'var(--warning)';
  } else {
    sharpeEl.style.color = 'var(--negative)';
  }
  var sharpeSub = document.getElementById('sharpe-sub');
  sharpeSub.textContent = data.sharpe > 2.0 ? 'Excellent' : data.sharpe > 1.0 ? 'Good' : data.sharpe >= 0 ? 'Moderate' : 'Poor';

  // ── Regime bar ──
  var regimeBar = document.getElementById('regime-bar');
  var regimeLabel = document.getElementById('regime-label');
  var regimeArrow = document.getElementById('regime-arrow');
  regimeBar.className = 'regime-bar ' + data.regime.toLowerCase();
  if (data.regime === 'UPTREND') {
    regimeLabel.innerHTML = '<span style="color:var(--positive)">\\ud83d\\udcc8 UPTREND</span>';
    regimeArrow.innerHTML = '<i class="fa-solid fa-arrow-trend-up" style="color:var(--positive)"></i>';
  } else if (data.regime === 'DOWNTREND') {
    regimeLabel.innerHTML = '<span style="color:var(--negative)">\\ud83d\\udcc9 DOWNTREND</span>';
    regimeArrow.innerHTML = '<i class="fa-solid fa-arrow-trend-down" style="color:var(--negative)"></i>';
  } else {
    regimeLabel.innerHTML = '<span style="color:var(--warning)">\\u27a1\\ufe0f NEUTRAL</span>';
    regimeArrow.innerHTML = '<i class="fa-solid fa-arrows-left-right" style="color:var(--warning)"></i>';
  }
  if (data.spy_price) {
    document.getElementById('spy-price').textContent = '$' + fmt(data.spy_price);
    document.getElementById('spy-ema').textContent = '$' + fmt(data.spy_ema);
  }

  // Watchlist pills
  var pillsContainer = document.getElementById('watchlist-pills');
  pillsContainer.innerHTML = data.watchlist.map(function(s) { return '<span class="wl-pill">' + s + '</span>'; }).join('');

  // ── Equity Chart ──
  if (equityChart && data.equity_dates && data.equity_dates.length > 0) {
    equityChart.data.labels = data.equity_dates;
    equityChart.data.datasets[0].data = data.equity_values;
    equityChart.data.datasets[1].data = data.equity_dates.map(function() { return 100000; });
    equityChart.update('none');
  }

  // ── Positions ──
  var posContainer = document.getElementById('positions-container');
  document.getElementById('pos-count').textContent = data.positions.length;
  if (data.positions.length === 0) {
    posContainer.innerHTML = '<div class="empty-state"><i class="fa-regular fa-folder-open"></i><p>No open positions</p></div>';
  } else {
    var html = '<table class="clean-table"><thead><tr>' +
      '<th>Symbol</th><th>Qty</th><th>Entry</th><th>Current</th>' +
      '<th>P&L $</th><th>P&L %</th><th>Stop</th><th>Target</th><th>Progress</th>' +
      '</tr></thead><tbody>';
    data.positions.forEach(function(p) {
      var cls = p.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
      var barHtml;
      if (p.progress >= 0) {
        var w = Math.min(Math.abs(p.progress), 100) / 2;
        barHtml = '<div class="progress-bar-wrap">' +
          '<div class="progress-center"></div>' +
          '<div class="progress-bar-fill pos" style="width:' + w + '%;left:50%"></div></div>';
      } else {
        var w2 = Math.min(Math.abs(p.progress), 100) / 2;
        barHtml = '<div class="progress-bar-wrap">' +
          '<div class="progress-center"></div>' +
          '<div class="progress-bar-fill neg" style="width:' + w2 + '%;right:50%"></div></div>';
      }
      html += '<tr>' +
        '<td><span class="ticker">' + p.symbol + '</span></td>' +
        '<td class="mono">' + p.qty + '</td>' +
        '<td class="mono">$' + fmt(p.entry) + '</td>' +
        '<td class="mono">$' + fmt(p.current) + '</td>' +
        '<td class="' + cls + '">$' + fmtSign(p.pnl) + '</td>' +
        '<td class="' + cls + '">' + fmtSign(p.pnl_pct) + '%</td>' +
        '<td class="mono" style="color:var(--negative);font-size:0.78rem">$' + fmt(p.stop) + '</td>' +
        '<td class="mono" style="color:var(--positive);font-size:0.78rem">$' + fmt(p.target) + '</td>' +
        '<td>' + barHtml + '</td>' +
        '</tr>';
    });
    html += '</tbody></table>';
    posContainer.innerHTML = html;
  }

  // ── Signal Strength ──
  var signalStack = document.getElementById('signal-stack');
  if (data.signals && data.signals.length > 0) {
    signalStack.innerHTML = data.signals.map(function(s) {
      var tierClass = s.tier.toLowerCase();
      var isPass = s.status === 'PASS';
      var isWeak = s.status === 'WEAK';
      var activeDots = isPass ? 4 : isWeak ? 2 : 1;
      var labels = ['EMA', 'RSI', 'BB', 'MACD', 'VWAP'];
      var dotsHtml = labels.map(function(lbl, i) {
        return '<div style="text-align:center">' +
          '<div class="signal-dot ' + (i < activeDots ? 'active' : '') + '"></div>' +
          '<div class="signal-dot-label">' + lbl + '</div></div>';
      }).join('');

      var scoreColor = activeDots >= 4 ? 'var(--positive)' : activeDots >= 2 ? 'var(--warning)' : 'var(--negative)';

      return '<div class="signal-card">' +
        '<div class="signal-left">' +
          '<div>' +
            '<div style="display:flex;align-items:center;gap:8px">' +
              '<span class="signal-ticker">' + s.stock + '</span>' +
              '<span class="tier-badge ' + tierClass + '">' + s.tier + '</span>' +
            '</div>' +
            '<div class="signal-dots" style="margin-top:6px">' + dotsHtml + '</div>' +
          '</div>' +
        '</div>' +
        '<div style="text-align:right">' +
          '<div class="signal-score" style="color:' + scoreColor + '">' + activeDots + '/5</div>' +
          '<div class="signal-status" style="color:' + (isPass ? 'var(--positive)' : isWeak ? 'var(--warning)' : 'var(--text-secondary)') + '">' + s.status + '</div>' +
        '</div>' +
      '</div>';
    }).join('');
  }

  // ── Sentiment ──
  var sentContainer = document.getElementById('sentiment-container');
  if (data.watchlist && data.watchlist.length > 0) {
    sentContainer.innerHTML = data.watchlist.map(function(stock) {
      var val = data.regime === 'UPTREND' ? 0.3 : data.regime === 'DOWNTREND' ? -0.3 : 0.05;
      var pct = ((val + 1) / 2) * 100;
      var dotColor = val > 0.15 ? 'var(--positive)' : val < -0.15 ? 'var(--negative)' : 'var(--text-secondary)';
      return '<div class="sentiment-row">' +
        '<span class="sentiment-ticker">' + stock + '</span>' +
        '<div class="sentiment-bar-wrap">' +
          '<div class="sentiment-dot" style="left:calc(' + pct + '% - 6px);background:' + dotColor + '"></div>' +
        '</div>' +
        '<span class="sentiment-val mono" style="color:' + dotColor + '">' + (val >= 0 ? '+' : '') + val.toFixed(2) + '</span>' +
      '</div>';
    }).join('');
  }

  // ── Trades ──
  var tradesContainer = document.getElementById('trades-container');
  if (data.recent_trades && data.recent_trades.length > 0) {
    var thtml = '<table class="clean-table"><thead><tr>' +
      '<th>Time</th><th>Stock</th><th>Action</th><th>Price</th>' +
      '</tr></thead><tbody>';
    data.recent_trades.forEach(function(t) {
      var cls = getActionClass(t.action);
      thtml += '<tr class="trade-row ' + cls + '">' +
        '<td class="mono" style="font-size:0.78rem;color:var(--text-secondary)">' + t.timestamp + '</td>' +
        '<td><span class="ticker">' + t.stock + '</span></td>' +
        '<td><span class="action-badge ' + cls + '">' + getActionIcon(t.action) + ' ' + t.action + '</span></td>' +
        '<td class="mono">$' + fmt(t.price) + '</td>' +
        '</tr>';
    });
    thtml += '</tbody></table>';
    tradesContainer.innerHTML = thtml;
  }

  // ── Footer ──
  document.getElementById('last-updated').textContent = 'Last updated: ' + data.updated;
  var fisherContainer = document.getElementById('fisher-container');
  if (data.fisher_p !== null && data.fisher_p !== undefined) {
    fisherContainer.innerHTML = '<span class="fisher-badge">Fisher p-value: ' + data.fisher_p.toFixed(4) + '</span>';
  }

  previousData = data;
}

// ═══════════════════════════════════════════════════════════════
//  FETCH & REFRESH
// ═══════════════════════════════════════════════════════════════

function fetchData() {
  fetch('/api/data')
    .then(function(resp) { return resp.json(); })
    .then(function(data) { render(data); })
    .catch(function(e) { console.error('Fetch error:', e); });
}

initChart();
fetchData();
setInterval(fetchData, 15000);
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("Dashboard running at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.")
    app.run(debug=False)
