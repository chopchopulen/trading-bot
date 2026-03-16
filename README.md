# Algorithmic Trading Bot

A fully automated intraday trading bot running on **Alpaca Paper Trading** ($100K portfolio). Built from scratch as a learning project exploring quantitative finance, technical analysis, and systematic trading.

## What It Does

The bot trades **26 stocks** across 7 sectors on 1-minute bars during market hours (9:45 AM - 3:45 PM ET). It combines multiple technical indicators with sentiment analysis to generate buy/sell signals, manages risk with ATR-based position sizing and dynamic stop losses, and serves a real-time dashboard.

## Architecture

```
bot.py              ← Live trading engine (runs during market hours)
dashboard.py        ← Flask web dashboard (portfolio, positions, trades)
walk_forward.py     ← Walk-forward parameter optimizer
backtest_minutes.py ← Minute-bar backtester with charting
backtest.py         ← Daily-bar backtester with grid search
diagnose_signals.py ← Signal diagnostics & debugging tool
```

## Signal Chain

Every buy signal must pass **all** of these filters (AND logic):

1. **SPY Regime Filter** — SPY must be above its 50-period EMA (uptrend)
2. **EMA Crossover** — Fast EMA (6) > Slow EMA (10) for buys
3. **RSI OR Bollinger Bands** — RSI(7) < 35 (oversold) **OR** price <= BB lower band (period 15, std 1.5)
4. **MACD** — MACD(12/26/9) bullish crossover for buys
5. **VWAP** — Price must be below VWAP (buying below fair value)

> **Design note:** RSI and BB use OR logic because both measure "oversold" conditions. Requiring both simultaneously (AND) on 1-minute bars produces zero signals — diagnosed via `diagnose_signals.py` which showed the progressive AND elimination: `Regime+EMA → 700 bars → +RSI → 30 bars → +BB → 0 bars`.

Sell signals mirror this with reversed conditions (EMA bearish, RSI > 55 OR price >= BB upper, MACD bearish, price > VWAP).

## Risk Management

| Feature | Implementation |
|---------|---------------|
| **Position Sizing** | ATR-based: `qty = (portfolio * 1%) / ATR(14)` — risk $1K per trade on $100K |
| **Stop Loss** | Dynamic: `entry - 1.5 * ATR(14)` — adapts to each stock's volatility |
| **Take Profit** | Dynamic: `entry + 3.0 * ATR(14)` — 1:2 risk/reward ratio |
| **Max Positions** | 5 concurrent positions |
| **Daily Loss Limit** | -$500 — stops all new buys if hit |
| **Market Window** | Only trades 9:45 AM - 3:45 PM ET (avoids open/close volatility) |
| **Regime Filter** | No buys when SPY < 50 EMA (bear market protection) |

## Stock Universe (26 Stocks)

| Sector | Stocks |
|--------|--------|
| Tech | AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA, META, AMD, NFLX, CRM |
| Finance | JPM, BAC, GS, V |
| ETFs | SPY, QQQ |
| Energy | XOM, CVX |
| Healthcare | JNJ, PFE, UNH |
| Consumer | WMT, COST, NKE |
| Semiconductor | INTC, QCOM |

## Sentiment Analysis

The bot uses **NewsAPI + VADER** to score each stock's news sentiment:
- Fetches recent headlines for each stock
- Scores each headline with VADER compound sentiment
- Time-weights scores (recent news counts more)
- Caches results for 30 minutes to stay within API limits
- Pre-market scan at 4 AM ET identifies high-sentiment opportunities

## Walk-Forward Optimizer

`walk_forward.py` prevents overfitting by splitting data into training and test windows:
- **Train:** 20 days of 1-minute bars — find best parameters from 3,312 combinations
- **Test:** Next 10 days — validate on unseen data
- **Pass criteria:** Test return > 0%, trades >= 2, Sharpe > 1.0
- **Parameter grid:** EMA fast/slow, RSI buy/sell thresholds, BB period/std

Includes a transaction cost model (spread, slippage, SEC fees) for realistic results.

## Dashboard

Flask-based web dashboard at `http://127.0.0.1:5000` with auto-refresh every 30 seconds:
- Portfolio value, cash, total return, alpha vs S&P 500
- Open positions with P&L
- Recent trade log

## Setup

### Prerequisites
- Python 3.x
- Alpaca paper trading account
- NewsAPI key (free tier)

### Installation

```bash
pip install alpaca-trade-api pandas numpy python-dotenv flask newsapi-python vaderSentiment matplotlib schedule
```

### Configuration

Create a `.env` file:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
NEWS_API_KEY=your_newsapi_key_here
```

### Running

```bash
# Live trading bot
python3 bot.py

# Dashboard (separate terminal)
python3 dashboard.py

# Run backtests
python3 backtest_minutes.py
python3 backtest.py

# Optimize parameters
python3 walk_forward.py

# Diagnose signal issues
python3 diagnose_signals.py
```

## Project Journey

This project evolved through several phases:

### Phase 1: Basic EMA Crossover
Started with a simple EMA crossover strategy on daily bars. Single stock (TSLA), fixed position sizes, no risk management.

### Phase 2: Multi-Indicator System
Added RSI, Bollinger Bands, and MACD as confirmation signals. Expanded to 10 stocks. Moved from daily to 1-minute bars for intraday trading.

### Phase 3: Risk Management
Replaced fixed percentage stops with **ATR-based dynamic stops** that adapt to each stock's volatility. Added ATR position sizing (risk 1% of portfolio per trade), max position limits, and daily loss limits.

### Phase 4: Market Context
Added **SPY regime detection** (only buy in uptrends) and **VWAP filter** (only buy below fair value). These macro filters prevent the bot from buying into falling markets or chasing momentum.

### Phase 5: Sentiment & Pre-Market
Integrated **NewsAPI + VADER sentiment analysis** with time-weighted scoring and 30-minute caching. Added a pre-market scan at 4 AM ET to identify stocks with positive news.

### Phase 6: Walk-Forward Optimization
Built a walk-forward optimizer to find optimal parameters without overfitting. Discovered that 6 AND conditions on 1-minute bars produce zero signals — **the progressive AND elimination problem**. Fixed by switching RSI and Bollinger Bands from AND to OR logic (both measure oversold, only one needs to trigger).

### Phase 7: Expansion (Planned)
- Re-run optimizer on all 26 stocks (only 10 tested so far)
- Optimize stop/profit multipliers per stock
- VIX regime filter (reduce size in high-volatility markets)
- Pairs trading (statistical arbitrage between correlated stocks)
- Expand to 36 stocks across 10 sectors

## Backtest Results

Charts from backtesting are saved as PNG files:
- `backtest_results.png` — Daily bar backtest equity curve
- `backtest_minutes.png` — Minute bar backtest equity curve
- `performance.png` — Strategy performance analysis

## Disclaimer

This is a **paper trading** project for educational purposes. Past backtest performance does not guarantee future results. Do not use this bot with real money without extensive additional testing and risk management.
