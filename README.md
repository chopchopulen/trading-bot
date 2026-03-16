# Algorithmic Trading Bot

A fully automated intraday trading bot running on **Alpaca Paper Trading** ($100K portfolio). Built from scratch as a learning project exploring quantitative finance, technical analysis, and systematic trading.

## What It Does

The bot trades a **tiered stock universe** (18 validated stocks across 3 tiers) on 1-minute bars during market hours (9:45 AM - 3:45 PM ET). It uses a **weighted signal scoring system** to generate buy, sell, short, and cover signals, manages risk with ATR-based position sizing and dynamic stop losses, supports **short selling in downtrends**, and serves a real-time dashboard. Stocks are validated through a 3-stage pipeline: backtest -> walk-forward optimization -> permutation test.

## Architecture

```
signals.py            <- Shared signal module (scoring, indicators, thresholds)
bot.py                <- Live trading engine (scoring, shorts, tiered stocks)
overnight_pipeline.py <- Master overnight test runner (4 phases)
dashboard.py          <- Flask web dashboard (portfolio, positions, trades)
walk_forward.py       <- Walk-forward parameter optimizer (3,312 combos/stock)
permutation_test.py   <- Statistical significance testing (dual methods)
backtest_minutes.py   <- Minute-bar backtester with charting
backtest.py           <- Daily-bar backtester with grid search
diagnose_signals.py   <- Signal diagnostics & debugging tool
```

## Signal Scoring System

The bot uses a **weighted scoring system** instead of strict AND conditions. Each indicator contributes a weight, and the total score must exceed a tier-based threshold to trigger a trade.

### Buy Signal Weights

| Signal | Weight | Condition |
|--------|--------|-----------|
| EMA Trend | 3.0 | Fast EMA > Slow EMA |
| RSI Oversold | 2.0 | RSI < buy threshold |
| Bollinger Band | 2.0 | Price <= BB lower band |
| MACD Bullish | 1.5 | MACD line > signal line |
| VWAP | 1.0 | Price < VWAP |
| Regime (Uptrend) | 2.5 | SPY > 50 EMA |
| Sentiment | 0.5 | News sentiment > 0.15 |

**Max possible score: 12.5**

### Thresholds by Tier

| Tier | Buy | Sell | Short Entry | Short Cover |
|------|-----|------|-------------|-------------|
| Tier 1 (proven) | 6.0 | 5.5 | 7.0 | 5.5 |
| Tier 2 (promising) | 7.0 | 6.0 | 8.0 | 6.0 |
| Tier 3 (monitoring) | 8.0 | 7.0 | disabled | disabled |

**Key design decision:** The SPY regime filter is a **soft signal** (2.5 points), not a hard gate. In strong downtrends, longs can still fire if other signals are overwhelmingly bullish. This also enables short selling when the regime is bearish.

All scoring logic lives in `signals.py` — a single source of truth shared by the live bot, backtester, walk-forward optimizer, and overnight pipeline.

## Short Selling

When the market is in a **downtrend** (SPY < 50 EMA), the bot can short highly liquid stocks:

```
SHORT_ELIGIBLE = AAPL, MSFT, META, AMD, TSLA, NFLX, NVDA, AMZN, GS
```

### Short Signals (inverted buy signals)

| Signal | Weight | Condition |
|--------|--------|-----------|
| EMA Bearish | 3.0 | Fast EMA < Slow EMA |
| RSI Overbought | 2.0 | RSI > sell threshold |
| BB Upper | 2.0 | Price >= BB upper band |
| MACD Bearish | 1.5 | MACD < signal line |
| Above VWAP | 1.0 | Price > VWAP |
| Regime (Downtrend) | 2.5 | SPY < 50 EMA |
| Negative Sentiment | 0.5 | Sentiment < -0.15 |

### Short Risk Guardrails

| Feature | Setting |
|---------|---------|
| Max short positions | 3 |
| Short daily loss limit | -$300 |
| Hard stop | 5% above entry (force close) |
| ATR stop loss | Entry + 1.5x ATR |
| ATR take profit | Entry - 3.0x ATR |
| Regime switch | Auto-cover all shorts if SPY flips to uptrend |

## Risk Management

| Feature | Implementation |
|---------|---------------|
| **Position Sizing** | ATR-based: `qty = (portfolio * 1%) / ATR(14)` — T1 full, T2 50%, T3 25% |
| **Stop Loss** | Dynamic: `entry - 1.5 * ATR(14)` (longs) / `entry + 1.5 * ATR` (shorts) |
| **Take Profit** | Dynamic: `entry + 3.0 * ATR(14)` (longs) / `entry - 3.0 * ATR` (shorts) |
| **Max Positions** | 5 concurrent (longs) + 3 concurrent (shorts) |
| **Daily Loss Limit** | -$500 (longs), -$300 (shorts) |
| **Market Window** | Only trades 9:45 AM - 3:45 PM ET |
| **Backtest Gate** | No buys if stock's backtest profit factor < 1.0 |
| **Sector Limits** | Max 2 positions per sector |
| **Anti-Whipsaw** | 5-bar cooldown after trades, 10-min minimum hold |

### Sector Diversification

```
tech:     MSFT, META, AAPL, AMZN, NFLX, CRM
semi:     AMD, NVDA, QCOM
finance:  GS, COIN, HOOD
consumer: WMT, COST, TSLA, UBER
etf:      SPY
other:    SPOT
```

## Stock Universe (Tiered)

Stocks are assigned to tiers based on permutation test p-values and backtest performance:

| Tier | Sizing | Stocks | Criteria |
|------|--------|--------|----------|
| **Tier 1** | Full ATR | MSFT, META, AMD, SPY | p < 0.05, proven edge |
| **Tier 2** | 50% ATR | AAPL, TSLA, NFLX, WMT, COST, GS, HOOD, UBER, COIN | p < 0.15, promising |
| **Tier 3** | 25% ATR | SPOT, CRM, QCOM, AMZN, NVDA | Monitoring, higher thresholds |

**Pipeline candidates** (tested overnight, not yet traded): AVGO, LLY, MA, PANW, CRWD, SNOW

The overnight pipeline auto-classifies stocks into tiers based on p-values and profit factors.

## Sentiment Analysis

The bot uses **NewsAPI + VADER** to score each stock's news sentiment:
- Fetches recent headlines for each stock
- Scores each headline with VADER compound sentiment
- Time-weights scores (recent news counts more)
- Caches results for 30 minutes to stay within API limits
- Pre-market scan at 4 AM ET identifies high-sentiment opportunities

## Walk-Forward Optimizer

`walk_forward.py` prevents overfitting by splitting data into training and test windows:
- **Lookback:** 90 days of 1-minute bars
- **Train:** 60 days — find best parameters from 3,312 combinations
- **Test:** 30 days — validate on unseen data
- **Pass criteria:** Test return > 0%, trades >= 2, Sharpe > 1.0
- **Parameter grid:** EMA fast/slow (24 combos), RSI buy/sell thresholds, BB period/std
- **Simple signal mode:** Optional flag to test EMA + RSI only (10-20x more trades)
- **All 26+ stocks** tested with per-stock JSON output to `walk_forward_results/`

Includes a transaction cost model (per-stock spread estimates, slippage, SEC fees) for realistic results.

## Permutation Test (Statistical Validation)

`permutation_test.py` answers: **"Is this strategy's performance real or just luck?"**

**Two test methods:**
1. **Day-block shuffle** — randomly reorder entire trading days, keeping intraday bars intact but destroying inter-day trends
2. **Signal shift** — randomly offset entry signal indices and simulate ATR stop/target outcomes at shifted locations (tests whether entry timing matters)

**Statistical framework:**
- **1,000 permutations** per stock (p-value resolution: 0.001)
- **Bonferroni correction** for 26 stocks: alpha = 0.05/26 = 0.00192
- **Fisher's method** combines per-stock p-values into a single portfolio-level number
- **White's Reality Check** adjustment for data mining bias: `0.05 / sqrt(3312) = 0.000869`
- **Sharpe cap:** Returns 0 when trades < 2 to prevent Sharpe explosion
- **Tiered summary** with actionable recommendations per stock

### Latest Results

| Verdict | Stocks |
|---------|--------|
| **Strong** (p < 0.05) | META (p=0.012), GS (p=0.002), AMD (p=0.002), SPY (p=0.002), HOOD (p=0.022), SPOT (p=0.020) |
| **Promising** (p < 0.15) | WMT (p=0.072), COST (p=0.092), MSFT (p=0.026), NFLX (p=0.064), UBER (p=0.106), QCOM (p=0.104), COIN (p=0.122), CRM (p=0.136) |
| **Weak** (p < 0.30) | AAPL, TSLA, AMZN, NVDA |

## Overnight Pipeline

`overnight_pipeline.py` runs all validation phases in sequence overnight (~2-4 hours):

| Phase | What | Output |
|-------|------|--------|
| **Phase 1** | Backtest all stocks (grid search, 10 param sets) | `backtest_results/{STOCK}.json` |
| **Phase 2** | Walk-forward on untested stocks | `walk_forward_results/{STOCK}.json` |
| **Phase 3** | Permutation test on profitable stocks (500 perms) | `permutation_test_results/{STOCK}.json` |
| **Phase 4** | Master summary report with tiered recommendations | `results/overnight_summary.txt` |

```bash
python3 overnight_pipeline.py              # Full run
python3 overnight_pipeline.py --dry-run    # Show what would run
python3 overnight_pipeline.py --skip-phase 1  # Skip backtest phase
```

Features: skips stocks with recent results, catches exceptions per-stock (won't crash if one stock fails), prints progress per phase, Fisher combined p-value across all tested stocks.

## Dashboard

Flask-based web dashboard at `http://127.0.0.1:5000` with auto-refresh every 30 seconds:
- Portfolio value, cash, total return, alpha vs S&P 500
- Open positions with P&L and LONG/SHORT labels
- Recent trade log with color-coded actions (BUY, SELL, SHORT, COVER)

## Setup

### Prerequisites
- Python 3.x
- Alpaca paper trading account
- NewsAPI key (free tier)

### Installation

```bash
pip install alpaca-trade-api pandas numpy python-dotenv flask newsapi-python vaderSentiment matplotlib schedule scipy
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

# Permutation test (statistical validation, ~30-45 min for all stocks)
python3 permutation_test.py

# Quick test on specific stocks
python3 permutation_test.py --stocks META,GS --perms 100

# Run overnight pipeline (all phases, ~2-4 hours)
python3 overnight_pipeline.py

# Dry run (see what would run without executing)
python3 overnight_pipeline.py --dry-run
```

## Project Journey

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

### Phase 7: Signal Relaxation & Full Expansion
Relaxed MACD + VWAP from AND to OR logic. Extended data window from 30 to 60 days, widened RSI grid, reduced trade cooldown from 5 to 2 bars. Expanded optimizer to all 26 stocks with **per-stock JSON output**. Results: **9/26 stocks PASS** (up from 1/10).

### Phase 8: Statistical Validation (Permutation Testing)
Built a **permutation test module** to determine if strategy performance is real or data-mining luck. Uses day-block shuffling (1,000 permutations), Bonferroni correction for 26 stocks, and Fisher's method to combine p-values.

### Phase 9: Enhanced Validation & Tiered Trading
Extended data windows (90-day lookback, 60-day train, 30-day test). Added dual-method permutation test, White's Reality Check, simple signal mode, Sharpe cap, and tiered summary.

### Phase 10: Overnight Pipeline & Live Trading Prep
Built `overnight_pipeline.py`. First live paper trading day: 0 trades fired — market in sustained downtrend (tariff sell-off), SPY regime filter blocked all buys. Bot correctly preserved capital (+0.03%).

### Phase 11: Weighted Scoring, Short Selling & Universe Expansion
Three major upgrades to address Day 1's 0-trade problem:
- **Weighted signal scoring** — replaced strict AND conditions with weighted scores. Regime filter changed from hard gate to 2.5-point soft signal. Shared `signals.py` module eliminates duplication across 4 files.
- **Short selling** — bot can now short eligible stocks in downtrends with full risk management (ATR stops, hard 5% cap, regime-switch auto-cover, separate position/loss limits).
- **Expanded universe** — 18 active stocks across 3 tiers (up from 12 across 2 tiers). Added Tier 3 at 25% sizing for monitoring stocks. 6 pipeline candidates for future testing.

### Next Steps
- Live paper trading with scoring + shorts (validate real-time performance)
- Run overnight pipeline with expanded universe
- Tune scoring thresholds based on live results
- Permutation test on short signals
- VIX regime filter (reduce size in high-volatility markets)

## Results & Output Files

```
backtest_results/           <- Per-stock backtest results (JSON)
walk_forward_results/       <- Per-stock optimized parameters (JSON)
permutation_test_results/   <- Statistical validation (JSON + PNG charts)
  summary.png               <- All stocks' p-values at a glance
  {STOCK}_day_shuffle.png   <- Day-shuffle Sharpe distribution vs actual
  {STOCK}_signal_shift.png  <- Signal-shift Sharpe distribution vs actual
  combined.json             <- Fisher combined p-value and all results
results/                    <- Overnight pipeline output
  overnight_summary.txt     <- Master report with tiered recommendations
  overnight_summary.json    <- Machine-readable summary
```

## Disclaimer

This is a **paper trading** project for educational purposes. Past backtest performance does not guarantee future results. Do not use this bot with real money without extensive additional testing and risk management.
