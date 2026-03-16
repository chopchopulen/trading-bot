# Algorithmic Trading Bot

A fully automated intraday trading bot running on **Alpaca Paper Trading** ($100K portfolio). Built from scratch as a learning project exploring quantitative finance, technical analysis, and systematic trading.

## What It Does

The bot trades a **tiered stock universe** (12 validated stocks across 2 tiers) on 1-minute bars during market hours (9:45 AM - 3:45 PM ET). It combines multiple technical indicators with sentiment analysis to generate buy/sell signals, manages risk with ATR-based position sizing and dynamic stop losses, and serves a real-time dashboard. Stocks are validated through a 3-stage pipeline: backtest → walk-forward optimization → permutation test.

## Architecture

```
bot.py                ← Live trading engine (tiered stocks, ATR sizing)
overnight_pipeline.py ← Master overnight test runner (4 phases)
dashboard.py          ← Flask web dashboard (portfolio, positions, trades)
walk_forward.py       ← Walk-forward parameter optimizer (3,312 combos/stock)
permutation_test.py   ← Statistical significance testing (dual methods)
backtest_minutes.py   ← Minute-bar backtester with charting
backtest.py           ← Daily-bar backtester with grid search
diagnose_signals.py   ← Signal diagnostics & debugging tool
```

## Signal Chain

Every buy signal must pass **all** of these filters:

1. **SPY Regime Filter** — SPY must be above its 50-period EMA (uptrend)
2. **EMA Crossover** — Fast EMA > Slow EMA (per-stock optimized)
3. **RSI OR Bollinger Bands** — RSI oversold **OR** price <= BB lower band
4. **MACD OR VWAP** — MACD bullish crossover **OR** price below VWAP

> **Design note:** RSI and BB use OR logic because both measure "oversold" conditions. MACD and VWAP use OR logic because requiring both simultaneously contradicts — VWAP wants price below fair value while MACD confirms upward momentum, which rarely coexist on 1-minute bars. Each pair requires at least one confirmation signal.

Sell signals mirror this with reversed conditions. Parameters are **per-stock optimized** via the walk-forward optimizer — the bot loads each stock's best EMA, RSI, and BB settings from `walk_forward_results/`.

## Risk Management

| Feature | Implementation |
|---------|---------------|
| **Position Sizing** | ATR-based: `qty = (portfolio * 1%) / ATR(14)` — Tier 1 full size, Tier 2 at 50% |
| **Stop Loss** | Dynamic: `entry - 1.5 * ATR(14)` — adapts to each stock's volatility |
| **Take Profit** | Dynamic: `entry + 3.0 * ATR(14)` — 1:2 risk/reward ratio |
| **Max Positions** | 5 concurrent positions |
| **Daily Loss Limit** | -$500 — stops all new buys if hit |
| **Market Window** | Only trades 9:45 AM - 3:45 PM ET (avoids open/close volatility) |
| **Regime Filter** | No buys when SPY < 50 EMA (bear market protection) |
| **Backtest Gate** | No buys if stock's backtest profit factor < 1.0 |

## Stock Universe (Tiered)

Stocks are assigned to tiers based on permutation test p-values and backtest profit factors:

| Tier | Sizing | Stocks | Criteria |
|------|--------|--------|----------|
| **Tier 1** | Full ATR size | META, SPY, AMD, GS | p < 0.05, proven edge |
| **Tier 2** | 50% ATR size | WMT, COST, MSFT, NFLX, QCOM, AAPL, CRM, TSLA | p < 0.15, promising |

The overnight pipeline can expand this universe by testing additional stocks (UBER, PLTR, COIN, SHOP, SQ, ROKU, ABNB, PYPL, SPOT, ZM, HOOD) and promoting them to the appropriate tier.

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
| **Strong** (p < 0.05) | META (p=0.012), GS (p=0.002), AMD (p=0.002), SPY (p=0.002) |
| **Promising** (p < 0.15) | WMT (p=0.072), COST (p=0.092), MSFT (p=0.026), NFLX (p=0.064) |
| **Weak** (p < 0.30) | AAPL, QCOM, CRM, TSLA, AMZN, NVDA |

Output: per-stock histograms (day shuffle + signal shift), summary chart, and JSON results in `permutation_test_results/`.

## Overnight Pipeline

`overnight_pipeline.py` runs all validation phases in sequence overnight (~2-4 hours):

| Phase | What | Output |
|-------|------|--------|
| **Phase 1** | Backtest all stocks (grid search, 10 param sets) | `backtest_results/{STOCK}.json` |
| **Phase 2** | Walk-forward on untested stocks (UBER, PLTR, COIN, etc.) | `walk_forward_results/{STOCK}.json` |
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
- Open positions with P&L
- Recent trade log

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

### Phase 7: Signal Relaxation & Full Expansion
Relaxed MACD + VWAP from AND to OR logic (same reasoning as RSI/BB — requiring both contradicts on 1-min bars). Extended data window from 30 to 60 days, widened RSI grid, reduced trade cooldown from 5 to 2 bars. Expanded optimizer to all 26 stocks with **per-stock JSON output** — bot now loads each stock's best parameters on startup. Results: **9/26 stocks PASS** (up from 1/10), all 26 generate trades (up from 5/10).

### Phase 8: Statistical Validation (Permutation Testing)
Built a **permutation test module** to determine if strategy performance is real or data-mining luck. Uses day-block shuffling (1,000 permutations), Bonferroni correction for 26 stocks, and Fisher's method to combine p-values. Initial results: **Fisher combined p = 0.071** — suggestive but not yet significant. META and GS showed strongest individual signals (p ~ 0.01). Key finding: most "PASS" stocks had too few trades (2-5) for statistical significance on a 20-day test window.

### Phase 9: Enhanced Validation & Tiered Trading
- Extended data windows: lookback 90 days, train 60 days, test 30 days
- Added **dual-method permutation test**: day-block shuffle + signal-shift test
- Fixed Sharpe explosion (cap at 0 when trades < 2)
- Added **White's Reality Check** adjustment for data mining bias
- Added **simple signal mode** flag (EMA + RSI only) for 10-20x more trades
- Tiered summary with actionable recommendations per stock
- Round 2 results: META p=0.012, GS p=0.002, AMD p=0.002, SPY p=0.002

### Phase 10: Overnight Pipeline & Live Trading Prep
- Built `overnight_pipeline.py` — runs backtest, walk-forward, and permutation test sequentially overnight
- **Tiered stock universe**: Tier 1 (META, SPY, AMD, GS) at full size, Tier 2 (8 stocks) at 50% size
- **Backtest quality gate**: bot skips stocks with profit factor < 1.0
- **Backtest results loader**: bot reads `backtest_results/` at startup for filtering
- **Expansion candidates**: 11 new stocks (UBER, PLTR, COIN, etc.) ready for pipeline testing

### Phase 11: Next Steps
- Live paper trading with tiered universe (validate real-time performance)
- Run overnight pipeline to test 11 new stocks and refresh all results
- VIX regime filter (reduce size in high-volatility markets)
- Pairs trading (statistical arbitrage between correlated stocks)
- Multi-timeframe confirmation (5-min + 1-min signals)

## Results & Output Files

```
backtest_results/           ← Per-stock backtest results (JSON)
walk_forward_results/       ← Per-stock optimized parameters (JSON)
permutation_test_results/   ← Statistical validation (JSON + PNG charts)
  summary.png               ← All stocks' p-values at a glance
  {STOCK}_day_shuffle.png   ← Day-shuffle Sharpe distribution vs actual
  {STOCK}_signal_shift.png  ← Signal-shift Sharpe distribution vs actual
  combined.json             ← Fisher combined p-value and all results
results/                    ← Overnight pipeline output
  overnight_summary.txt     ← Master report with tiered recommendations
  overnight_summary.json    ← Machine-readable summary
```

## Disclaimer

This is a **paper trading** project for educational purposes. Past backtest performance does not guarantee future results. Do not use this bot with real money without extensive additional testing and risk management.
