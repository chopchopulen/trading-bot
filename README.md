# Algorithmic Trading Bot

A fully automated intraday trading bot running on **Alpaca Paper Trading** ($100K portfolio). Built from scratch as a learning project exploring quantitative finance, technical analysis, and systematic trading.

## What It Does

The bot trades **26 stocks** across 7 sectors on 1-minute bars during market hours (9:45 AM - 3:45 PM ET). It combines multiple technical indicators with sentiment analysis to generate buy/sell signals, manages risk with ATR-based position sizing and dynamic stop losses, and serves a real-time dashboard.

## Architecture

```
bot.py                ← Live trading engine (runs during market hours)
dashboard.py          ← Flask web dashboard (portfolio, positions, trades)
walk_forward.py       ← Walk-forward parameter optimizer (3,312 combos/stock)
permutation_test.py   ← Statistical significance testing (day-block shuffle)
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
- **Train:** 40 days of 1-minute bars — find best parameters from 3,312 combinations
- **Test:** Next 20 days — validate on unseen data
- **Pass criteria:** Test return > 0%, trades >= 2, Sharpe > 1.0
- **Parameter grid:** EMA fast/slow (24 combos), RSI buy/sell thresholds, BB period/std
- **All 26 stocks** tested with per-stock JSON output to `walk_forward_results/`
- **Results:** 9/26 stocks PASS, all 26 generate trades

Includes a transaction cost model (per-stock spread estimates, slippage, SEC fees) for realistic results.

## Permutation Test (Statistical Validation)

`permutation_test.py` answers: **"Is this strategy's performance real or just luck?"**

- **Method:** Day-block shuffle — randomly reorder entire trading days (1,000 permutations per stock), keeping intraday price structure intact but destroying inter-day trends
- **Metric:** Sharpe ratio — if the real Sharpe beats 99%+ of permuted Sharpes, the edge is real
- **Multiple testing:** Bonferroni correction (alpha = 0.05/26 = 0.00192) to account for testing 26 stocks
- **Aggregation:** Fisher's method combines per-stock p-values into a single portfolio-level number

### Latest Results

| Verdict | Stocks |
|---------|--------|
| **Marginal** (p < 0.05) | META (p=0.010), GS (p=0.011) |
| **Promising** (p < 0.20) | COST, WMT, NFLX, TSLA, MSFT |
| **No signal** (p > 0.30) | 15 other stocks |

**Fisher combined p-value: 0.071** — the strategy shows suggestive but not yet statistically significant edge at the portfolio level. META and GS show the strongest individual signals.

Output: per-stock histograms, summary chart, and JSON results in `permutation_test_results/`.

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
Built a **permutation test module** to determine if strategy performance is real or data-mining luck. Uses day-block shuffling (1,000 permutations), Bonferroni correction for 26 stocks, and Fisher's method to combine p-values. Results: **Fisher combined p = 0.071** — suggestive but not yet significant. META and GS showed strongest individual signals (p ~ 0.01). Key finding: most "PASS" stocks had too few trades (2-5) for statistical significance on a 20-day test window.

### Phase 9: Next Steps
- Extend test window to 30 days for more trades per stock
- Trim universe to statistically promising stocks (META, GS, WMT, COST, TSLA, MSFT, NFLX)
- Test simpler signal chain (EMA + RSI only) to increase trade frequency
- VIX regime filter (reduce size in high-volatility markets)
- Pairs trading (statistical arbitrage between correlated stocks)

## Results & Output Files

```
walk_forward_results/       ← Per-stock optimized parameters (JSON)
permutation_test_results/   ← Statistical validation (JSON + PNG charts)
  summary.png               ← All stocks' p-values at a glance
  {STOCK}_histogram.png     ← Per-stock Sharpe distribution vs actual
  combined.json             ← Fisher combined p-value and all results
```

## Disclaimer

This is a **paper trading** project for educational purposes. Past backtest performance does not guarantee future results. Do not use this bot with real money without extensive additional testing and risk management.
