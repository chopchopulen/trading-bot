# Algorithmic Trading Bot

A fully automated intraday trading bot running on **Alpaca Paper Trading** ($100K portfolio). Built from scratch as a learning project exploring quantitative finance, technical analysis, and systematic trading.

## What It Does

The bot trades a **tiered stock universe** on 1-minute bars during market hours (9:45 AM - 3:45 PM ET). It uses a **gradient signal scoring system** (v2) to generate buy, sell, short, and cover signals with 3 scored indicators and 2 hard gates. Risk is managed with ATR-based position sizing and dynamic stop losses. Stocks are validated through a 3-stage pipeline: backtest -> walk-forward optimization -> permutation test.

## Architecture

```
signals.py            <- Shared signal module (gradient scoring v2, indicators, costs)
bot.py                <- Live trading engine (gradient scoring, shorts, tiered stocks)
overnight_pipeline.py <- Master overnight test runner (4 phases, 53 stocks)
dashboard.py          <- Flask web dashboard (portfolio, positions, trades)
walk_forward.py       <- Walk-forward parameter optimizer (~500 combos/stock)
permutation_test.py   <- Statistical significance testing (dual methods)
backtest_minutes.py   <- Minute-bar backtester with charting
backtest.py           <- Daily-bar backtester with grid search
diagnose_signals.py   <- Signal diagnostics & debugging tool
```

## Signal Scoring System (v2 — Gradient)

The bot uses a **gradient scoring system** where each indicator contributes a continuous score based on signal strength, not binary on/off. This replaced the original 8-indicator binary system that was producing a 40% win rate.

### Scored Indicators

| Signal | Max Score | How It Works |
|--------|-----------|-------------|
| EMA Trend | 3.0 | Gradient based on % separation between fast/slow EMA. Wider gap = higher score |
| RSI Oversold | 2.0 | Gradient based on depth below buy level. RSI at 15 scores higher than RSI at 29 |
| Bollinger %B | 2.0 | Gradient based on position within bands. Closer to lower band = higher score |

**Max possible score: 7.0**

### Hard Gates (block entry, not scored)

| Gate | Condition | Effect |
|------|-----------|--------|
| Regime | SPY must be above 20-day EMA (daily bars) | Blocks all longs in downtrends |
| Volume | Current volume must be > 50% of 20-bar average | Blocks entries on thin volume |

### Buy Thresholds by Tier

| Tier | Buy Threshold | Sell Threshold | Short Threshold |
|------|--------------|----------------|-----------------|
| Tier 1 (proven) | 3.5 | 3.0 | 4.5 |
| Tier 2 (promising) | 4.0 | 3.5 | 5.0 |
| Tier 3 (monitoring) | 4.5 | 4.0 | disabled |

**Key design decisions:**
- Regime is a **hard gate**, not a soft signal — no longs in downtrends, period
- Dropped MACD (too slow on 1-min bars), VWAP (weak signal), Sentiment (unreliable)
- Gradient scoring means strong confluence naturally scores high without threshold gaming
- All scoring logic lives in `signals.py` — single source of truth across all files

## Regime Detection (Daily 20-Day EMA)

The regime filter uses **SPY daily bars with a 20-day EMA** — not intraday. This flips at most once per day, compared to the old 50-bar intraday EMA that flipped dozens of times per day on noise.

```
SPY close > 20-day EMA on daily bars → UPTREND (longs allowed)
SPY close < 20-day EMA on daily bars → DOWNTREND (shorts allowed, longs blocked)
```

For backtests, daily regime is mapped to each minute bar by date.

## Short Selling

When the market is in a **downtrend** (SPY < 20-day EMA on daily), the bot can short highly liquid stocks:

```
SHORT_ELIGIBLE = AAPL, MSFT, META, AMD, TSLA, NFLX, NVDA, AMZN, GS
```

Short signals use the same gradient scoring system (inverted: EMA bearish, RSI overbought, BB upper).

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
| **Min Hold Time** | 5 bars (5 minutes) — prevents whipsaw exits |
| **Max Positions** | 5 concurrent (longs) + 3 concurrent (shorts) |
| **Daily Loss Limit** | -$500 (longs), -$300 (shorts) |
| **Market Window** | Only trades 9:45 AM - 3:45 PM ET |
| **Backtest Gate** | No buys if stock's backtest profit factor < 1.0 |
| **Sector Limits** | Max 2 positions per sector |

### Sector Diversification

```
tech:     MSFT, META, AAPL, AMZN, NFLX, CRM
semi:     AMD, NVDA, QCOM
finance:  GS, COIN, HOOD
consumer: WMT, COST, TSLA, UBER
etf:      SPY
other:    SPOT
```

## Stock Universe

Stocks are assigned to tiers based on permutation test p-values, backtest performance, and walk-forward results:

| Tier | Sizing | Criteria |
|------|--------|----------|
| **Tier 1** | Full ATR | p < 0.05, PF >= 1.5, WR >= 45%, 5-100 trades |
| **Tier 2** | 50% ATR | p < 0.15, PF >= 1.3, WR >= 42%, trades >= 5 |
| **Tier 3** | 25% ATR | p < 0.30, PF >= 1.1, return > 0 |

**Pipeline stocks** (53 total tested across original 26, 10 new, and 17 pipeline candidates).

The overnight pipeline auto-classifies stocks into tiers based on p-values and profit factors.

## Walk-Forward Optimizer

`walk_forward.py` prevents overfitting by splitting data into training and test windows:
- **Lookback:** 90 days of 1-minute bars
- **Train:** 60 days — find best parameters from ~500 combinations
- **Test:** 30 days — validate on unseen data
- **Regime:** Daily SPY 20-day EMA mapped to minute bars
- **Pass criteria:** Test return > 0%, trades >= 2, Sharpe > 1.0

### Quality Gates (strict)

| Gate | Threshold |
|------|-----------|
| Minimum trades | 15 |
| Maximum trades | 80 |
| Minimum profit factor | 1.5 |
| Minimum win rate | 48% |
| Minimum avg $/trade | $0.50 |

### Parameter Grid (~500 combos)

```
EMA fast:         [3, 5, 8]
EMA slow:         [12, 18, 25]
RSI buy:          [25, 30, 35]
RSI sell:         [65, 70, 75]
BB period:        [15, 20]
BB std:           [1.5, 2.0]
Score threshold:  [3.5, 4.0, 4.5, 5.0]
```

Includes a transaction cost model (per-stock spread estimates, slippage, SEC fees) for realistic results.

## Permutation Test (Statistical Validation)

`permutation_test.py` answers: **"Is this strategy's performance real or just luck?"**

**Two test methods:**
1. **Day-block shuffle** — randomly reorder entire trading days, keeping intraday bars intact but destroying inter-day trends
2. **Signal shift** — randomly offset entry signal indices and simulate ATR stop/target outcomes at shifted locations (tests whether entry timing matters)

**Statistical framework:**
- **500+ permutations** per stock
- **Bonferroni correction** for multiple comparisons
- **Fisher's method** combines per-stock p-values into a single portfolio-level number
- **White's Reality Check** adjustment for data mining bias

## Overnight Pipeline

`overnight_pipeline.py` runs all validation phases in sequence overnight (~2-4 hours):

| Phase | What | Output |
|-------|------|--------|
| **Phase 1** | Backtest all 26 original stocks (grid search, 10 param sets) + run walk-forward first if needed | `backtest_results/{STOCK}.json` |
| **Phase 2** | Walk-forward on 10 new stocks | `walk_forward_results/{STOCK}.json` |
| **Phase 3** | Permutation test on profitable stocks (500 perms) | `permutation_test_results/{STOCK}.json` |
| **Phase 4** | Master summary report with tiered recommendations | `results/overnight_summary.txt` |

```bash
python3 overnight_pipeline.py              # Full run
python3 overnight_pipeline.py --dry-run    # Show what would run
python3 overnight_pipeline.py --skip-phase 1  # Skip backtest phase
```

Features: per-stock optimal thresholds from walk-forward, daily regime detection, skips stocks with recent results, catches exceptions per-stock, Fisher combined p-value.

## Sentiment Analysis

The bot uses **NewsAPI + VADER** to score each stock's news sentiment for pre-market scanning:
- Fetches recent headlines for each stock
- Scores each headline with VADER compound sentiment
- Time-weights scores (recent news counts more)
- Caches results for 30 minutes to stay within API limits
- Pre-market scan at 4 AM ET identifies high-sentiment opportunities

Note: Sentiment is used for pre-market watchlist filtering only — it was removed from the signal scoring system (v2) due to unreliability.

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

# Permutation test (statistical validation)
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
Built a walk-forward optimizer to find optimal parameters without overfitting. Discovered that 6 AND conditions on 1-minute bars produce zero signals — **the progressive AND elimination problem**. Fixed by switching RSI and Bollinger Bands from AND to OR logic.

### Phase 7: Signal Relaxation & Full Expansion
Relaxed MACD + VWAP from AND to OR logic. Extended data window from 30 to 60 days, widened RSI grid, reduced trade cooldown. Expanded optimizer to all 26 stocks. Results: **9/26 stocks PASS** (up from 1/10).

### Phase 8: Statistical Validation (Permutation Testing)
Built a **permutation test module** to determine if strategy performance is real or data-mining luck. Uses day-block shuffling (1,000 permutations), Bonferroni correction for 26 stocks, and Fisher's method to combine p-values.

### Phase 9: Enhanced Validation & Tiered Trading
Extended data windows (90-day lookback, 60-day train, 30-day test). Added dual-method permutation test, White's Reality Check, simple signal mode, Sharpe cap, and tiered summary.

### Phase 10: Overnight Pipeline & Live Trading Prep
Built `overnight_pipeline.py`. First live paper trading day: 0 trades fired — market in sustained downtrend (tariff sell-off), SPY regime filter blocked all buys. Bot correctly preserved capital (+0.03%).

### Phase 11: Weighted Scoring, Short Selling & Universe Expansion
Replaced strict AND conditions with weighted scores. Added short selling with full risk management. Expanded universe to 18 active stocks across 3 tiers. Added 6 pipeline candidates.

### Phase 12: Gradient Scoring Overhaul & Strategy Diagnosis
Diagnosed the strategy as **fundamentally broken** after overnight pipeline revealed:
- Mean win rate: 40%, mean profit factor: 0.81, mean Sharpe: -1.36
- Only 4/26 stocks profitable (COST, XOM, CVX, JNJ — all marginal)
- 250-350 trades per 90 days per stock — too many, mostly noise
- 14 stocks had p < 0.05 but were **losing** — reliably worse than random

**Root cause:** 8 binary indicators with threshold gaming + regime filter on 1-min bars flipping constantly + MACD too slow for 1-min.

**Fixes implemented:**
- **Gradient scoring (v2)** — replaced 8 binary indicators with 3 gradient + 2 hard gates
- **Daily regime filter** — 20-day EMA on daily SPY bars (flips once/day, not dozens of times)
- **Dropped MACD, VWAP, Sentiment** from scoring entirely
- **Tighter quality gates** — min 15 trades, max 80, PF >= 1.5, WR >= 48%, avg $/trade >= $0.50
- **Smaller parameter grid** — ~500 combos (was 7,200), prevents optimizer from finding noise
- **Minimum hold time** — 5 bars to prevent whipsaw exits

### Next Steps
- Validate gradient scoring with full overnight pipeline run
- Opening Range Breakout (ORB) strategy as secondary signal
- Pairs trading (market-neutral) for cointegrated stock pairs
- Slippage sensitivity analysis to find break-even cost levels
- Paper trade validated stocks and monitor live performance

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
