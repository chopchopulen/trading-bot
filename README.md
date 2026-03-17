# Algorithmic Trading Bot

A fully automated intraday trading bot running on **Alpaca Paper Trading** ($100K portfolio). Built from scratch as a learning project exploring quantitative finance, technical analysis, and systematic trading.

## What It Does

The bot trades a **tiered stock universe** on 1-minute bars during market hours (9:45 AM - 3:45 PM ET). It uses a **gradient signal scoring system** (v3) to generate buy, sell, short, and cover signals. Scores combine 3 base indicators (7.0 pts) with 3 contextual overlays (ORB, relative strength, VWAP bands) for a max of 12.5 points. Risk is managed with ATR-based position sizing and dynamic stop losses. Stocks are validated through a 3-stage pipeline: backtest → walk-forward optimization → permutation test.

## Architecture

```
signals.py            <- Shared signal module (gradient scoring v3, ORB, RS, VWAP bands)
bot.py                <- Live trading engine (gradient scoring, shorts, tiered stocks, pairs)
overnight_pipeline.py <- Master overnight test runner (4 phases, 53+ stocks)
dashboard.py          <- Flask web dashboard (portfolio, positions, trades)
walk_forward.py       <- Walk-forward parameter optimizer (~500 combos/stock)
permutation_test.py   <- Statistical significance testing (dual methods)
backtest_minutes.py   <- Minute-bar backtester with charting
backtest.py           <- Daily-bar backtester with grid search
diagnose_signals.py   <- Signal diagnostics & debugging tool
```

## Signal Scoring System (v3 — Gradient + Contextual Overlays)

The bot uses a **gradient scoring system** where each indicator contributes a continuous score based on signal strength, not binary on/off. v3 adds three contextual overlays on top of the v2 base.

### Base Indicators (max 7.0 pts)

| Signal | Max Score | How It Works |
|--------|-----------|-------------|
| EMA Trend | 3.0 | Gradient based on % separation between fast/slow EMA. Wider gap = higher score |
| RSI Oversold | 2.0 | Gradient based on depth below buy level. RSI at 15 scores higher than RSI at 29 |
| Bollinger %B | 2.0 | Gradient based on position within bands. Closer to lower band = higher score |

### Contextual Overlays (v3 additions, max +5.5 pts)

| Signal | Max Buy | Max Short | How It Works |
|--------|---------|-----------|-------------|
| Relative Strength vs SPY | +1.5 | +1.5 | 5-bar return vs SPY. Stock outperforming = buy signal; underperforming = short signal. Normalized: 0.2% diff = 1.0 score unit |
| Opening Range Breakout (ORB) | +2.0 | +2.0 | Price breaks above/below 9:30–9:45 AM ET range with 1.5× avg volume. Regime-independent entry |
| VWAP Bands | +1.5 | +2.0 | Price vs rolling VWAP ± 1/2 std dev bands. Below lower 2std + green candle = buy; above upper 2std + red candle = short |

**Max possible score: 12.5**

### Hard Gates (block entry, not scored)

| Gate | Condition | Effect |
|------|-----------|--------|
| Regime | SPY must be above 20-day EMA (daily bars) | Blocks all longs in downtrends |
| Volume | Current volume must be > 50% of 20-bar average | Blocks entries on thin volume |

### Entry Thresholds by Tier

| Tier | Buy Threshold | Sell Threshold | Short Threshold |
|------|--------------|----------------|-----------------|
| Tier 1 (proven) | 3.5 | 3.0 | 3.5 |
| Tier 2 (promising) | 4.0 | 3.5 | 4.0 |
| Tier 3 (monitoring) | 4.5 | 4.0 | disabled |

**Key design decisions:**
- Regime is a **hard gate**, not a soft signal — no longs in downtrends, period
- Short threshold is now equal to buy threshold (removed old +1.0 penalty — regime gate provides the necessary conservatism)
- ORB signals bypass the regime hard gate (a confirmed breakout is valid regardless of daily trend)
- All scoring logic lives in `signals.py` — single source of truth across all files

## Regime Detection (Daily 20-Day EMA)

The regime filter uses **SPY daily bars with a 20-day EMA** — not intraday. This flips at most once per day, compared to the old 50-bar intraday EMA that flipped dozens of times per day on noise.

```
SPY close > 20-day EMA on daily bars → UPTREND (longs allowed)
SPY close < 20-day EMA on daily bars → DOWNTREND (shorts allowed, longs blocked)
```

For backtests, daily regime is mapped to each minute bar by date.

## Opening Range Breakout (ORB)

The 9:30–9:45 AM ET opening range is computed once per stock per day and cached in `ORB_DATA`. After the range finalizes at 9:45 AM, any bar where price crosses above the high (with volume ≥ 1.5× avg) adds +2.0 to the buy score, regardless of daily regime. A breakdown below the range adds +2.0 to the short score.

- ORB requires at least 3 bars in the 15-minute window to be valid
- Implementation uses `America/New_York` timezone with DST-safe `pd.Timestamp` handling
- One ORB range per stock per calendar day; reprints "ORB finalized" to stdout once

## Relative Strength vs SPY

Every stock's 5-bar return is compared to SPY's 5-bar return. The difference is normalized to a ±2.0 score (0.2% spread = 1.0 point):

```python
score = (stock_5bar_return - spy_5bar_return) / 0.002
# capped at [-2.0, +2.0]
```

- RS > +0.5: stock outperforming → +1.5 to buy score, -1.0 penalty to short
- RS < -0.5: stock underperforming → +1.5 to short score, -1.0 penalty to buy
- SPY itself is excluded from RS scoring (no self-comparison)

## VWAP Bands

Rolling VWAP with ±1 and ±2 standard deviation bands computed over a 20-bar window:

```
upper_2std = vwap + 2 * std(typical_price - vwap)
lower_2std = vwap - 2 * std(typical_price - vwap)
```

| Position | Buy Score | Short Score |
|----------|-----------|-------------|
| Below lower 2std + green candle | +1.5 | — |
| Below lower 1std | +0.5 | — |
| Above upper 2std + red candle | — | +2.0 |
| Above upper 1std | — | +1.0 |
| Below VWAP (for shorts) | — | -1.0 |

## Short Selling

When the market is in a **downtrend** (SPY < 20-day EMA on daily), the bot can short highly liquid stocks:

```
SHORT_ELIGIBLE = NVDA, AMD, AMZN, AAPL, MSFT, META, TSLA, NFLX, GS
```

WMT and COST are explicitly excluded — defensive names that may rise in bear markets.

Short signals use the same gradient scoring system (inverted: EMA bearish, RSI overbought, BB upper) plus the v3 overlays.

### Short Risk Guardrails

| Feature | Setting |
|---------|---------|
| Max short positions | 3 |
| Short daily loss limit | -$300 |
| Hard stop | 5% above entry (force close) |
| ATR stop loss | Entry + 1.5x ATR |
| ATR take profit | Entry - 3.0x ATR |
| Regime switch | Auto-cover all shorts if SPY flips to uptrend |

## Pairs Tracking (Diagnostic)

The bot tracks two correlated pairs using a z-score on the price ratio spread:

```python
PAIRS = {"SEMI": ("AMD", "NVDA"), "MEGACAP": ("MSFT", "AAPL")}
```

- Uses 1.5-hour bar lookback, aligned by index
- When |z-score| ≥ 2.0, prints a directional signal (e.g., "LONG AMD / SHORT NVDA")
- **Diagnostic only** — no auto-trading. Runs at end of each `run_bot()` cycle when `DIAGNOSTIC_MODE = True`

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
| **Phase 1** | Backtest all original stocks (grid search) + supplemental backtest for walk-forward passers lacking results | `backtest_results/{STOCK}.json` |
| **Phase 2** | Walk-forward on NEW_STOCKS + all PIPELINE_STOCKS (17 candidates) | `walk_forward_results/{STOCK}.json` |
| **Phase 3** | Permutation test on profitable stocks (500 perms, FORCE_REPERM bypasses stale cache) | `permutation_test_results/{STOCK}.json` |
| **Phase 4** | Master summary report with tiered recommendations | `results/overnight_summary.txt` |

```bash
python3 overnight_pipeline.py              # Full run
python3 overnight_pipeline.py --dry-run    # Show what would run
python3 overnight_pipeline.py --skip-phase 1  # Skip backtest phase
```

**Pipeline improvements:**
- `FORCE_REPERM = True` — re-runs all permutation tests, ignoring stale results from before gradient scoring overhaul
- `WALK_FORWARD_STOCKS = NEW_STOCKS + PIPELINE_STOCKS` — 17 pipeline candidates now run walk-forward alongside new stocks
- Supplemental backtest block after Phase 2 — auto-backtests any walk-forward passer that lacks a backtest result

## Diagnostic & Calibration Modes

### DIAGNOSTIC_MODE

When `DIAGNOSTIC_MODE = True`, prints a full per-stock per-minute breakdown:

```
DIAG AMZN [DOWNTREND] SHORT | EMA:2.1/3 RSI:0.0/2 BB:0.0/2 RS:1.5[weak -0.82] ORB:2.0[below_low($185.40)] VWAP:1.0[>+1std] | Total:6.6/12.5 Thresh:4.5 | eligible | >>> FIRE <<<
```

Shows: regime, all 6 scoring components with labels (strong/neutral/weak, above_high/inside/below_low, >±1std/>±2std/inside_bands), total vs max score, gap to threshold.

### TEST_MODE

When `TEST_MODE = True`, subtracts 1.0 from all entry/exit thresholds (floored at 0.5). Used for calibration — verifies that signals flow end-to-end before going live. **Not for live trading.**

## Sentiment Analysis

The bot uses **NewsAPI + VADER** to score each stock's news sentiment for pre-market scanning:
- Fetches recent headlines for each stock
- Scores each headline with VADER compound sentiment
- Time-weights scores (recent news counts more)
- Caches results to `sentiment_cache.json` with today's date — bot restarts reuse the same scores and don't burn API quota
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
# Live trading bot (single scan + one cycle)
python3 bot.py

# Live trading bot (continuous loop — checks every minute)
python3 bot.py --trade

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

### Phase 13: Contextual Overlays, ORB & Short Calibration

**The 0-trade problem:** In sustained downtrends, the old short scoring was fundamentally contradictory — it required RSI overbought + BB upper band, but in a real downtrend RSI is oversold and price is near the lower band. Max achievable short score was ~3.0/7.0, far below the 5.0-6.0 thresholds (+1.0 penalty on top of already-high buy thresholds).

**Fixes and additions:**
- **Gradient scoring v3** — added three contextual overlays (ORB, Relative Strength vs SPY, VWAP Bands). Max score raised from 7.0 to 12.5
- **ORB (Opening Range Breakout)** — regime-independent entries based on 9:30–9:45 AM ET breakouts/breakdowns. Adds up to ±2.0 pts. DST-safe UTC-aware implementation
- **Relative Strength vs SPY** — 5-bar return differential normalized to ±2.0 pts. Strong RS adds to buy score, weak RS adds to short score
- **VWAP Bands** — rolling VWAP ± 1/2 std dev. Price below lower 2std + green candle = +1.5 buy; above upper 2std + red candle = +2.0 short
- **Removed short_threshold +1.0 penalty** — the regime hard gate already ensures shorts only fire in downtrends; stacking a +1.0 penalty was preventing all short trades
- **Sentiment cache file** — `sentiment_cache.json` stores today's scores so bot restarts don't burn API quota (was exhausting the 100-request developer limit on multiple restarts)
- **DIAGNOSTIC_MODE + TEST_MODE** — per-stock per-minute full score breakdown; TEST_MODE lowers all thresholds by 1.0 for calibration
- **Pipeline: FORCE_REPERM + PIPELINE_STOCKS in walk-forward** — ensures all 17 pipeline candidates get walk-forward optimization and all permutation results are re-run fresh (not reusing stale MACD/VWAP era results)
- **Pairs tracking** — z-score monitoring for SEMI (AMD/NVDA) and MEGACAP (MSFT/AAPL) pairs; diagnostic only, no auto-trading
- **Fixed `is_safe_trading_window()` nested function bug** — inner `def` shadowed the outer function, causing `return True` to always execute without checking the time window
- **Fixed `--trade` flag** — now loops continuously with `schedule` rather than firing once

### Next Steps
- Run overnight pipeline with v3 scoring — validate ORB/RS/VWAP improve profit factors
- Regime-specific walk-forward optimization (bull params for longs, bear params for shorts)
- Slippage sensitivity analysis to find break-even cost levels
- Monitor paper trading performance with DIAGNOSTIC_MODE enabled

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
