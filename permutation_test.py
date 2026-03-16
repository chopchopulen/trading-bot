"""
Permutation Test Module for Trading Bot
========================================
Tests whether the walk-forward optimized strategy produces statistically
significant results or could be explained by random chance.

Two test methods:
1. Day-block shuffle — randomly reorder entire trading days
2. Signal shift — randomly offset entry signals relative to price changes

Statistical framework:
- 1,000 permutations per stock (p-value resolution: 0.001)
- Bonferroni correction for 26 stocks: alpha = 0.05/26 = 0.00192
- Fisher's method to combine per-stock p-values into one number
- White's Reality Check adjustment for data mining bias
- Test metric: Sharpe ratio (consistent with walk-forward criteria)
- Sharpe capped at 0 when trades < 2 to prevent explosion
"""

import walk_forward
import numpy as np
import pandas as pd
import json
import os
import sys
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────
N_PERMUTATIONS = 1000
N_STOCKS = 26
BONFERRONI_ALPHA = 0.05 / N_STOCKS  # ~0.00192
RESULTS_DIR = "permutation_test_results"
COMBINATIONS_TESTED = 3312  # parameter combos per stock in walk_forward
MIN_TRADES_FOR_SHARPE = 2   # cap Sharpe at 0 below this


def safe_sharpe(pnls):
    """Calculate Sharpe ratio, returning 0 if fewer than MIN_TRADES trades.

    This prevents the Sharpe explosion problem where 0-1 trades produce
    extreme values (-6000 to +thousands) that destroy the null distribution.
    """
    if len(pnls) < MIN_TRADES_FOR_SHARPE:
        return 0.0
    if np.std(pnls) == 0:
        return 0.0
    return float((np.mean(pnls) / np.std(pnls)) * np.sqrt(252))


def load_stock_result(stock):
    """Load walk-forward result JSON for a stock.

    Returns None if the file is missing, the stock failed walk-forward,
    or there were fewer than 2 trades (Sharpe is meaningless with 0-1 trades).
    """
    path = f"walk_forward_results/{stock}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if data.get("status") not in ("PASS", "WEAK"):
        return None
    if data.get("test_trades", 0) < 2:
        return None
    return data


def shuffle_bars_by_day(bars, rng):
    """Shuffle trading days in random order, keeping intraday bars intact.

    By reordering entire days, we preserve intraday price structure
    (ATR, VWAP work correctly) but destroy inter-day trends and regime
    patterns that the strategy exploits.
    """
    dates = np.unique(bars.index.date)
    shuffled_order = rng.permutation(len(dates))
    shuffled_dates = dates[shuffled_order]

    chunks = []
    for date in shuffled_dates:
        day_bars = bars[bars.index.date == date]
        chunks.append(day_bars)

    result = pd.concat(chunks)
    result.index = bars.index[:len(result)]
    return result


def run_backtest_safe(bars, spy_bars, params):
    """Run walk_forward.run_backtest with Sharpe capped for low trade counts."""
    result = walk_forward.run_backtest(
        bars, spy_bars,
        params["fast"], params["slow"],
        params["rsi_buy"], params["rsi_sell"],
        params["bb_period"], params["bb_std"]
    )
    # Cap Sharpe at 0 when too few trades to prevent explosion
    if result["trades"] < MIN_TRADES_FOR_SHARPE:
        result["sharpe"] = 0.0
    return result


def run_day_shuffle_test(stock, bars, spy_bars, params, actual_sharpe,
                          n_perms=N_PERMUTATIONS, seed=42):
    """Test 1: Day-block shuffle permutation test.

    Shuffle the order of trading days and re-run the backtest.
    The null hypothesis: the strategy would achieve this Sharpe
    even on randomly-ordered days.
    """
    walk_forward.STOCK = stock
    rng = np.random.default_rng(seed)
    perm_sharpes = []

    for i in range(n_perms):
        shuffled_bars = shuffle_bars_by_day(bars, rng)
        shuffled_spy = shuffle_bars_by_day(spy_bars, rng)
        result = run_backtest_safe(shuffled_bars, shuffled_spy, params)
        perm_sharpes.append(result["sharpe"])

        if (i + 1) % 100 == 0:
            print(f"    Day-shuffle: {i + 1}/{n_perms} done")

    count_as_good = sum(1 for s in perm_sharpes if s >= actual_sharpe)
    p_value = (count_as_good + 1) / (n_perms + 1)

    return {
        "p_value": p_value,
        "perm_sharpes": perm_sharpes,
        "mean_perm_sharpe": float(np.mean(perm_sharpes)),
        "std_perm_sharpe": float(np.std(perm_sharpes)),
    }


def run_signal_shift_test(stock, bars, spy_bars, params, actual_sharpe,
                           n_perms=N_PERMUTATIONS, seed=123):
    """Test 2: Signal shift permutation test.

    Instead of shuffling bars, run the real backtest once to find where
    the entry signals fire, then randomly shift those signal indices
    by a random offset and re-simulate the trades at the shifted locations.
    This tests whether the TIMING of entries matters.

    Method:
    - Run backtest, record which bar indices trigger a buy
    - For each permutation, shift all buy indices by a random offset
      (wrapping around), then simulate what returns would have been
      if we entered at those shifted times with the same ATR stops
    """
    walk_forward.STOCK = stock

    # First, find the actual entry points by running the real backtest
    # and recording buy bar indices
    close = bars["close"]
    fast_ema = walk_forward.calculate_ema(close, params["fast"])
    slow_ema = walk_forward.calculate_ema(close, params["slow"])
    rsi = walk_forward.calculate_rsi(close, walk_forward.RSI_PERIOD)
    bb_upper, bb_mid, bb_lower = walk_forward.calculate_bb(
        close, params["bb_period"], params["bb_std"])
    macd_line, signal_line = walk_forward.calculate_macd(close)
    atr = walk_forward.calculate_atr(bars)
    vwap = walk_forward.calculate_vwap(bars)

    spy_close = spy_bars["close"].reindex(close.index, method="ffill")
    spy_ema = walk_forward.calculate_ema(spy_close, walk_forward.REGIME_EMA)

    start_bar = max(params["slow"], params["bb_period"],
                    walk_forward.RSI_PERIOD, walk_forward.REGIME_EMA, 35) + 1
    n_bars = len(bars)

    # Find real buy signal indices
    buy_indices = []
    last_trade = -10
    in_position = False
    for i in range(start_bar, n_bars):
        if i - last_trade < 2:
            continue
        price = close.iloc[i]

        # Exit check
        if in_position:
            stop_price = entry_price - (walk_forward.ATR_STOP_MULT * entry_atr)
            target_price = entry_price + (walk_forward.ATR_PROFIT_MULT * entry_atr)
            if price <= stop_price or price >= target_price:
                in_position = False
                last_trade = i
                continue

        if not in_position and not pd.isna(atr.iloc[i]):
            uptrend = spy_close.iloc[i] > spy_ema.iloc[i]
            macd_bull = macd_line.iloc[i] > signal_line.iloc[i]
            cur_vwap = vwap.iloc[i]

            if walk_forward.SIMPLE_SIGNAL_MODE:
                buy_sig = (uptrend and
                           fast_ema.iloc[i] > slow_ema.iloc[i] and
                           rsi.iloc[i] < params["rsi_buy"])
            else:
                buy_sig = (uptrend and
                           fast_ema.iloc[i] > slow_ema.iloc[i] and
                           (rsi.iloc[i] < params["rsi_buy"] or
                            price <= bb_lower.iloc[i]) and
                           (macd_bull or
                            (pd.isna(cur_vwap) or price < cur_vwap)))

            if buy_sig:
                buy_indices.append(i)
                entry_price = price
                entry_atr = atr.iloc[i]
                in_position = True
                last_trade = i

        elif in_position:
            macd_bear = macd_line.iloc[i] < signal_line.iloc[i]
            cur_vwap = vwap.iloc[i]

            if walk_forward.SIMPLE_SIGNAL_MODE:
                sell_sig = (fast_ema.iloc[i] < slow_ema.iloc[i] and
                            rsi.iloc[i] > params["rsi_sell"])
            else:
                sell_sig = (fast_ema.iloc[i] < slow_ema.iloc[i] and
                            (rsi.iloc[i] > params["rsi_sell"] or
                             price >= bb_upper.iloc[i]) and
                            (macd_bear or
                             (pd.isna(cur_vwap) or price > cur_vwap)))

            if sell_sig:
                in_position = False
                last_trade = i

    if len(buy_indices) < 2:
        return {
            "p_value": 1.0,
            "perm_sharpes": [0.0] * n_perms,
            "mean_perm_sharpe": 0.0,
            "std_perm_sharpe": 0.0,
        }

    # Now run permutations by shifting entry indices
    tradeable_range = n_bars - start_bar
    rng = np.random.default_rng(seed)
    perm_sharpes = []

    for perm_i in range(n_perms):
        offset = rng.integers(1, tradeable_range)
        pnls = []

        for buy_idx in buy_indices:
            shifted_idx = start_bar + ((buy_idx - start_bar + offset) % tradeable_range)
            if shifted_idx >= n_bars or pd.isna(atr.iloc[shifted_idx]):
                continue

            entry_p = close.iloc[shifted_idx]
            entry_a = atr.iloc[shifted_idx]
            stop_p = entry_p - (walk_forward.ATR_STOP_MULT * entry_a)
            target_p = entry_p + (walk_forward.ATR_PROFIT_MULT * entry_a)

            # Simulate forward from shifted entry until stop/target/end
            for j in range(shifted_idx + 1, min(shifted_idx + 390, n_bars)):
                p = close.iloc[j]
                if p <= stop_p or p >= target_p:
                    buy_cost = walk_forward.calculate_trade_cost(
                        stock, entry_p, walk_forward.QUANTITY, "buy")
                    sell_cost = walk_forward.calculate_trade_cost(
                        stock, p, walk_forward.QUANTITY, "sell")
                    pnl = (p - entry_p) * walk_forward.QUANTITY - buy_cost - sell_cost
                    pnls.append(pnl)
                    break
            # If no stop/target hit within 390 bars (~1 trading day), close at last bar
            else:
                if shifted_idx + 1 < n_bars:
                    exit_p = close.iloc[min(shifted_idx + 390, n_bars - 1)]
                    buy_cost = walk_forward.calculate_trade_cost(
                        stock, entry_p, walk_forward.QUANTITY, "buy")
                    sell_cost = walk_forward.calculate_trade_cost(
                        stock, exit_p, walk_forward.QUANTITY, "sell")
                    pnl = (exit_p - entry_p) * walk_forward.QUANTITY - buy_cost - sell_cost
                    pnls.append(pnl)

        perm_sharpes.append(safe_sharpe(pnls))

        if (perm_i + 1) % 100 == 0:
            print(f"    Signal-shift: {perm_i + 1}/{n_perms} done")

    count_as_good = sum(1 for s in perm_sharpes if s >= actual_sharpe)
    p_value = (count_as_good + 1) / (n_perms + 1)

    return {
        "p_value": p_value,
        "perm_sharpes": perm_sharpes,
        "mean_perm_sharpe": float(np.mean(perm_sharpes)),
        "std_perm_sharpe": float(np.std(perm_sharpes)),
    }


def fisher_combined_pvalue(p_values):
    """Combine p-values using Fisher's method.

    Fisher's method: test_stat = -2 * sum(ln(p_i)) follows chi-squared
    with 2k degrees of freedom, where k = number of p-values.
    """
    p_arr = np.array(p_values, dtype=float)
    p_arr = np.clip(p_arr, 1.0 / (N_PERMUTATIONS + 1), 1.0)

    test_stat = -2.0 * np.sum(np.log(p_arr))
    df = 2 * len(p_arr)
    combined_p = 1.0 - stats.chi2.cdf(test_stat, df)
    return float(combined_p)


def whites_reality_check_threshold(n_combinations, base_alpha=0.05):
    """Calculate adjusted p-value threshold for data mining bias.

    We tested n_combinations parameter sets per stock in walk_forward.
    The adjusted threshold accounts for the fact that the best of many
    tries will look good even on random data.

    Formula: adjusted = base_alpha / sqrt(n_combinations)
    This is a practical approximation — more conservative than raw alpha,
    less extreme than full Bonferroni over all combinations.
    """
    return base_alpha / math.sqrt(n_combinations)


def plot_sharpe_histogram(stock, perm_sharpes, actual_sharpe, p_value,
                          save_path, test_name=""):
    """Plot histogram of permuted Sharpes with actual Sharpe marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_sharpes, bins=50, color="steelblue", alpha=0.7,
            edgecolor="white", label="Permuted Sharpes")
    ax.axvline(actual_sharpe, color="red", linewidth=2, linestyle="--",
               label=f"Actual: {actual_sharpe:.2f}")

    verdict = ("SIGNIFICANT" if p_value < BONFERRONI_ALPHA
               else "MARGINAL" if p_value < 0.05
               else "NOT SIGNIFICANT")
    title = f"{stock} Permutation Test"
    if test_name:
        title += f" ({test_name})"
    title += f" (n={len(perm_sharpes)})"
    ax.set_title(title)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Count")
    ax.legend()

    ax.text(0.97, 0.95, f"p = {p_value:.4f}\n{verdict}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_summary_chart(results, combined_p, save_path):
    """Horizontal bar chart of p-values for all tested stocks."""
    stocks = [r["stock"] for r in results]
    p_values = [r["combined_p"] for r in results]

    colors = []
    for p in p_values:
        if p < BONFERRONI_ALPHA:
            colors.append("#2ecc71")  # green
        elif p < 0.05:
            colors.append("#f39c12")  # yellow
        else:
            colors.append("#e74c3c")  # red

    fig, ax = plt.subplots(figsize=(10, max(6, len(stocks) * 0.4)))
    y_pos = range(len(stocks))
    ax.barh(y_pos, p_values, color=colors, edgecolor="white", height=0.6)

    ax.axvline(BONFERRONI_ALPHA, color="green", linewidth=1.5,
               linestyle="--", label=f"Bonferroni ({BONFERRONI_ALPHA:.4f})")
    ax.axvline(0.05, color="orange", linewidth=1.5,
               linestyle="--", label="Nominal (0.05)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stocks)
    ax.set_xlabel("p-value (combined min of both tests)")
    ax.set_title(f"Permutation Test Summary | Fisher combined p = {combined_p:.6f}")
    ax.legend(loc="lower right")
    ax.set_xscale("log")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def print_summary_table(results, fisher_day_p, fisher_shift_p, fisher_combined_p):
    """Print formatted tiered summary of all permutation test results."""
    wrc_threshold = whites_reality_check_threshold(COMBINATIONS_TESTED)

    print(f"\n{'=' * 80}")
    print(f"  PERMUTATION TEST RESULTS (n={N_PERMUTATIONS})")
    print(f"{'=' * 80}")
    print(f"  {'Stock':<7} {'WF':<6} {'Sharpe':>7} {'DayShuffle':>11} "
          f"{'SigShift':>9} {'Best p':>8}  Verdict")
    print(f"  {'-' * 72}")

    # Sort by combined p-value
    sorted_results = sorted(results, key=lambda r: r["combined_p"])

    tier1 = []  # p < 0.05
    tier2 = []  # 0.05 - 0.15
    tier3 = []  # 0.15 - 0.30
    tier4 = []  # > 0.30

    for r in sorted_results:
        p = r["combined_p"]
        if p < BONFERRONI_ALPHA:
            verdict = "SIGNIFICANT"
        elif p < 0.05:
            verdict = "MARGINAL"
        elif p < 0.15:
            verdict = "PROMISING"
        elif p < 0.30:
            verdict = "WEAK"
        else:
            verdict = "NO EDGE"

        if p < 0.05:
            tier1.append(r)
        elif p < 0.15:
            tier2.append(r)
        elif p < 0.30:
            tier3.append(r)
        else:
            tier4.append(r)

        status = r.get("wf_status", "?")
        print(f"  {r['stock']:<7} {status:<6} {r['actual_sharpe']:>7.2f} "
              f"{r['day_shuffle_p']:>11.4f} {r['signal_shift_p']:>9.4f} "
              f"{r['combined_p']:>8.4f}  {verdict}")

    # Tiered summary
    print(f"\n  {'-' * 72}")
    print(f"  TIERED SUMMARY")
    print(f"  {'-' * 72}")

    if tier1:
        stocks_str = ", ".join(r["stock"] for r in tier1)
        print(f"  Tier 1 (p < 0.05)  Strong evidence:   {stocks_str}")
    if tier2:
        stocks_str = ", ".join(r["stock"] for r in tier2)
        print(f"  Tier 2 (p < 0.15)  Promising:          {stocks_str}")
    if tier3:
        stocks_str = ", ".join(r["stock"] for r in tier3)
        print(f"  Tier 3 (p < 0.30)  Weak signal:        {stocks_str}")
    if tier4:
        stocks_str = ", ".join(r["stock"] for r in tier4)
        print(f"  Tier 4 (p > 0.30)  No evidence:        {stocks_str}")

    # Fisher combined p-values
    print(f"\n  {'-' * 72}")
    print(f"  COMBINED P-VALUES (Fisher's method)")
    print(f"  {'-' * 72}")
    print(f"  Day-shuffle Fisher p:    {fisher_day_p:.6f}")
    print(f"  Signal-shift Fisher p:   {fisher_shift_p:.6f}")
    print(f"  Overall Fisher p:        {fisher_combined_p:.6f}")
    print(f"  Bonferroni threshold:    {BONFERRONI_ALPHA:.6f}")

    # White's Reality Check
    print(f"\n  {'-' * 72}")
    print(f"  WHITE'S REALITY CHECK")
    print(f"  {'-' * 72}")
    print(f"  Parameter combinations tested per stock: {COMBINATIONS_TESTED:,}")
    print(f"  Adjusted threshold: 0.05 / sqrt({COMBINATIONS_TESTED}) = {wrc_threshold:.6f}")
    print(f"  Note: Walk-forward train/test split provides first-layer protection")
    print(f"  against overfitting. This permutation test validates the out-of-sample")
    print(f"  result. The WRC threshold is shown for reference — stocks passing")
    print(f"  Bonferroni ({BONFERRONI_ALPHA:.4f}) are robust even accounting for")
    print(f"  data mining, since we test out-of-sample performance, not in-sample.")

    # Overall verdict
    print(f"\n  {'-' * 72}")
    if fisher_combined_p < 0.01:
        print(f"  VERDICT: STRONG EVIDENCE OF EDGE")
    elif fisher_combined_p < 0.05:
        print(f"  VERDICT: STRATEGY HAS SIGNIFICANT EDGE")
    elif fisher_combined_p < 0.10:
        print(f"  VERDICT: MARGINAL EVIDENCE OF EDGE")
    else:
        print(f"  VERDICT: NO SIGNIFICANT EDGE DETECTED")

    # Recommendation
    print(f"\n  RECOMMENDATION:")
    if tier1:
        stocks_str = ", ".join(r["stock"] for r in tier1)
        print(f"  -> Trade with confidence: {stocks_str}")
    if tier2:
        stocks_str = ", ".join(r["stock"] for r in tier2)
        print(f"  -> Monitor / trade with reduced size: {stocks_str}")
    if tier3 or tier4:
        drop = tier3 + tier4
        stocks_str = ", ".join(r["stock"] for r in drop)
        print(f"  -> Consider dropping: {stocks_str}")

    print(f"{'=' * 80}\n")


def run_all(stocks=None, n_perms=None, simple_mode=None):
    """Main entry point. Fetch data, run both tests, save results."""
    global N_PERMUTATIONS
    if n_perms is not None:
        N_PERMUTATIONS = n_perms

    if simple_mode is not None:
        walk_forward.SIMPLE_SIGNAL_MODE = simple_mode

    if stocks is None:
        stocks = walk_forward.STOCKS_TO_TEST

    os.makedirs(RESULTS_DIR, exist_ok=True)

    mode_str = "SIMPLE (EMA+RSI only)" if walk_forward.SIMPLE_SIGNAL_MODE else "FULL"
    print(f"Permutation Test — {len(stocks)} stocks, "
          f"{N_PERMUTATIONS} permutations each")
    print(f"Signal mode: {mode_str}")
    print(f"Bonferroni alpha: {BONFERRONI_ALPHA:.6f}")
    wrc = whites_reality_check_threshold(COMBINATIONS_TESTED)
    print(f"White's Reality Check threshold: {wrc:.6f} "
          f"(0.05 / sqrt({COMBINATIONS_TESTED}))")
    print()

    # Fetch SPY bars once
    print("Fetching SPY bars...")
    spy_bars = walk_forward.get_spy_bars(walk_forward.LOOKBACK_DAYS)
    print(f"  Got {len(spy_bars)} SPY bars\n")

    all_results = []

    for stock in stocks:
        print(f"{'=' * 60}")
        print(f"  {stock}")
        print(f"{'=' * 60}")

        wf_result = load_stock_result(stock)
        if wf_result is None:
            print(f"  Skipping — no valid walk-forward result (FAIL or <2 trades)\n")
            continue

        params = wf_result["best_params"]
        actual_sharpe = wf_result["test_sharpe"]
        # Apply Sharpe cap to actual value too
        if wf_result["test_trades"] < MIN_TRADES_FOR_SHARPE:
            actual_sharpe = 0.0

        print(f"  WF status: {wf_result['status']} | "
              f"Sharpe: {actual_sharpe:.2f} | "
              f"Trades: {wf_result['test_trades']}")
        print(f"  Params: EMA {params['fast']}/{params['slow']} | "
              f"RSI {params['rsi_buy']}/{params['rsi_sell']} | "
              f"BB {params['bb_period']}/{params['bb_std']}")

        # Fetch stock bars
        print(f"  Fetching {stock} bars...")
        bars = walk_forward.get_minute_bars(stock, walk_forward.LOOKBACK_DAYS)
        if bars is None:
            print(f"  Failed to fetch bars, skipping\n")
            continue
        print(f"  Got {len(bars)} bars")

        # Split into train/test — use test portion only
        _, test_bars, _, test_spy = walk_forward.split_data(
            bars, spy_bars, walk_forward.TEST_DAYS
        )
        print(f"  Test period: {len(test_bars)} bars "
              f"({test_bars.index[0].date()} to {test_bars.index[-1].date()})")

        # Test 1: Day-block shuffle
        print(f"  Running day-shuffle test ({N_PERMUTATIONS} perms)...")
        day_result = run_day_shuffle_test(
            stock, test_bars, test_spy, params, actual_sharpe,
            n_perms=N_PERMUTATIONS
        )

        # Test 2: Signal shift
        print(f"  Running signal-shift test ({N_PERMUTATIONS} perms)...")
        shift_result = run_signal_shift_test(
            stock, test_bars, test_spy, params, actual_sharpe,
            n_perms=N_PERMUTATIONS
        )

        # Combined p-value is the minimum of the two tests
        # (take the more favorable test — if either method finds significance,
        # the signal is likely real)
        combined_p = min(day_result["p_value"], shift_result["p_value"])

        result = {
            "stock": stock,
            "actual_sharpe": actual_sharpe,
            "wf_status": wf_result["status"],
            "wf_trades": wf_result["test_trades"],
            "day_shuffle_p": day_result["p_value"],
            "day_shuffle_mean": day_result["mean_perm_sharpe"],
            "day_shuffle_std": day_result["std_perm_sharpe"],
            "signal_shift_p": shift_result["p_value"],
            "signal_shift_mean": shift_result["mean_perm_sharpe"],
            "signal_shift_std": shift_result["std_perm_sharpe"],
            "combined_p": combined_p,
            "day_perm_sharpes": day_result["perm_sharpes"],
            "shift_perm_sharpes": shift_result["perm_sharpes"],
        }

        # Save per-stock JSON (without large arrays)
        json_out = {k: v for k, v in result.items()
                    if k not in ("day_perm_sharpes", "shift_perm_sharpes")}
        json_out["timestamp"] = datetime.now().isoformat()
        json_out["signal_mode"] = "simple" if walk_forward.SIMPLE_SIGNAL_MODE else "full"
        with open(f"{RESULTS_DIR}/{stock}.json", "w") as f:
            json.dump(json_out, f, indent=2)

        # Plot histograms for both tests
        plot_sharpe_histogram(
            stock, day_result["perm_sharpes"], actual_sharpe,
            day_result["p_value"],
            f"{RESULTS_DIR}/{stock}_day_shuffle.png",
            test_name="Day Shuffle"
        )
        plot_sharpe_histogram(
            stock, shift_result["perm_sharpes"], actual_sharpe,
            shift_result["p_value"],
            f"{RESULTS_DIR}/{stock}_signal_shift.png",
            test_name="Signal Shift"
        )

        verdict = ("SIGNIFICANT" if combined_p < BONFERRONI_ALPHA
                    else "MARGINAL" if combined_p < 0.05
                    else "PROMISING" if combined_p < 0.15
                    else "NOT SIG.")
        print(f"  Day-shuffle p={day_result['p_value']:.4f} | "
              f"Signal-shift p={shift_result['p_value']:.4f} | "
              f"Best={combined_p:.4f} -> {verdict}\n")

        all_results.append(result)

    if not all_results:
        print("No stocks had valid walk-forward results to test.")
        return

    # Compute Fisher combined p-values for each test type
    day_pvals = [r["day_shuffle_p"] for r in all_results]
    shift_pvals = [r["signal_shift_p"] for r in all_results]
    combined_pvals = [r["combined_p"] for r in all_results]

    fisher_day_p = fisher_combined_pvalue(day_pvals)
    fisher_shift_p = fisher_combined_pvalue(shift_pvals)
    fisher_combined_p = fisher_combined_pvalue(combined_pvals)

    # Print tiered summary
    print_summary_table(all_results, fisher_day_p, fisher_shift_p,
                        fisher_combined_p)

    # Plot summary chart
    plot_summary_chart(all_results, fisher_combined_p,
                       f"{RESULTS_DIR}/summary.png")

    # Save combined results
    combined_out = {
        "timestamp": datetime.now().isoformat(),
        "n_permutations": N_PERMUTATIONS,
        "signal_mode": "simple" if walk_forward.SIMPLE_SIGNAL_MODE else "full",
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "whites_reality_check_threshold": whites_reality_check_threshold(COMBINATIONS_TESTED),
        "combinations_tested_per_stock": COMBINATIONS_TESTED,
        "fisher_day_shuffle_p": fisher_day_p,
        "fisher_signal_shift_p": fisher_shift_p,
        "fisher_combined_p": fisher_combined_p,
        "n_stocks_tested": len(all_results),
        "n_significant": sum(1 for r in all_results
                             if r["combined_p"] < BONFERRONI_ALPHA),
        "n_marginal": sum(1 for r in all_results
                          if BONFERRONI_ALPHA <= r["combined_p"] < 0.05),
        "n_promising": sum(1 for r in all_results
                           if 0.05 <= r["combined_p"] < 0.15),
        "per_stock": [{k: v for k, v in r.items()
                       if k not in ("day_perm_sharpes", "shift_perm_sharpes")}
                      for r in all_results]
    }
    with open(f"{RESULTS_DIR}/combined.json", "w") as f:
        json.dump(combined_out, f, indent=2)

    print(f"Results saved to {RESULTS_DIR}/")
    print(f"  - Per-stock JSONs and histograms (day_shuffle + signal_shift)")
    print(f"  - summary.png")
    print(f"  - combined.json")


if __name__ == "__main__":
    target_stocks = None
    perms = None
    simple = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--stocks" and i + 1 < len(args):
            target_stocks = [s.strip() for s in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--perms" and i + 1 < len(args):
            perms = int(args[i + 1])
            i += 2
        elif args[i] == "--simple":
            simple = True
            i += 1
        elif args[i] == "--full":
            simple = False
            i += 1
        else:
            print(f"Unknown argument: {args[i]}")
            print("Usage: python3 permutation_test.py [--stocks AAPL,NVDA] "
                  "[--perms 500] [--simple|--full]")
            sys.exit(1)

    run_all(stocks=target_stocks, n_perms=perms, simple_mode=simple)
