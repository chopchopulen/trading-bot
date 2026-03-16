"""
Permutation Test Module for Trading Bot
========================================
Tests whether the walk-forward optimized strategy produces statistically
significant results or could be explained by random chance.

Method: Day-block permutation
- Randomly reorder entire trading days (keeping intraday bars intact)
- This preserves ATR, VWAP, and intraday microstructure
- But destroys inter-day trends that the strategy relies on
- If the real strategy beats most permuted versions, the edge is real

Statistical framework:
- 1,000 permutations per stock (p-value resolution: 0.001)
- Bonferroni correction for 26 stocks: alpha = 0.05/26 = 0.00192
- Fisher's method to combine per-stock p-values into one number
- Test metric: Sharpe ratio (consistent with walk-forward criteria)
"""

import walk_forward
import numpy as np
import pandas as pd
import json
import os
import sys
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

    This is the core of the permutation test. By reordering entire days,
    we preserve intraday price structure (ATR, VWAP work correctly) but
    destroy inter-day trends and regime patterns that the strategy exploits.

    Args:
        bars: minute-bar DataFrame with DatetimeIndex
        rng: numpy random generator for reproducibility

    Returns:
        New DataFrame with days in random order, index reset to sequential
        timestamps so EMA/RSI lookback calculations work correctly.
    """
    dates = np.unique(bars.index.date)
    shuffled_order = rng.permutation(len(dates))
    shuffled_dates = dates[shuffled_order]

    chunks = []
    for date in shuffled_dates:
        day_bars = bars[bars.index.date == date]
        chunks.append(day_bars)

    result = pd.concat(chunks)
    # Reassign the original sequential index so indicator lookback works
    result.index = bars.index[:len(result)]
    return result


def run_permutation_test_single_stock(stock, bars, spy_bars, params,
                                       actual_sharpe, n_perms=N_PERMUTATIONS,
                                       seed=42):
    """Run permutation test for one stock.

    For each permutation, shuffle the trading days and re-run the backtest
    with the same optimized parameters. Count how many permuted Sharpes
    beat the actual Sharpe to compute the p-value.

    Args:
        stock: ticker symbol (e.g. "AAPL")
        bars: test-period minute bars for this stock
        spy_bars: SPY minute bars (same period)
        params: dict with keys: fast, slow, rsi_buy, rsi_sell, bb_period, bb_std
        actual_sharpe: the real Sharpe from walk-forward verification
        n_perms: number of permutations (default 1000)
        seed: random seed for reproducibility

    Returns:
        dict with: stock, actual_sharpe, p_value, perm_sharpes,
                   mean_perm_sharpe, std_perm_sharpe, n_perms, significant
    """
    # Set global STOCK so transaction costs are correct
    walk_forward.STOCK = stock

    rng = np.random.default_rng(seed)
    perm_sharpes = []

    for i in range(n_perms):
        shuffled_bars = shuffle_bars_by_day(bars, rng)
        shuffled_spy = shuffle_bars_by_day(spy_bars, rng)

        result = walk_forward.run_backtest(
            shuffled_bars, shuffled_spy,
            params["fast"], params["slow"],
            params["rsi_buy"], params["rsi_sell"],
            params["bb_period"], params["bb_std"]
        )
        perm_sharpes.append(result["sharpe"])

        if (i + 1) % 100 == 0:
            print(f"  {stock}: {i + 1}/{n_perms} permutations done")

    # Standard permutation p-value: (count >= actual + 1) / (N + 1)
    count_as_good = sum(1 for s in perm_sharpes if s >= actual_sharpe)
    p_value = (count_as_good + 1) / (n_perms + 1)

    return {
        "stock": stock,
        "actual_sharpe": actual_sharpe,
        "p_value": p_value,
        "perm_sharpes": perm_sharpes,
        "mean_perm_sharpe": float(np.mean(perm_sharpes)),
        "std_perm_sharpe": float(np.std(perm_sharpes)),
        "n_perms": n_perms,
        "significant": p_value < BONFERRONI_ALPHA
    }


def fisher_combined_pvalue(p_values):
    """Combine p-values using Fisher's method.

    Fisher's method: test_stat = -2 * sum(ln(p_i)) follows chi-squared
    with 2k degrees of freedom, where k = number of p-values.

    More sensitive than Stouffer's at detecting a few strong signals
    among many nulls, which matches our situation (9 PASS out of 26).
    """
    p_arr = np.array(p_values, dtype=float)
    # Clamp to avoid log(0) — minimum possible p is 1/(N+1)
    p_arr = np.clip(p_arr, 1.0 / (N_PERMUTATIONS + 1), 1.0)

    test_stat = -2.0 * np.sum(np.log(p_arr))
    df = 2 * len(p_arr)
    combined_p = 1.0 - stats.chi2.cdf(test_stat, df)
    return float(combined_p)


def plot_sharpe_histogram(stock, perm_sharpes, actual_sharpe, p_value,
                          save_path):
    """Plot histogram of permuted Sharpes with actual Sharpe marked.

    This is the most informative visual — shows where the real strategy
    falls in the distribution of random outcomes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_sharpes, bins=50, color="steelblue", alpha=0.7,
            edgecolor="white", label="Permuted Sharpes")
    ax.axvline(actual_sharpe, color="red", linewidth=2, linestyle="--",
               label=f"Actual: {actual_sharpe:.2f}")

    verdict = ("SIGNIFICANT" if p_value < BONFERRONI_ALPHA
               else "MARGINAL" if p_value < 0.05
               else "NOT SIGNIFICANT")
    ax.set_title(f"{stock} Permutation Test (n={len(perm_sharpes)})")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Count")
    ax.legend()

    # Annotate p-value in top right
    ax.text(0.97, 0.95, f"p = {p_value:.4f}\n{verdict}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_summary_chart(results, combined_p, save_path):
    """Horizontal bar chart of p-values for all tested stocks.

    Green = significant (p < 0.002), yellow = marginal, red = not significant.
    Vertical lines mark Bonferroni and nominal thresholds.
    """
    stocks = [r["stock"] for r in results]
    p_values = [r["p_value"] for r in results]

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
    ax.set_xlabel("p-value")
    ax.set_title(f"Permutation Test Summary | Fisher combined p = {combined_p:.6f}")
    ax.legend(loc="lower right")
    ax.set_xscale("log")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def print_summary_table(results, combined_p):
    """Print formatted terminal summary of all permutation test results."""
    print(f"\n{'=' * 64}")
    print(f"  PERMUTATION TEST RESULTS (n={N_PERMUTATIONS})")
    print(f"{'=' * 64}")
    print(f"  {'Stock':<8} {'Status':<10} {'Sharpe':>8} {'p-value':>9}  Verdict")
    print(f"  {'-' * 56}")

    n_sig = 0
    n_marg = 0
    n_fail = 0

    for r in results:
        if r["p_value"] < BONFERRONI_ALPHA:
            verdict = "SIGNIFICANT"
            n_sig += 1
        elif r["p_value"] < 0.05:
            verdict = "MARGINAL"
            n_marg += 1
        else:
            verdict = "NOT SIG."
            n_fail += 1

        status = r.get("wf_status", "?")
        print(f"  {r['stock']:<8} {status:<10} {r['actual_sharpe']:>8.2f} "
              f"{r['p_value']:>9.4f}  {verdict}")

    print(f"  {'-' * 56}")
    print(f"  Tested: {len(results)} stocks | "
          f"Significant: {n_sig} | Marginal: {n_marg} | Not sig: {n_fail}")
    print(f"\n  Combined Fisher p-value: {combined_p:.6f}")
    print(f"  Bonferroni threshold:    {BONFERRONI_ALPHA:.6f}")

    if combined_p < 0.05:
        print(f"\n  VERDICT: STRATEGY HAS SIGNIFICANT EDGE")
    elif combined_p < 0.10:
        print(f"\n  VERDICT: MARGINAL EVIDENCE OF EDGE")
    else:
        print(f"\n  VERDICT: NO SIGNIFICANT EDGE DETECTED")

    print(f"\n  Note: Walk-forward train/test split controls for parameter")
    print(f"  optimization. This permutation test validates whether the")
    print(f"  out-of-sample performance is distinguishable from chance.")
    print(f"{'=' * 64}\n")


def run_all(stocks=None, n_perms=None):
    """Main entry point. Fetch data, run tests, save results.

    Args:
        stocks: optional list of tickers to test. Defaults to all 26.
        n_perms: optional override for number of permutations.
    """
    global N_PERMUTATIONS
    if n_perms is not None:
        N_PERMUTATIONS = n_perms

    if stocks is None:
        stocks = walk_forward.STOCKS_TO_TEST

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Permutation Test — {len(stocks)} stocks, "
          f"{N_PERMUTATIONS} permutations each")
    print(f"Bonferroni alpha: {BONFERRONI_ALPHA:.6f}")
    print()

    # Fetch SPY bars once (reuse for all stocks)
    print("Fetching SPY bars...")
    spy_bars = walk_forward.get_spy_bars(walk_forward.LOOKBACK_DAYS)
    print(f"  Got {len(spy_bars)} SPY bars\n")

    all_results = []

    for stock in stocks:
        print(f"{'=' * 50}")
        print(f"  {stock}")
        print(f"{'=' * 50}")

        # Load walk-forward result
        wf_result = load_stock_result(stock)
        if wf_result is None:
            print(f"  Skipping {stock} — no valid walk-forward result "
                  f"(FAIL or <2 trades)\n")
            continue

        params = wf_result["best_params"]
        actual_sharpe = wf_result["test_sharpe"]
        print(f"  WF status: {wf_result['status']} | "
              f"Sharpe: {actual_sharpe:.2f} | "
              f"Trades: {wf_result['test_trades']}")
        print(f"  Params: EMA {params['fast']}/{params['slow']} | "
              f"RSI {params['rsi_buy']}/{params['rsi_sell']} | "
              f"BB {params['bb_period']}/{params['bb_std']}")

        # Fetch stock bars
        print(f"  Fetching {stock} bars...")
        bars = walk_forward.get_minute_bars(stock, walk_forward.LOOKBACK_DAYS)
        print(f"  Got {len(bars)} bars")

        # Split into train/test — use test portion only
        _, test_bars, _, test_spy = walk_forward.split_data(
            bars, spy_bars, walk_forward.TEST_DAYS
        )
        print(f"  Test period: {len(test_bars)} bars "
              f"({test_bars.index[0].date()} to {test_bars.index[-1].date()})")

        # Run permutation test
        print(f"  Running {N_PERMUTATIONS} permutations...")
        result = run_permutation_test_single_stock(
            stock, test_bars, test_spy, params, actual_sharpe,
            n_perms=N_PERMUTATIONS
        )
        result["wf_status"] = wf_result["status"]

        # Save per-stock JSON (without the large perm_sharpes array)
        json_out = {k: v for k, v in result.items() if k != "perm_sharpes"}
        json_out["timestamp"] = datetime.now().isoformat()
        with open(f"{RESULTS_DIR}/{stock}.json", "w") as f:
            json.dump(json_out, f, indent=2)

        # Plot histogram
        plot_sharpe_histogram(
            stock, result["perm_sharpes"], actual_sharpe, result["p_value"],
            f"{RESULTS_DIR}/{stock}_histogram.png"
        )

        verdict = ("SIGNIFICANT" if result["p_value"] < BONFERRONI_ALPHA
                    else "MARGINAL" if result["p_value"] < 0.05
                    else "NOT SIG.")
        print(f"  Result: p = {result['p_value']:.4f} -> {verdict}\n")

        all_results.append(result)

    if not all_results:
        print("No stocks had valid walk-forward results to test.")
        return

    # Compute Fisher combined p-value
    p_values = [r["p_value"] for r in all_results]
    combined_p = fisher_combined_pvalue(p_values)

    # Print summary
    print_summary_table(all_results, combined_p)

    # Plot summary chart
    plot_summary_chart(all_results, combined_p,
                       f"{RESULTS_DIR}/summary.png")

    # Save combined results
    combined_out = {
        "timestamp": datetime.now().isoformat(),
        "n_permutations": N_PERMUTATIONS,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "fisher_combined_p": combined_p,
        "n_stocks_tested": len(all_results),
        "n_significant": sum(1 for r in all_results
                             if r["p_value"] < BONFERRONI_ALPHA),
        "n_marginal": sum(1 for r in all_results
                          if BONFERRONI_ALPHA <= r["p_value"] < 0.05),
        "per_stock": [{k: v for k, v in r.items() if k != "perm_sharpes"}
                      for r in all_results]
    }
    with open(f"{RESULTS_DIR}/combined.json", "w") as f:
        json.dump(combined_out, f, indent=2)

    print(f"Results saved to {RESULTS_DIR}/")
    print(f"  - Per-stock JSONs and histograms")
    print(f"  - summary.png")
    print(f"  - combined.json")


if __name__ == "__main__":
    # Parse optional CLI args
    target_stocks = None
    perms = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--stocks" and i + 1 < len(args):
            target_stocks = [s.strip() for s in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--perms" and i + 1 < len(args):
            perms = int(args[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {args[i]}")
            print("Usage: python3 permutation_test.py [--stocks AAPL,NVDA] [--perms 500]")
            sys.exit(1)

    run_all(stocks=target_stocks, n_perms=perms)
