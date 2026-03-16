#!/usr/bin/env python3
"""
Phase 1 Optimization Runner

This script runs the enhanced preprocessing pipeline with:
- Expanded universe (S&P 500 or Russell 1000)
- Multi-source data aggregation (Finnhub, FMP, Yahoo RSS)
- Extended lookback period (2 years)

Usage:
    python run_phase1.py [--synthetic] [--universe {original,sp500,russell1000}]

Options:
    --synthetic: Use synthetic historical data instead of fetching from APIs
    --universe: Select universe size (default: russell1000)
"""

import argparse
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_with_synthetic_data():
    """Generate and use synthetic data for quick testing."""
    print("=" * 70)
    print("Phase 1: Running with SYNTHETIC DATA")
    print("=" * 70)

    from src.kaggle_data_loader import generate_synthetic_historical_data
    from config import UNIVERSE_PATH, RAW_DATA_PATH
    import pandas as pd

    # Load universe
    universe = pd.read_csv(UNIVERSE_PATH)
    tickers = universe['ticker'].tolist()
    print(f"Using {len(tickers)} tickers from {UNIVERSE_PATH}")

    # Generate synthetic data
    df = generate_synthetic_historical_data(
        tickers=tickers,
        start_date="2024-01-01",
        end_date="2025-03-01",
        avg_news_per_day=3
    )

    # Save
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"\nSaved {len(df)} synthetic observations to {RAW_DATA_PATH}")

    return df


def run_with_live_apis():
    """Fetch live data from multiple APIs."""
    print("=" * 70)
    print("Phase 1: Running with LIVE API DATA")
    print("=" * 70)

    from src.preprocess_v2 import main_v2

    # Check API keys
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    fmp_key = os.getenv("FMP_API_KEY")

    if not finnhub_key and not fmp_key:
        print("\nWARNING: No API keys found!")
        print("Set environment variables:")
        print("  export FINNHUB_API_KEY='your_key'")
        print("  export FMP_API_KEY='your_key'")
        print("\nFalling back to Yahoo RSS only (limited coverage)\n")

    return main_v2()


def run_feature_engineering():
    """Run feature engineering on the raw data."""
    print("\n" + "=" * 70)
    print("Running Feature Engineering")
    print("=" * 70)

    from src import feature_engineering
    import pandas as pd
    from config import RAW_DATA_PATH

    df = pd.read_csv(RAW_DATA_PATH)
    df = feature_engineering.main(df)

    return df


def run_signal_construction():
    """Run signal construction."""
    print("\n" + "=" * 70)
    print("Running Signal Construction")
    print("=" * 70)

    from src import signal_construction
    import pandas as pd
    from config import PROCESSED_PATH

    df = pd.read_csv(PROCESSED_PATH)
    df = signal_construction.main(df)

    return df


def run_evaluation():
    """Run evaluation with expanded metrics."""
    print("\n" + "=" * 70)
    print("Running Evaluation")
    print("=" * 70)

    from src import evaluation
    from src.walkforward import run_walkforward_analysis
    import pandas as pd
    from config import PROCESSED_PATH

    df = pd.read_csv(PROCESSED_PATH)
    df['date'] = pd.to_datetime(df['date'])

    # Baseline comparison
    print("\n--- Baseline Comparison ---")
    summary = evaluation.run_baseline_comparison(df)
    print(summary.to_string())

    # Walk-forward
    print("\n--- Walk-Forward Analysis ---")
    wf_results = run_walkforward_analysis(df)
    print(wf_results.to_string())

    return summary, wf_results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Optimization Runner")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of live APIs"
    )
    parser.add_argument(
        "--universe",
        choices=["original", "sp500", "russell1000"],
        default="russell1000",
        help="Select universe size (default: russell1000)"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (use existing raw data)"
    )
    args = parser.parse_args()

    # Set universe in config
    import config
    config.UNIVERSE_TYPE = args.universe
    config.UNIVERSE_PATH = config.UNIVERSE_PATHS[args.universe]

    print(f"Configuration:")
    print(f"  Universe: {args.universe} ({config.UNIVERSE_PATH})")
    print(f"  Data source: {'Synthetic' if args.synthetic else 'Live APIs'}")
    print()

    # Step 1: Preprocessing
    if not args.skip_preprocess:
        if args.synthetic:
            run_with_synthetic_data()
        else:
            run_with_live_apis()
    else:
        print("Skipping preprocessing (using existing data)")

    # Step 2: Feature Engineering
    df_features = run_feature_engineering()

    # Step 3: Signal Construction
    df_signals = run_signal_construction()

    # Step 4: Evaluation
    summary, wf_results = run_evaluation()

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)

    if len(summary) > 0:
        best_signal = summary.loc[summary['mean_ic'].abs().idxmax()]
        print(f"\nBest performing signal: {best_signal['signal']}")
        print(f"  Mean IC: {best_signal['mean_ic']:.4f}")
        print(f"  T-Stat: {best_signal['ic_t_statistic']:.4f}")
        print(f"  IR: {best_signal['annualized_ir']:.4f}")

        if abs(best_signal['ic_t_statistic']) > 2.0:
            print("  ✓ Statistically significant at 5% level!")
        elif abs(best_signal['ic_t_statistic']) > 1.5:
            print("  ~ Approaching significance (need more data)")
        else:
            print("  ✗ Not significant (consider expanding universe/history)")

    print("\nNext steps:")
    print("  1. Run Jupyter notebook for detailed analysis")
    print("  2. Commit results: git add -A && git commit -m 'Phase 1 results'")
    print("  3. Update resume with new metrics")


if __name__ == "__main__":
    main()
