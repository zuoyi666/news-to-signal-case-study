"""
walkforward.py

Walk-forward validation framework for signal evaluation.
Implements time-series cross-validation to test signal robustness
across different market regimes.
"""

from typing import Optional

import numpy as np
import pandas as pd

from config import SIGNAL_COLS
from src.evaluation import (
    compute_daily_ic,
    compute_daily_spread,
    compute_ic_statistics,
    compute_risk_adjusted_metrics,
)


def split_data_by_date(
    df: pd.DataFrame,
    train_days: int = 180,
    val_days: int = 90,
    test_days: int = 95,
) -> dict[str, pd.DataFrame]:
    """
    Split data into train/validation/test sets by date.

    Args:
        df: DataFrame with 'date' column.
        train_days: Number of days for training period.
        val_days: Number of days for validation period.
        test_days: Number of days for test period.

    Returns:
        Dictionary with 'train', 'validation', 'test' DataFrames.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    unique_dates = pd.to_datetime(df["date"].unique())
    unique_dates = sorted(unique_dates)

    n_dates = len(unique_dates)

    if n_dates < train_days + val_days + test_days:
        print(
            f"Warning: Insufficient data ({n_dates} unique dates). "
            f"Required: {train_days + val_days + test_days}"
        )
        # Adjust split proportionally
        total = train_days + val_days + test_days
        train_end = int(n_dates * train_days / total)
        val_end = int(n_dates * (train_days + val_days) / total)
    else:
        train_end = train_days
        val_end = train_days + val_days

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:val_end + test_days]

    train_df = df[df["date"].isin(train_dates)]
    val_df = df[df["date"].isin(val_dates)]
    test_df = df[df["date"].isin(test_dates)]

    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
        "train_dates": len(train_dates),
        "val_dates": len(val_dates),
        "test_dates": len(test_dates),
    }


def evaluate_on_split(df: pd.DataFrame, signal_col: str) -> dict:
    """
    Evaluate a signal on a data split.

    Args:
        df: DataFrame containing the split data.
        signal_col: Signal column name.

    Returns:
        Dictionary with evaluation metrics.
    """
    if df.empty or signal_col not in df.columns:
        return {
            "n_days": 0,
            "mean_ic": np.nan,
            "ic_t_stat": np.nan,
            "ic_p_value": np.nan,
            "annualized_ir": np.nan,
            "mean_spread": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
        }

    # Compute metrics
    ic_series = compute_daily_ic(df, signal_col)
    spread_series = compute_daily_spread(df, signal_col)

    ic_stats = compute_ic_statistics(ic_series)
    risk_metrics = compute_risk_adjusted_metrics(spread_series)

    return {
        "n_days": len(ic_series),
        "mean_ic": ic_stats["mean_ic"],
        "ic_t_stat": ic_stats["t_statistic"],
        "ic_p_value": ic_stats["p_value"],
        "significant_5pct": ic_stats["significant_5pct"],
        "annualized_ir": ic_stats["annualized_ir"],
        "mean_spread": spread_series.mean() if len(spread_series) > 0 else np.nan,
        "sharpe_ratio": risk_metrics["sharpe_ratio"],
        "sortino_ratio": risk_metrics["sortino_ratio"],
        "max_drawdown": risk_metrics["max_drawdown"],
        "calmar_ratio": risk_metrics["calmar_ratio"],
    }


def run_walkforward_analysis(
    df: pd.DataFrame,
    signal_cols: Optional[list] = None,
    train_days: int = 180,
    val_days: int = 90,
    test_days: int = 95,
) -> pd.DataFrame:
    """
    Run walk-forward validation analysis.

    Evaluates signal performance across in-sample (train),
    validation, and out-of-sample (test) periods.

    Args:
        df: Full DataFrame with signals and returns.
        signal_cols: List of signal columns to evaluate.
        train_days: Days for training set.
        val_days: Days for validation set.
        test_days: Days for test set.

    Returns:
        DataFrame with results for each signal and period.
    """
    if signal_cols is None:
        signal_cols = SIGNAL_COLS

    # Split data
    splits = split_data_by_date(df, train_days, val_days, test_days)

    print(f"Data split:")
    print(f"  Training:   {splits['train_dates']} days")
    print(f"  Validation: {splits['val_dates']} days")
    print(f"  Test:       {splits['test_dates']} days")

    results = []

    for signal_col in signal_cols:
        if signal_col not in df.columns:
            continue

        # Evaluate on each split
        train_metrics = evaluate_on_split(splits["train"], signal_col)
        val_metrics = evaluate_on_split(splits["validation"], signal_col)
        test_metrics = evaluate_on_split(splits["test"], signal_col)

        # Add to results
        for period, metrics in [
            ("train", train_metrics),
            ("validation", val_metrics),
            ("test", test_metrics),
        ]:
            row = {"signal": signal_col, "period": period}
            row.update(metrics)
            results.append(row)

    result_df = pd.DataFrame(results)

    return result_df


def check_robustness(walkforward_df: pd.DataFrame) -> dict:
    """
    Check signal robustness across periods.

    A robust signal should:
    1. Have consistent sign of IC across periods
    2. Be statistically significant in at least one period
    3. Not deteriorate significantly from validation to test

    Args:
        walkforward_df: Output from run_walkforward_analysis.

    Returns:
        Dictionary with robustness assessment.
    """
    assessments = {}

    for signal in walkforward_df["signal"].unique():
        signal_df = walkforward_df[walkforward_df["signal"] == signal]

        # Get metrics for each period
        train_ic = signal_df[signal_df["period"] == "train"]["mean_ic"].values
        val_ic = signal_df[signal_df["period"] == "validation"]["mean_ic"].values
        test_ic = signal_df[signal_df["period"] == "test"]["mean_ic"].values

        train_sig = signal_df[signal_df["period"] == "train"]["significant_5pct"].values
        val_sig = signal_df[signal_df["period"] == "validation"]["significant_5pct"].values
        test_sig = signal_df[signal_df["period"] == "test"]["significant_5pct"].values

        # Check consistency of sign
        all_ics = []
        if len(train_ic) > 0 and not np.isnan(train_ic[0]):
            all_ics.append(train_ic[0])
        if len(val_ic) > 0 and not np.isnan(val_ic[0]):
            all_ics.append(val_ic[0])
        if len(test_ic) > 0 and not np.isnan(test_ic[0]):
            all_ics.append(test_ic[0])

        sign_consistent = all(ic > 0 for ic in all_ics) or all(ic < 0 for ic in all_ics)

        # Check significance
        any_significant = any([
            len(train_sig) > 0 and train_sig[0],
            len(val_sig) > 0 and val_sig[0],
            len(test_sig) > 0 and test_sig[0],
        ])

        # Check deterioration
        deterioration = None
        if len(val_ic) > 0 and len(test_ic) > 0:
            if not np.isnan(val_ic[0]) and not np.isnan(test_ic[0]) and val_ic[0] != 0:
                deterioration = (val_ic[0] - test_ic[0]) / abs(val_ic[0])

        assessments[signal] = {
            "sign_consistent": sign_consistent,
            "any_significant": any_significant,
            "deterioration": deterioration,
            "robust": sign_consistent and any_significant,
        }

    return assessments


if __name__ == "__main__":
    # Example usage
    from config import PROCESSED_PATH

    df = pd.read_csv(PROCESSED_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print("Running walk-forward analysis...")
    results = run_walkforward_analysis(df)
    print("\nResults:")
    print(results)

    print("\nRobustness check:")
    robustness = check_robustness(results)
    for signal, assessment in robustness.items():
        print(f"\n{signal}:")
        for key, value in assessment.items():
            print(f"  {key}: {value}")
