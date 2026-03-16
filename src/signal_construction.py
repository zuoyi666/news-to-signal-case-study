"""
signal_construction.py

This module handles signal construction for the news-to-signal case study.
It applies cross-sectional z-score standardization and constructs
three distinct signals from the engineered features.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from config import PROCESSED_PATH, SIGNAL_COLS


def cross_sectional_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Compute cross-sectional z-score for a column within each trading date.

    For each date, computes: z = (value - mean) / std
    If std == 0 for a date, sets z-score to 0 for that date.

    Args:
        df: DataFrame with 'date' column and the target column.
        col: Name of the column to z-score.

    Returns:
        Series with z-scores, indexed like df.
    """
    def zscore_group(group):
        values = group[col]
        mean = values.mean()
        std = values.std()

        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=group.index)

        return (values - mean) / std

    return df.groupby("date", group_keys=False).apply(zscore_group)


def add_signal_sentiment_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add signal_sentiment_only = z(sentiment_score).

    Args:
        df: DataFrame with sentiment_score column.

    Returns:
        DataFrame with added signal_sentiment_only column.
    """
    df = df.copy()
    df["signal_sentiment_only"] = cross_sectional_zscore(df, "sentiment_score")
    return df


def add_signal_sentiment_minus_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add signal_sentiment_minus_uncertainty = z(sentiment_score) - z(uncertainty_score).

    Args:
        df: DataFrame with sentiment_score and uncertainty_score columns.

    Returns:
        DataFrame with added signal_sentiment_minus_uncertainty column.
    """
    df = df.copy()
    z_sentiment = cross_sectional_zscore(df, "sentiment_score")
    z_uncertainty = cross_sectional_zscore(df, "uncertainty_score")
    df["signal_sentiment_minus_uncertainty"] = z_sentiment - z_uncertainty
    return df


def add_signal_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add signal_full = z(sentiment_score) - z(uncertainty_score) + z(event_intensity).

    Args:
        df: DataFrame with sentiment_score, uncertainty_score, and event_intensity columns.

    Returns:
        DataFrame with added signal_full column.
    """
    df = df.copy()
    z_sentiment = cross_sectional_zscore(df, "sentiment_score")
    z_uncertainty = cross_sectional_zscore(df, "uncertainty_score")
    z_event = cross_sectional_zscore(df, "event_intensity")
    df["signal_full"] = z_sentiment - z_uncertainty + z_event
    return df


def main(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Run the full signal construction pipeline.

    Steps:
    1. Load data if not provided.
    2. Compute cross-sectional z-scores for each feature.
    3. Construct three signals:
       - signal_sentiment_only = z(sentiment_score)
       - signal_sentiment_minus_uncertainty = z(sentiment_score) - z(uncertainty_score)
       - signal_full = z(sentiment_score) - z(uncertainty_score) + z(event_intensity)
    4. Save final feature table to PROCESSED_PATH.

    Args:
        df: Input DataFrame. If None, will be loaded from RAW_DATA_PATH
            and feature engineering will be applied.

    Returns:
        DataFrame with all signals constructed.
    """
    from config import RAW_DATA_PATH
    from src.feature_engineering import main as feature_main

    if df is None:
        print("Loading raw data and applying feature engineering...")
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(
                f"Raw data not found at {RAW_DATA_PATH}. "
                "Please run preprocess.py first."
            )
        df = feature_main()

    print("Constructing signals...")

    # Ensure required columns exist
    required_cols = ["sentiment_score", "uncertainty_score", "event_intensity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Construct signals
    df = add_signal_sentiment_only(df)
    df = add_signal_sentiment_minus_uncertainty(df)
    df = add_signal_full(df)

    # Save to processed path
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved processed data to {PROCESSED_PATH}")

    print(f"\nSignal construction complete. Shape: {df.shape}")
    print(f"Signal columns: {SIGNAL_COLS}")

    # Print summary statistics
    print("\nSignal Summary Statistics:")
    print(df[SIGNAL_COLS].describe())

    return df


if __name__ == "__main__":
    df = main()
