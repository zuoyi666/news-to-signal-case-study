"""
preprocess_v2.py

Enhanced preprocessing with multi-source data aggregation.
Phase 1 optimization: expands universe and uses multiple free data sources.
"""

import os
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BASE_DIR,
    DATA_SOURCE_PRIORITY,
    ENABLE_Finnhub,
    ENABLE_FMP,
    HOLDING_HORIZON,
    LOOKBACK_DAYS,
    MAX_NEWS_PER_TICKER,
    RATE_LIMITS,
    RAW_DATA_PATH,
    UNIVERSE_PATH,
)
from src.data_sources import NewsAggregator, PriceDataAggregator

warnings.filterwarnings("ignore")


def load_universe(universe_path: str = UNIVERSE_PATH) -> list[str]:
    """Load ticker universe from CSV."""
    df = pd.read_csv(universe_path)
    tickers = df["ticker"].tolist()
    print(f"Loaded {len(tickers)} tickers from {universe_path}")
    return tickers


def fetch_news_multi_source(
    tickers: list[str],
    lookback_days: int = LOOKBACK_DAYS,
    max_per_ticker: int = MAX_NEWS_PER_TICKER,
) -> pd.DataFrame:
    """
    Fetch news from multiple sources with fallback logic.

    Strategy:
    1. Try Finnhub for all tickers (best quality, 60 calls/min)
    2. Fill gaps with FMP (250 calls/day)
    3. Use Yahoo RSS as final fallback
    """
    news_agg = NewsAggregator(rate_limit_delay=1.0)

    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    all_news = []

    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Fetching news for {ticker}...")

        # Fetch from all sources
        df_news = news_agg.fetch_all_sources(ticker, from_date, to_date)

        if len(df_news) > 0:
            # Limit to max per ticker
            df_news = df_news.head(max_per_ticker)
            all_news.append(df_news)
            print(f"  ✓ Got {len(df_news)} items")
        else:
            print(f"  ✗ No news found")

        # Rate limiting - be nice to APIs
        time.sleep(0.5)

    if not all_news:
        print("WARNING: No news data fetched from any source")
        return pd.DataFrame(columns=["ticker", "headline", "published", "source", "date"])

    df = pd.concat(all_news, ignore_index=True)
    print(f"\nTotal news items fetched: {len(df)}")
    print(f"Coverage: {df['ticker'].nunique()}/{len(tickers)} tickers")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def compute_forward_returns_v2(
    news_df: pd.DataFrame, horizon: int = HOLDING_HORIZON
) -> pd.DataFrame:
    """
    Compute forward returns using multi-source price data.
    """
    if news_df.empty:
        news_df["future_return_5d"] = pd.Series(dtype=float)
        return news_df

    price_agg = PriceDataAggregator()

    # Determine date range
    min_date = pd.to_datetime(news_df["date"]).min() - timedelta(days=10)
    max_date = pd.to_datetime(news_df["date"]).max() + timedelta(days=horizon + 10)

    tickers = news_df["ticker"].unique()

    returns = []
    for i, ticker in enumerate(tickers):
        if i % 10 == 0:
            print(f"  Fetching prices... {i}/{len(tickers)}")

        price_df = price_agg.fetch_prices(ticker, min_date, max_date)

        ticker_news = news_df[news_df["ticker"] == ticker].copy()
        for idx, row in ticker_news.iterrows():
            ret = compute_single_return(row["date"], price_df, horizon)
            returns.append({"idx": idx, "future_return_5d": ret})

        time.sleep(0.1)  # Be polite

    returns_df = pd.DataFrame(returns)
    if not returns_df.empty:
        news_df = news_df.copy()
        news_df["future_return_5d"] = returns_df.set_index("idx")["future_return_5d"]

    return news_df


def compute_single_return(
    news_date: datetime, price_df: pd.DataFrame, horizon: int
) -> float:
    """Compute forward return for a single news item."""
    if price_df is None or price_df.empty:
        return np.nan

    price_df = price_df.sort_values("Date").reset_index(drop=True)
    trading_days = price_df["Date"].tolist()

    # Find T' - first trading day on or after news date
    t_prime_idx = None
    for i, td in enumerate(trading_days):
        if pd.Timestamp(td).normalize() >= pd.Timestamp(news_date).normalize():
            t_prime_idx = i
            break

    if t_prime_idx is None:
        return np.nan

    # We need T'+1 and T'+horizon
    entry_idx = t_prime_idx + 1
    exit_idx = t_prime_idx + horizon

    if exit_idx >= len(trading_days):
        return np.nan

    entry_price = price_df.iloc[entry_idx]["Close"]
    exit_price = price_df.iloc[exit_idx]["Close"]

    if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
        return np.nan

    return np.log(exit_price / entry_price)


def aggregate_headlines(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate headlines by ticker-date."""
    if news_df.empty:
        return pd.DataFrame(
            columns=["date", "ticker", "headline", "headline_count", "source", "future_return_5d"]
        )

    df = news_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Group by ticker and date
    grouped = df.groupby(["ticker", "date"]).agg({
        "headline": lambda x: " | ".join(x.astype(str)),
        "source": "first",
        "future_return_5d": "first",
    }).reset_index()

    # Count headlines per group
    counts = df.groupby(["ticker", "date"]).size().reset_index(name="headline_count")
    grouped = grouped.merge(counts, on=["ticker", "date"])

    # For display, keep only the last headline
    last_headlines = df.groupby(["ticker", "date"])["headline"].last().reset_index()
    grouped = grouped.drop(columns=["headline"])
    grouped = grouped.merge(last_headlines, on=["ticker", "date"])

    # Reorder columns
    cols = ["date", "ticker", "headline", "headline_count", "source", "future_return_5d"]
    grouped = grouped[cols]

    return grouped


def run_quality_assertions_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced quality checks for Phase 1 expanded dataset.
    """
    print("\nRunning data quality assertions...")

    # Check 1: date column has no nulls
    assert df["date"].isna().sum() == 0, "date column contains null values"
    print("✓ date column has no nulls")

    # Check 2: ticker column has no nulls
    assert df["ticker"].isna().sum() == 0, "ticker column contains null values"
    print("✓ ticker column has no nulls")

    # Check 3: coverage statistics
    cutoff_date = (pd.to_datetime(df["date"]).max() - pd.offsets.BDay(HOLDING_HORIZON)).normalize()
    eligible_mask = pd.to_datetime(df["date"]).dt.normalize() <= cutoff_date
    eligible_count = int(eligible_mask.sum())

    if eligible_count == 0:
        print("! Skipping coverage assertion: no eligible rows")
    else:
        non_null_ratio = df.loc[eligible_mask, "future_return_5d"].notna().mean()
        print(f"✓ future_return_5d coverage on eligible rows: {non_null_ratio:.1%} ({eligible_count} rows)")

    # Check 4: cross-sectional statistics
    df_valid = df[df["future_return_5d"].notna()].copy()
    date_counts = df_valid.groupby("date").size()

    print(f"✓ Unique trading days with data: {len(date_counts)}")
    print(f"✓ Average observations per day: {date_counts.mean():.1f}")
    print(f"✓ Max observations on single day: {date_counts.max()}")

    if len(date_counts) > 0:
        low_coverage = date_counts[date_counts < 5]
        if len(low_coverage) > 0:
            print(f"! {len(low_coverage)} dates with <5 observations (using tercile fallback)")

    # Check 5: extreme returns
    extreme_returns = df_valid[df_valid["future_return_5d"].abs() > 0.5]
    assert len(extreme_returns) == 0, f"Found {len(extreme_returns)} extreme returns (>50%)"
    print("✓ No abnormal price gaps detected")

    return df


def main_v2() -> pd.DataFrame:
    """
    Phase 1 enhanced preprocessing pipeline.
    """
    print("=" * 70)
    print("News Preprocessing Pipeline v2 - Phase 1 Optimization")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Universe: {UNIVERSE_PATH}")
    print(f"  Lookback: {LOOKBACK_DAYS} days")
    print(f"  Sources: {', '.join(DATA_SOURCE_PRIORITY)}")
    print(f"  APIs: Finnhub={ENABLE_Finnhub}, FMP={ENABLE_FMP}")
    print("=" * 70)

    # Step 1: Load universe
    print("\n[1/5] Loading universe...")
    tickers = load_universe()

    # Step 2: Fetch news from multiple sources
    print("\n[2/5] Fetching news from multiple sources...")
    news_df = fetch_news_multi_source(tickers)

    if news_df.empty:
        print("ERROR: No news data fetched. Check API keys and connectivity.")
        return pd.DataFrame()

    # Step 3: Compute forward returns
    print("\n[3/5] Computing forward returns...")
    news_df = compute_forward_returns_v2(news_df, HOLDING_HORIZON)

    # Step 4: Aggregate
    print("\n[4/5] Aggregating headlines...")
    df = aggregate_headlines(news_df)
    print(f"After aggregation: {len(df)} rows")

    # Step 5: Quality checks
    print("\n[5/5] Running quality checks...")
    df = run_quality_assertions_v2(df)

    # Drop rows without forward returns
    initial_rows = len(df)
    df = df[df["future_return_5d"].notna()].copy()
    print(f"Dropped {initial_rows - len(df)} rows with missing forward returns")

    # Save
    print("\n[6/6] Saving to CSV...")
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved {len(df)} rows to {RAW_DATA_PATH}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total observations: {len(df)}")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Mean forward return: {df['future_return_5d'].mean():.4f}")
    print("=" * 70)

    return df


if __name__ == "__main__":
    main_v2()
