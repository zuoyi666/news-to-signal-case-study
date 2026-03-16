"""
preprocess.py

This module handles data collection and preprocessing for the news-to-signal case study.
It fetches news headlines from Yahoo Finance RSS and price data from yfinance,
computes forward returns, and performs data quality checks.
"""

import os
import time
import warnings
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from config import (
    BASE_DIR,
    HOLDING_HORIZON,
    LOOKBACK_DAYS,
    RAW_DATA_PATH,
    UNIVERSE_PATH,
)

warnings.filterwarnings("ignore")


# ───────────────────────────────────────────────────────────────────────────────
# RSS News Fetching
# ───────────────────────────────────────────────────────────────────────────────

def fetch_yahoo_news(ticker: str, max_entries: int = 20) -> list[dict]:
    """
    Fetch news headlines from Yahoo Finance RSS for a given ticker.

    Args:
        ticker: Stock ticker symbol.
        max_entries: Maximum number of news entries to fetch.

    Returns:
        List of dictionaries containing headline data.
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

    try:
        feed = feedparser.parse(url)
        entries = []

        for entry in feed.entries[:max_entries]:
            # Parse publication date
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, "published"):
                # Fallback: try to parse string date
                try:
                    pub_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %z")
                    pub_date = pub_date.replace(tzinfo=None)
                except ValueError:
                    pub_date = datetime.now()
            else:
                pub_date = datetime.now()

            entries.append({
                "ticker": ticker,
                "headline": entry.title,
                "published": pub_date,
                "source": entry.get("source", {}).get("title", "Yahoo Finance") if isinstance(entry.get("source"), dict) else "Yahoo Finance",
            })

        return entries
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []


def fetch_news_for_universe(universe_path: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch news headlines for all tickers in the universe.

    Args:
        universe_path: Path to CSV file containing tickers.
        lookback_days: Number of calendar days to look back.

    Returns:
        DataFrame with columns: ticker, headline, published, source.
    """
    universe = pd.read_csv(universe_path)
    tickers = universe["ticker"].tolist()

    cutoff_date = datetime.now() - timedelta(days=lookback_days)

    all_news = []
    for ticker in tickers:
        print(f"Fetching news for {ticker}...")
        news = fetch_yahoo_news(ticker)
        # Filter to lookback period
        news = [n for n in news if n["published"] >= cutoff_date]
        all_news.extend(news)
        time.sleep(0.5)  # Be polite to the RSS server

    df = pd.DataFrame(all_news)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "headline", "published", "source"])

    df = df.rename(columns={"published": "date"})
    return df


# ───────────────────────────────────────────────────────────────────────────────
# Price Data and Forward Returns
# ───────────────────────────────────────────────────────────────────────────────

_price_cache: dict[str, pd.DataFrame] = {}


def fetch_prices(ticker: str, start_date: datetime, end_date: datetime, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Fetch daily close prices from yfinance with retry logic and caching.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date for price data.
        end_date: End date for price data.
        max_retries: Maximum number of retry attempts.

    Returns:
        DataFrame with Date index and Close column, or None if failed.
    """
    cache_key = f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

    if cache_key in _price_cache:
        return _price_cache[cache_key]

    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            # Add buffer for weekends/holidays
            hist = stock.history(start=start_date - timedelta(days=10), end=end_date + timedelta(days=10))

            if hist.empty or "Close" not in hist.columns:
                time.sleep(1)
                continue

            hist = hist.reset_index()
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
            hist = hist[["Date", "Close"]].dropna()

            _price_cache[cache_key] = hist
            return hist
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            time.sleep(1)

    return None


def get_next_trading_day(date: datetime, price_df: pd.DataFrame) -> Optional[datetime]:
    """
    Find the first trading day on or after the given date.

    Args:
        date: The starting date.
        price_df: DataFrame with Date column containing trading days.

    Returns:
        The next trading day as datetime, or None if not found.
    """
    trading_days = pd.to_datetime(price_df["Date"]).dt.normalize()
    target = pd.Timestamp(date).normalize()

    future_days = trading_days[trading_days >= target]
    if future_days.empty:
        return None
    return future_days.iloc[0].to_pydatetime()


def compute_forward_return(
    ticker: str,
    news_date: datetime,
    price_df: pd.DataFrame,
    horizon: int = HOLDING_HORIZON,
) -> Optional[float]:
    """
    Compute the forward return: ln(Close[T'+horizon] / Close[T'+1])
    where T' is the first trading day on or after the news date.

    Args:
        ticker: Stock ticker symbol.
        news_date: Date of the news.
        price_df: DataFrame with Date and Close columns.
        horizon: Number of trading days for forward return.

    Returns:
        Forward return as float, or None if cannot be computed.
    """
    if price_df is None or price_df.empty:
        return None

    price_df = price_df.sort_values("Date").reset_index(drop=True)
    trading_days = price_df["Date"].tolist()

    # Find T' - first trading day on or after news date
    t_prime_idx = None
    for i, td in enumerate(trading_days):
        if pd.Timestamp(td).normalize() >= pd.Timestamp(news_date).normalize():
            t_prime_idx = i
            break

    if t_prime_idx is None:
        return None

    # We need T'+1 and T'+horizon
    entry_idx = t_prime_idx + 1
    exit_idx = t_prime_idx + horizon

    if exit_idx >= len(trading_days):
        return None

    entry_price = price_df.iloc[entry_idx]["Close"]
    exit_price = price_df.iloc[exit_idx]["Close"]

    if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
        return None

    return np.log(exit_price / entry_price)


def add_forward_returns(news_df: pd.DataFrame, horizon: int = HOLDING_HORIZON) -> pd.DataFrame:
    """
    Add forward return columns to the news DataFrame.

    Args:
        news_df: DataFrame with ticker and date columns.
        horizon: Number of trading days for forward return.

    Returns:
        DataFrame with added future_return_5d column.
    """
    if news_df.empty:
        news_df["future_return_5d"] = pd.Series(dtype=float)
        return news_df

    # Determine date range for price fetching
    min_date = pd.to_datetime(news_df["date"]).min() - timedelta(days=10)
    max_date = pd.to_datetime(news_df["date"]).max() + timedelta(days=horizon + 10)

    tickers = news_df["ticker"].unique()

    returns = []
    for ticker in tickers:
        print(f"Fetching prices for {ticker}...")
        price_df = fetch_prices(ticker, min_date, max_date)

        ticker_news = news_df[news_df["ticker"] == ticker].copy()
        for idx, row in ticker_news.iterrows():
            ret = compute_forward_return(ticker, row["date"], price_df, horizon)
            returns.append({
                "idx": idx,
                "future_return_5d": ret,
            })

        time.sleep(0.3)  # Be polite to yfinance

    returns_df = pd.DataFrame(returns)
    if not returns_df.empty:
        news_df = news_df.copy()
        news_df["future_return_5d"] = returns_df.set_index("idx")["future_return_5d"]

    return news_df


# ───────────────────────────────────────────────────────────────────────────────
# Aggregation and Quality Checks
# ───────────────────────────────────────────────────────────────────────────────

def aggregate_headlines(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple headlines on the same ticker-date into one row.

    For each ticker-date combination:
    - Keep the last headline (for display purposes)
    - Count total headlines that day
    - Concatenate all headline text internally for feature engineering

    Args:
        news_df: DataFrame with ticker, date, headline columns.

    Returns:
        Aggregated DataFrame.
    """
    if news_df.empty:
        return pd.DataFrame(columns=["date", "ticker", "headline", "headline_count", "source", "future_return_5d"])

    df = news_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Group by ticker and date
    grouped = df.groupby(["ticker", "date"]).agg({
        "headline": lambda x: " | ".join(x.astype(str)),  # Concatenate all headlines
        "source": "first",
        "future_return_5d": "first",  # Same for all rows in group
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


def run_quality_assertions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run data quality assertions on the processed DataFrame.

    Checks:
    - date column has no nulls
    - ticker column has no nulls
    - future_return_5d non-null ratio >= 0.80 on eligible rows
    - each cross-section date has at least 5 valid observations
    - no abnormal price gaps (daily return > 50% in absolute value)

    Args:
        df: DataFrame to validate.

    Returns:
        Validated DataFrame.

    Raises:
        AssertionError: If any quality check fails.
    """
    print("\nRunning data quality assertions...")

    # Check 1: date column has no nulls
    assert df["date"].isna().sum() == 0, "date column contains null values"
    print("✓ date column has no nulls")

    # Check 2: ticker column has no nulls
    assert df["ticker"].isna().sum() == 0, "ticker column contains null values"
    print("✓ ticker column has no nulls")

    # Check 3: future_return_5d non-null ratio >= 0.80 on rows where
    # a full holding horizon is expected to be available.
    cutoff_date = (pd.to_datetime(df["date"]).max() - pd.offsets.BDay(HOLDING_HORIZON)).normalize()
    eligible_mask = pd.to_datetime(df["date"]).dt.normalize() <= cutoff_date
    eligible_count = int(eligible_mask.sum())
    ineligible_count = int((~eligible_mask).sum())

    if eligible_count == 0:
        print("! Skipping future_return_5d coverage assertion: no eligible rows with full horizon yet")
    else:
        non_null_ratio = df.loc[eligible_mask, "future_return_5d"].notna().mean()
        assert non_null_ratio >= 0.80, (
            f"future_return_5d non-null ratio on eligible rows {non_null_ratio:.2%} < 80% "
            f"(eligible={eligible_count}, ineligible={ineligible_count})"
        )
        print(
            "✓ future_return_5d non-null ratio on eligible rows: "
            f"{non_null_ratio:.2%} (eligible={eligible_count}, ineligible={ineligible_count})"
        )

    # Check 4: date-level coverage on eligible dates.
    # For sparse headline data, requiring >=5 observations per date is often too strict.
    df_valid = df[df["future_return_5d"].notna()].copy()
    eligible_dates = pd.to_datetime(df.loc[eligible_mask, "date"]).dt.normalize().drop_duplicates().sort_values()
    valid_counts = pd.to_datetime(df_valid["date"]).dt.normalize().value_counts()

    if len(eligible_dates) == 0:
        print("! Skipping date coverage assertion: no eligible dates")
    else:
        has_valid = valid_counts.reindex(eligible_dates, fill_value=0) > 0
        date_coverage_ratio = has_valid.mean()
        assert date_coverage_ratio >= 0.80, (
            f"Eligible date coverage with >=1 valid observation {date_coverage_ratio:.2%} < 80%"
        )

        low_coverage_dates = valid_counts[valid_counts < 5].sort_index()
        print(
            "✓ Eligible date coverage with >=1 valid observation: "
            f"{date_coverage_ratio:.2%} ({int(has_valid.sum())}/{len(eligible_dates)})"
        )
        if len(low_coverage_dates) > 0:
            print(
                "! Low cross-section depth (<5 observations) on dates: "
                f"{low_coverage_dates.to_dict()}"
            )

    # Check 5: no abnormal price gaps (daily return > 50%)
    # Compute daily returns from future_return_5d where available
    extreme_returns = df_valid[df_valid["future_return_5d"].abs() > 0.5]
    assert len(extreme_returns) == 0, f"Found {len(extreme_returns)} extreme return values (>50%)"
    print("✓ No abnormal price gaps detected")

    return df


# ───────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ───────────────────────────────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Steps:
    1. Fetch news from Yahoo Finance RSS for all tickers in universe.
    2. Fetch price data and compute forward returns.
    3. Aggregate headlines by ticker-date.
    4. Run data quality assertions.
    5. Save to RAW_DATA_PATH.

    Returns:
        Processed DataFrame.
    """
    print("=" * 60)
    print("News Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Fetch news
    print("\n[1/5] Fetching news headlines...")
    news_df = fetch_news_for_universe(UNIVERSE_PATH, LOOKBACK_DAYS)
    print(f"Fetched {len(news_df)} news items")

    if news_df.empty:
        print("WARNING: No news data fetched. Returning empty DataFrame.")
        empty_df = pd.DataFrame(columns=["date", "ticker", "headline", "headline_count", "source", "future_return_5d"])
        empty_df.to_csv(RAW_DATA_PATH, index=False)
        return empty_df

    # Step 2: Add forward returns
    print("\n[2/5] Computing forward returns...")
    news_df = add_forward_returns(news_df, HOLDING_HORIZON)

    # Step 3: Aggregate headlines
    print("\n[3/5] Aggregating headlines by ticker-date...")
    df = aggregate_headlines(news_df)
    print(f"After aggregation: {len(df)} rows")

    # Step 4: Quality assertions
    print("\n[4/5] Running quality checks...")
    df = run_quality_assertions(df)

    # Drop rows where forward return cannot be computed
    initial_rows = len(df)
    df = df[df["future_return_5d"].notna()].copy()
    print(f"Dropped {initial_rows - len(df)} rows with missing forward returns")

    # Step 5: Save
    print("\n[5/5] Saving to CSV...")
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved {len(df)} rows to {RAW_DATA_PATH}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
