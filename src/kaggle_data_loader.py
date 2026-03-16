"""
kaggle_data_loader.py

Load historical financial news data from Kaggle datasets.
This provides a way to get historical news without API rate limits.

Recommended datasets:
1. "Financial News Headlines" - various sources
2. "Stock Market News with Sentiment" - labeled data
3. "Daily Financial News for 6000+ Stocks" - comprehensive coverage

To use:
1. Download dataset from Kaggle
2. Place CSV in data/external/
3. Run this script to format for the pipeline
"""

import os
from datetime import datetime, timedelta

import pandas as pd


def load_kaggle_sentiment_dataset(filepath: str) -> pd.DataFrame:
    """
    Load and format Kaggle financial sentiment dataset.

    Expected columns (standard format):
    - ticker: Stock symbol
    - headline: News headline text
    - date: Publication date
    - sentiment: Optional pre-labeled sentiment

    Args:
        filepath: Path to Kaggle CSV file

    Returns:
        Formatted DataFrame matching pipeline schema
    """
    if not os.path.exists(filepath):
        print(f"Kaggle dataset not found at {filepath}")
        print("Download from: https://www.kaggle.com/datasets?search=financial+news")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from Kaggle dataset")
    print(f"Columns: {df.columns.tolist()}")

    # Standardize column names (handle common variations)
    column_mapping = {
        # Ticker variations
        'symbol': 'ticker',
        'stock': 'ticker',
        'ticker_symbol': 'ticker',
        # Headline variations
        'title': 'headline',
        'text': 'headline',
        'news': 'headline',
        'content': 'headline',
        # Date variations
        'published': 'date',
        'timestamp': 'date',
        'datetime': 'date',
        'published_at': 'date',
        # Source variations
        'publisher': 'source',
        'provider': 'source',
    }

    df = df.rename(columns=column_mapping)

    # Ensure required columns exist
    required = ['ticker', 'headline', 'date']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return pd.DataFrame()

    # Standardize date format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Add source if not present
    if 'source' not in df.columns:
        df['source'] = 'Kaggle'

    # Add placeholder columns for pipeline compatibility
    df['headline_count'] = 1
    df['future_return_5d'] = None  # Will be computed later

    # Reorder columns
    cols = ['date', 'ticker', 'headline', 'headline_count', 'source', 'future_return_5d']
    df = df[[c for c in cols if c in df.columns]]

    print(f"\nFormatted dataset:")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Tickers: {df['ticker'].nunique()}")

    return df


def generate_synthetic_historical_data(
    tickers: list,
    start_date: str = "2024-01-01",
    end_date: str = "2025-03-01",
    avg_news_per_day: int = 3,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic historical news data for testing expanded pipeline.

    This creates realistic-looking data with:
    - Variable news frequency per ticker
    - Some dates with no news
    - Plausible headline lengths

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        avg_news_per_day: Average news items per ticker per day
        random_seed: For reproducibility

    Returns:
        Synthetic news DataFrame
    """
    import numpy as np

    np.random.seed(random_seed)

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Sample headline templates
    templates = [
        "{ticker} reports earnings beat, revenue up {pct}%",
        "Analyst upgrades {ticker} to buy, price target ${price}",
        "{ticker} announces partnership with tech giant",
        "Market volatility hits {ticker} shares",
        "{ticker} CEO discusses growth strategy at conference",
        "Institutional investors increase stake in {ticker}",
        "{ticker} faces regulatory scrutiny over operations",
        "Supply chain concerns weigh on {ticker} outlook",
        "{ticker} expands into new geographic markets",
        "Competition intensifies for {ticker} in key segment",
    ]

    all_news = []

    for ticker in tickers:
        # Each ticker has different news frequency
        ticker_frequency = np.random.poisson(avg_news_per_day)

        for date in date_range:
            # Random chance of news on this date
            n_news = np.random.poisson(ticker_frequency * 0.3)  # Scale down

            for _ in range(n_news):
                template = np.random.choice(templates)
                headline = template.format(
                    ticker=ticker,
                    pct=np.random.randint(5, 50),
                    price=np.random.randint(50, 500)
                )

                all_news.append({
                    'date': date,
                    'ticker': ticker,
                    'headline': headline,
                    'headline_count': 1,
                    'source': np.random.choice(['Reuters', 'Bloomberg', 'CNBC', 'WSJ']),
                    'future_return_5d': None
                })

    df = pd.DataFrame(all_news)

    print(f"Generated synthetic data:")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Tickers: {df['ticker'].nunique()}")
    print(f"  Avg news per ticker: {len(df) / len(tickers):.1f}")

    return df


def merge_with_historical_prices(news_df: pd.DataFrame, prices_dir: str) -> pd.DataFrame:
    """
    Merge news data with historical prices to compute forward returns.

    Args:
        news_df: News DataFrame
        prices_dir: Directory containing historical price CSVs

    Returns:
        News DataFrame with forward returns
    """
    import numpy as np

    if news_df.empty:
        return news_df

    news_df = news_df.copy()
    news_df['future_return_5d'] = np.nan

    tickers = news_df['ticker'].unique()

    for ticker in tickers:
        price_path = os.path.join(prices_dir, f"{ticker}.csv")

        if not os.path.exists(price_path):
            continue

        prices = pd.read_csv(price_path, parse_dates=['Date'])
        prices = prices.sort_values('Date')

        # Compute 5-day forward returns
        prices['future_return_5d'] = np.log(
            prices['Close'].shift(-6) / prices['Close'].shift(-1)
        )

        # Merge with news
        ticker_news = news_df[news_df['ticker'] == ticker].copy()
        for idx, row in ticker_news.iterrows():
            news_date = pd.Timestamp(row['date'])

            # Find closest trading day
            mask = prices['Date'] >= news_date
            if mask.any():
                future_ret = prices.loc[mask, 'future_return_5d'].iloc[0]
                news_df.loc[idx, 'future_return_5d'] = future_ret

    coverage = news_df['future_return_5d'].notna().mean()
    print(f"\nForward return coverage: {coverage:.1%}")

    return news_df


if __name__ == "__main__":
    # Example usage
    print("Kaggle Data Loader - Example Usage\n")

    # Option 1: Try to load real Kaggle data
    kaggle_path = "data/external/kaggle_financial_news.csv"
    if os.path.exists(kaggle_path):
        df = load_kaggle_sentiment_dataset(kaggle_path)
    else:
        print(f"No Kaggle data found at {kaggle_path}")
        print("Generating synthetic data for testing...\n")

        # Option 2: Generate synthetic data
        from config import UNIVERSE_PATH

        universe = pd.read_csv(UNIVERSE_PATH)
        tickers = universe['ticker'].tolist()[:50]  # Use first 50

        df = generate_synthetic_historical_data(
            tickers=tickers,
            start_date="2024-01-01",
            end_date="2025-03-01",
            avg_news_per_day=2
        )

        # Save for pipeline use
        output_path = "data/raw/news_sample_synthetic.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved synthetic data to {output_path}")
