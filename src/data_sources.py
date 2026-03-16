"""
data_sources.py

Multi-source data aggregation module for news and price data.
Combines free-tier APIs: Finnhub, FMP, yfinance with fallback logic.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import pandas as pd
import requests
import yfinance as yf

# API Keys from environment (user should set these)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")


class NewsAggregator:
    """
    Aggregate news from multiple free sources with fallback logic.
    Priority: Finnhub > FMP > Yahoo RSS
    """

    def __init__(self, rate_limit_delay: float = 1.0):
        self.rate_limit_delay = rate_limit_delay
        self.finnhub_base = "https://finnhub.io/api/v1"
        self.fmp_base = "https://financialmodelingprep.com/api/v3"

    def fetch_finnhub_news(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        """Fetch news from Finnhub (60 calls/minute free tier)."""
        if not FINNHUB_API_KEY:
            return []

        url = f"{self.finnhub_base}/company-news"
        params = {
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
            "token": FINNHUB_API_KEY,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                news_items = []
                for item in data:
                    news_items.append({
                        "ticker": ticker,
                        "headline": item.get("headline", ""),
                        "published": datetime.fromtimestamp(item.get("datetime", 0)),
                        "source": item.get("source", "Finnhub"),
                        "url": item.get("url", ""),
                        "summary": item.get("summary", ""),
                    })
                return news_items
            else:
                print(f"Finnhub error for {ticker}: {response.status_code}")
                return []
        except Exception as e:
            print(f"Finnhub exception for {ticker}: {e}")
            return []

    def fetch_fmp_news(self, ticker: str, limit: int = 50) -> list[dict]:
        """Fetch news from FMP (250 calls/day free tier)."""
        if not FMP_API_KEY:
            return []

        url = f"{self.fmp_base}/stock_news"
        params = {"tickers": ticker, "limit": limit, "apikey": FMP_API_KEY}

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                news_items = []
                for item in data:
                    news_items.append({
                        "ticker": ticker,
                        "headline": item.get("title", ""),
                        "published": datetime.strptime(
                            item.get("publishedDate", "2024-01-01"),
                            "%Y-%m-%dT%H:%M:%S.%fZ",
                        ),
                        "source": item.get("site", "FMP"),
                        "url": item.get("url", ""),
                        "text": item.get("text", ""),
                    })
                return news_items
            else:
                print(f"FMP error for {ticker}: {response.status_code}")
                return []
        except Exception as e:
            print(f"FMP exception for {ticker}: {e}")
            return []

    def fetch_yahoo_rss(self, ticker: str, max_entries: int = 20) -> list[dict]:
        """Fetch news from Yahoo Finance RSS (fallback)."""
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

        try:
            feed = feedparser.parse(url)
            entries = []
            for entry in feed.entries[:max_entries]:
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, "published"):
                    try:
                        pub_date = datetime.strptime(
                            entry.published, "%a, %d %b %Y %H:%M:%S %z"
                        )
                        pub_date = pub_date.replace(tzinfo=None)
                    except ValueError:
                        pub_date = datetime.now()
                else:
                    pub_date = datetime.now()

                entries.append({
                    "ticker": ticker,
                    "headline": entry.title,
                    "published": pub_date,
                    "source": "Yahoo Finance",
                    "url": entry.get("link", ""),
                })
            return entries
        except Exception as e:
            print(f"Yahoo RSS error for {ticker}: {e}")
            return []

    def fetch_all_sources(
        self, ticker: str, from_date: str, to_date: str
    ) -> pd.DataFrame:
        """
        Fetch news from all available sources and merge.
        Deduplicates by headline similarity.
        """
        all_news = []

        # Try Finnhub first (best quality)
        if FINNHUB_API_KEY:
            print(f"  Fetching Finnhub news for {ticker}...")
            finnhub_news = self.fetch_finnhub_news(ticker, from_date, to_date)
            all_news.extend(finnhub_news)
            time.sleep(self.rate_limit_delay)

        # Try FMP second
        if FMP_API_KEY and len(all_news) < 10:  # Only if Finnhub returned few results
            print(f"  Fetching FMP news for {ticker}...")
            fmp_news = self.fetch_fmp_news(ticker)
            all_news.extend(fmp_news)
            time.sleep(self.rate_limit_delay)

        # Fallback to Yahoo RSS
        if len(all_news) < 5:
            print(f"  Fetching Yahoo RSS for {ticker}...")
            yahoo_news = self.fetch_yahoo_rss(ticker)
            all_news.extend(yahoo_news)
            time.sleep(0.5)

        if not all_news:
            return pd.DataFrame(
                columns=["ticker", "headline", "published", "source", "url"]
            )

        df = pd.DataFrame(all_news)
        df["date"] = pd.to_datetime(df["published"]).dt.normalize()

        # Deduplicate similar headlines (same date + similar text)
        df = df.drop_duplicates(subset=["ticker", "date", "headline"], keep="first")

        return df


class PriceDataAggregator:
    """
    Aggregate price data from multiple sources.
    Primary: yfinance (most reliable for historical)
    Fallback: FMP (for current quotes)
    """

    def __init__(self):
        self.cache = {}

    def fetch_yfinance(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch prices from yfinance with caching."""
        cache_key = f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(
                start=start_date - timedelta(days=10),
                end=end_date + timedelta(days=10),
            )

            if hist.empty or "Close" not in hist.columns:
                return None

            hist = hist.reset_index()
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
            hist = hist[["Date", "Close", "Volume"]].dropna()

            self.cache[cache_key] = hist
            return hist
        except Exception as e:
            print(f"yfinance error for {ticker}: {e}")
            return None

    def fetch_fmp_prices(
        self, ticker: str, from_date: str, to_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch prices from FMP as fallback."""
        if not FMP_API_KEY:
            return None

        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
        params = {"from": from_date, "to": to_date, "apikey": FMP_API_KEY}

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "historical" in data:
                    df = pd.DataFrame(data["historical"])
                    df["Date"] = pd.to_datetime(df["date"])
                    df = df.rename(columns={"close": "Close", "volume": "Volume"})
                    return df[["Date", "Close", "Volume"]]
            return None
        except Exception as e:
            print(f"FMP prices error for {ticker}: {e}")
            return None

    def fetch_prices(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch prices with fallback logic."""
        # Try yfinance first
        df = self.fetch_yfinance(ticker, start_date, end_date)
        if df is not None and len(df) > 0:
            return df

        # Fallback to FMP
        df = self.fetch_fmp_prices(
            ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )
        return df


if __name__ == "__main__":
    # Test the aggregators
    print("Testing NewsAggregator...")
    news_agg = NewsAggregator()

    # Test with AAPL
    test_news = news_agg.fetch_all_sources(
        "AAPL",
        (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        datetime.now().strftime("%Y-%m-%d"),
    )
    print(f"Fetched {len(test_news)} news items for AAPL")
    if len(test_news) > 0:
        print(test_news.head())
