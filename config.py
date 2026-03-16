# config.py
# Central configuration for news-to-signal-case-study.
# All pipeline parameters are defined here to keep src modules clean and reusable.

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_PATH   = os.path.join(BASE_DIR, "data", "reference", "universe.csv")
LMD_DICT_PATH   = os.path.join(BASE_DIR, "data", "reference", "loughran_mcdonald.csv")
RAW_DATA_PATH   = os.path.join(BASE_DIR, "data", "raw", "news_sample.csv")
PROCESSED_PATH  = os.path.join(BASE_DIR, "data", "processed", "news_features.csv")
FIGURES_DIR     = os.path.join(BASE_DIR, "results", "figures")
TABLES_DIR      = os.path.join(BASE_DIR, "results", "tables")

# ── Data collection ───────────────────────────────────────────────────────────
LOOKBACK_DAYS    = 365
HOLDING_HORIZON  = 5   # number of trading days for future_return_5d

# ── Grouping fallback rules ────────────────────────────────────────────────────
# Applied per cross-section date based on number of valid observations N
MIN_SAMPLE_DROP     = 9    # N < 9:  drop date entirely
MIN_SAMPLE_TERCILE  = 15   # 9 <= N < 15: use tercile; N >= 15: use quintile

# ── Feature engineering ────────────────────────────────────────────────────────
EVENT_KEYWORDS = [
    "earnings", "guidance", "downgrade", "upgrade",
    "acquisition", "lawsuit", "partnership"
]

# ── Signal columns ────────────────────────────────────────────────────────────
SIGNAL_COLS = [
    "signal_sentiment_only",
    "signal_sentiment_minus_uncertainty",
    "signal_full"
]
