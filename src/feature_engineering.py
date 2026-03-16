"""
feature_engineering.py

This module handles feature engineering for the news-to-signal case study.
It uses the Loughran-McDonald Master Dictionary to compute sentiment scores
from financial news headlines.
"""

import re
import string
from typing import Optional

import pandas as pd
from config import EVENT_KEYWORDS, LMD_DICT_PATH


def load_lmd_dictionary(path: str = LMD_DICT_PATH) -> pd.DataFrame:
    """
    Load the Loughran-McDonald Master Dictionary from CSV.

    The dictionary contains word classifications including:
    - Positive: words indicating positive sentiment
    - Negative: words indicating negative sentiment
    - Uncertainty: words indicating uncertainty

    Args:
        path: Path to the LMD CSV file.

    Returns:
        DataFrame with the LMD dictionary.

    Raises:
        FileNotFoundError: If the dictionary file does not exist.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Loughran-McDonald dictionary not found at {path}.\n"
            "Please download the dictionary from:\n"
            "https://sraf.nd.edu/loughranmcdonald-master-dictionary/\n"
            "and place it at the above path."
        )

    return df


def parse_lmd_categories(lmd_df: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
    """
    Parse the LMD DataFrame into category word sets.

    Args:
        lmd_df: DataFrame with columns including Word, Positive, Negative, Uncertainty.

    Returns:
        Tuple of (positive_words, negative_words, uncertainty_words).
    """
    # A word belongs to a category if that column value > 0
    positive_words = set(lmd_df[lmd_df.get("Positive", 0) > 0]["Word"].str.lower())
    negative_words = set(lmd_df[lmd_df.get("Negative", 0) > 0]["Word"].str.lower())
    uncertainty_words = set(lmd_df[lmd_df.get("Uncertainty", 0) > 0]["Word"].str.lower())

    return positive_words, negative_words, uncertainty_words


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text into lowercase words, removing punctuation.

    Args:
        text: Input text string.

    Returns:
        List of lowercase tokens.
    """
    if pd.isna(text):
        return []

    # Convert to lowercase
    text = str(text).lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Split into words
    words = text.split()

    return words


def compute_sentiment_score(
    text: str,
    positive_words: set[str],
    negative_words: set[str],
) -> float:
    """
    Compute sentiment score from text using LMD dictionary.

    sentiment_score = (positive_count - negative_count) / total_words
    Result is clipped to [-1, 1]. If total_words == 0, returns 0.

    Args:
        text: Input headline text.
        positive_words: Set of positive sentiment words.
        negative_words: Set of negative sentiment words.

    Returns:
        Sentiment score in range [-1, 1].
    """
    words = tokenize_text(text)
    total_words = len(words)

    if total_words == 0:
        return 0.0

    word_set = set(words)
    positive_count = len(word_set.intersection(positive_words))
    negative_count = len(word_set.intersection(negative_words))

    score = (positive_count - negative_count) / total_words
    return max(-1.0, min(1.0, score))


def compute_uncertainty_score(
    text: str,
    uncertainty_words: set[str],
) -> float:
    """
    Compute uncertainty score from text using LMD dictionary.

    uncertainty_score = uncertainty_count / total_words
    If total_words == 0, returns 0.

    Args:
        text: Input headline text.
        uncertainty_words: Set of uncertainty words.

    Returns:
        Uncertainty score in range [0, 1].
    """
    words = tokenize_text(text)
    total_words = len(words)

    if total_words == 0:
        return 0.0

    word_set = set(words)
    uncertainty_count = len(word_set.intersection(uncertainty_words))

    return uncertainty_count / total_words


def compute_event_intensity(text: str, event_keywords: list[str] = EVENT_KEYWORDS) -> float:
    """
    Compute event intensity score based on presence of event keywords.

    Counts how many EVENT_KEYWORDS appear in the headline and normalizes
    by the total number of keywords. Result is in [0, 1].

    Args:
        text: Input headline text.
        event_keywords: List of event keywords to search for.

    Returns:
        Event intensity score in range [0, 1].
    """
    words = tokenize_text(text)
    text_str = " ".join(words)

    if not event_keywords:
        return 0.0

    count = 0
    for keyword in event_keywords:
        if keyword.lower() in text_str:
            count += 1

    return count / len(event_keywords)


def add_sentiment_features(
    df: pd.DataFrame,
    positive_words: set[str],
    negative_words: set[str],
    uncertainty_words: set[str],
) -> pd.DataFrame:
    """
    Add sentiment-related features to the DataFrame.

    Args:
        df: Input DataFrame with 'headline' column.
        positive_words: Set of positive sentiment words.
        negative_words: Set of negative sentiment words.
        uncertainty_words: Set of uncertainty words.

    Returns:
        DataFrame with added sentiment_score column.
    """
    df = df.copy()
    df["sentiment_score"] = df["headline"].apply(
        lambda x: compute_sentiment_score(x, positive_words, negative_words)
    )
    return df


def add_uncertainty_features(
    df: pd.DataFrame,
    uncertainty_words: set[str],
) -> pd.DataFrame:
    """
    Add uncertainty-related features to the DataFrame.

    Args:
        df: Input DataFrame with 'headline' column.
        uncertainty_words: Set of uncertainty words.

    Returns:
        DataFrame with added uncertainty_score column.
    """
    df = df.copy()
    df["uncertainty_score"] = df["headline"].apply(
        lambda x: compute_uncertainty_score(x, uncertainty_words)
    )
    return df


def add_event_intensity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add event intensity feature to the DataFrame.

    Args:
        df: Input DataFrame with 'headline' column.

    Returns:
        DataFrame with added event_intensity column.
    """
    df = df.copy()
    df["event_intensity"] = df["headline"].apply(compute_event_intensity)
    return df


def add_ai_label_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ai_label column filled with None (reserved for future Claude API integration).

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with added ai_label column.
    """
    df = df.copy()
    df["ai_label"] = None
    return df


def main(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Steps:
    1. Load LMD dictionary.
    2. Compute sentiment_score for each headline.
    3. Compute uncertainty_score for each headline.
    4. Compute event_intensity for each headline.
    5. Add ai_label placeholder.

    Args:
        df: Input DataFrame. If None, will be loaded from RAW_DATA_PATH.

    Returns:
        DataFrame with all features added.
    """
    from config import RAW_DATA_PATH

    if df is None:
        df = pd.read_csv(RAW_DATA_PATH)

    print("Loading Loughran-McDonald dictionary...")
    lmd_df = load_lmd_dictionary()
    positive_words, negative_words, uncertainty_words = parse_lmd_categories(lmd_df)

    print(f"Loaded {len(positive_words)} positive, {len(negative_words)} negative, "
          f"{len(uncertainty_words)} uncertainty words")

    print("Computing sentiment scores...")
    df = add_sentiment_features(df, positive_words, negative_words, uncertainty_words)

    print("Computing uncertainty scores...")
    df = add_uncertainty_features(df, uncertainty_words)

    print("Computing event intensity...")
    df = add_event_intensity_features(df)

    print("Adding AI label placeholder...")
    df = add_ai_label_placeholder(df)

    print(f"Feature engineering complete. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    df = main()
    print(df.head())
