# News-to-Signal Case Study

A lightweight quant research case study that converts financial news text
into structured features and tests whether those features can form
cross-sectional ranking signals.

## 1. Project Overview

This project demonstrates a simple research workflow for turning unstructured
financial news into structured features, combining them into a signal, and
evaluating signal behavior through ranking, grouped performance, and stability
diagnostics.

The goal is not to build a production trading strategy. The goal is to show
a clear and reproducible quant research process.

## 2. Research Question

Can simple text-derived features from financial news be transformed into a
cross-sectional ranking signal with measurable predictive behavior?

## 3. Repository Structure

```
news-to-signal-case-study/
├── README.md
├── requirements.txt
├── config.py
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── reference/
│       └── universe.csv
├── notebook/
│   └── case_study.ipynb
├── results/
│   ├── figures/
│   │   └── .gitkeep
│   └── tables/
│       └── .gitkeep
└── src/
    ├── __init__.py
    ├── preprocess.py
    ├── feature_engineering.py
    ├── signal_construction.py
    └── evaluation.py
```

## 4. Data

- News source: Yahoo Finance RSS (community-grade, no API key required)
- Price data: yfinance (community-grade, with retry and cache logic)
- Universe: 20 large-cap US equities defined in data/reference/universe.csv
- Sample window: 90 calendar days
- Evaluation unit: ticker-date observations (~800+ rows)

future_return_5d definition:
  Let T be the news date (advanced to next trading day T' if non-trading).
  future_return_5d = ln(Close[T'+5] / Close[T'+1])
  Rows with insufficient forward data are dropped.

## 5. Feature Engineering

Features are computed using the Loughran-McDonald Master Dictionary
(https://sraf.nd.edu/loughranmcdonald-master-dictionary/).

- sentiment_score: (positive_count - negative_count) / total_words, clipped to [-1, 1]
- uncertainty_score: uncertainty_count / total_words
- event_intensity: fraction of event keywords present in headline

## 6. Signal Definition

Three signals are constructed using cross-sectional z-score standardization
(computed independently per trading date):

- signal_sentiment_only             = z(sentiment_score)
- signal_sentiment_minus_uncertainty = z(sentiment_score) - z(uncertainty_score)
- signal_full                        = z(sentiment_score) - z(uncertainty_score) + z(event_intensity)

## 7. Baseline Comparison

The baseline comparison tests whether adding uncertainty and event intensity
contributes incremental signal value beyond sentiment alone.

## 8. Evaluation

All evaluation is cross-sectional (per trading date):
- Grouped forward returns (tercile or quintile based on sample size)
- Top-minus-Bottom spread and hit rate
- Spearman Rank IC and IC hit rate
- Monthly stability of daily spread

Grouping fallback rules:
- N < 9: drop date
- 9 <= N < 15: tercile
- N >= 15: quintile

## 9. How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Download the Loughran-McDonald Master Dictionary CSV from
   https://sraf.nd.edu/loughranmcdonald-master-dictionary/
   and place it at:
   data/reference/loughran_mcdonald.csv

3. Run the notebook:
   jupyter notebook notebook/case_study.ipynb

   Or run scripts directly:
   python -c "from src.preprocess import main; main()"

## 10. Results

[To be filled in after running the notebook.]

## 11. Limitations

1. **Small sample and short horizon**: ~20 tickers over a 90-day window.
   Statistical conclusions should be interpreted with caution.

2. **Survivorship bias**: Tickers were selected based on current liquidity,
   not historical availability, introducing look-ahead bias in sample construction.

3. **Community-grade data sources**: Both news (Yahoo Finance) and price data
   (yfinance) are community-maintained. Quality and completeness are not guaranteed.

4. **Prototype-level text labeling and signal design**: Features use simple
   word-count rules and a static dictionary. The signal has not been optimized
   or validated out-of-sample.

## 12. Next Steps

- Expand dataset and ticker universe
- Integrate Claude API for AI-assisted event labeling (ai_label)
- Add neutralization and transaction cost assumptions
- Validate on out-of-sample periods

## 13. Acknowledgements

- Loughran, T. and McDonald, B., 2011. "When is a Liability not a Liability?"
  Journal of Finance.
- yfinance open-source community
- feedparser open-source community
