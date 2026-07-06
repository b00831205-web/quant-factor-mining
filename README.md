# Quant Factor Mining

An end-to-end equity factor research pipeline on the S&P 500, built around one
principle: **statistical honesty**. Every step that commonly inflates backtest
results — survivorship bias, overlapping-return autocorrelation, multiple
testing, look-ahead in orthogonalization, unrealistic transaction costs — is
explicitly addressed, and the final conclusions are reported with their
uncertainty, not just their point estimates.

## Headline result

The 20-day average volume factor (`TwentyDayAvgVol`) is the only candidate
that survives the full testing gauntlet:

| Stage | Result |
|---|---|
| Train IC (2015–2023, Newey-West) | t = 3.78 (20d holding), passes Bonferroni & Benjamini-Hochberg across all 18 factor × horizon tests |
| Out-of-sample quintiles (2024–2026) | Monotonic (Spearman 0.9); long-short gross ~16 %/yr, Sharpe ~1.6 |
| Carhart 4-factor attribution (daily, HAC) | Market beta 0.24 (significant), large-cap tilt; momentum & value loadings insignificant |
| Net alpha | ~7–10 %/yr point estimate, **not statistically confirmable** on 2.4 years of out-of-sample data (p ≈ 0.26) |

The honest conclusion: the factor's IC is robustly significant in-sample under
conservative testing; its out-of-sample alpha is economically meaningful but
statistically unproven. Live verification over a longer window is required —
and that is exactly what a research report should say.

The remaining seven candidate factors (momentum, short-term reversal,
volatility, downside volatility, volume-price correlation, …) fail the
corrected significance tests. Documenting *why* they fail is part of the
point.

## Methodology highlights

- **Survivorship-bias correction** — the universe is rebuilt from historical
  S&P 500 membership (764 tickers over 2015–2026); 569 were recoverable via
  yfinance, and the residual gap is disclosed rather than hidden.
- **Point-in-time universe** — each backtest cross-section only contains
  stocks that were actually index members on that date.
- **Newey-West IC tests** — daily ICs on overlapping k-day forward returns are
  autocorrelated; plain `t = IR·√n` overstates significance several-fold. NW
  (Bartlett kernel, lag = 2(k−1)) uses all daily observations while correcting
  the standard error. A down-sampled IID test is kept as a robustness control.
- **Multiple-testing control** — Bonferroni and Benjamini-Hochberg across all
  factor × holding-period combinations.
- **Train/test split with embargo** — factors are selected and the
  orthogonalization is fit on 2015–2023 only; a gap of one month before the
  test window prevents overlapping forward returns from leaking across the
  split.
- **Expanding-window orthogonalization** — correlated factors are residualized
  with betas estimated on data available up to each date (no full-sample
  look-ahead).
- **Turnover-based transaction costs** — costs are charged on actual
  membership turnover per rebalance, not on a flat 100 %-turnover assumption.
- **Sanity checks** — factor displacement and cross-sectional shuffling tests
  confirm the backtest machinery itself is not the source of the returns.

## Pipeline

```
yfinance (batch download, retry, blacklist, checkpoints)
        │  data_acquisition.py / task_1.py
        ▼
cleaning & merge (ffill, dedup)          Airflow DAG: DAG_pipeline.py
        │  task_2.py
        ▼
factor computation (8 candidates, vectorized pandas)
        │  factor_mining.py / task_3.py
        ▼
IC testing: cross-sectional & time-series IC, NW t, BH/Bonferroni,
train/test split, orthogonalization        IC_calculator.py
        ▼
quintile backtest: PIT universe, monotonicity, turnover costs,
displacement/shuffle sanity tests          back_testing.py
        ▼
Carhart 4-factor attribution (daily, HAC)  factor_attribution.py
```

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

**Data is not included** (Yahoo Finance terms of service do not permit
redistribution). To reproduce:

1. Historical S&P 500 membership: provide a CSV with `ticker`, `start_date`,
   `end_date` columns and point `SP500_MEMBERSHIP_CSV` at it.
2. Prices/volumes: `python task_1.py --date <ds> --batch manual` downloads in
   batches with checkpointing, then `task_2.py` cleans and merges.
3. Fama-French factors: download the daily FF3 and momentum CSVs from the
   [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
   into `tmp/ff3/`.
4. Run the research chain: `task_3.py` → `IC_calculator.py` →
   `back_testing.py` → `factor_attribution.py`.

For the Airflow DAG, set `QUANT_PROJECT_ROOT` and `QUANT_PYTHON_BIN` (see
`DAG_pipeline.py` docstring) and copy `airflow.cfg.example` keys into your own
config — never commit a real `airflow.cfg`.

## Known limitations

- ~195 of 764 historical members could not be recovered from yfinance (mostly
  true delistings/acquisitions), so a residual survivorship bias remains and
  likely flatters the results slightly.
- The out-of-sample window (2024–2026) covers a single market regime.
- The long-short portfolio carries a significant 0.24 market beta; a
  beta-hedged variant is on the roadmap.
- Transaction cost model is a flat per-turnover rate; no market-impact or
  borrow-cost modeling.

## Roadmap

- [ ] Migrate storage from parquet files to PostgreSQL/DuckDB
- [ ] Extend the Airflow DAG to cover IC testing → backtest → reporting
- [ ] Rebuild the analytics dashboard (frontend rewrite in progress)
- [ ] RAG-based automated research reports (ChromaDB + LLM)
- [ ] Scale the data layer (full US market) with PySpark
- [ ] Beta-hedged long-short variant; GARCH volatility targeting

## License

MIT
