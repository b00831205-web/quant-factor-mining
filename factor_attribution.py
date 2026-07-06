"""Carhart four-factor attribution of the long-short backtest returns.

Regresses the daily long-short return series on the Fama-French three factors
plus momentum (all taken from the Ken French Data Library) with Newey-West
(HAC) standard errors. This answers whether the strategy return is explained
by known risk premia or carries unexplained alpha.

Factor data (not included in the repo, download from the Ken French Data
Library and place under ``tmp/ff3/``):
    - F-F_Research_Data_Factors_daily.csv
    - F-F_Momentum_Factor_daily.csv
"""
import pandas as pd
import statsmodels.api as sm

from back_testing import expand_to_daily_returns


def load_french_factors(ff3_path: str, mom_path: str) -> pd.DataFrame:
    """Load and merge daily FF3 and momentum factors from Ken French CSVs.

    Args:
        ff3_path: Path to ``F-F_Research_Data_Factors_daily.csv``.
        mom_path: Path to ``F-F_Momentum_Factor_daily.csv``.

    Returns:
        A dataframe indexed by date with columns ``Mkt-RF``, ``SMB``, ``HML``,
        ``RF`` and ``Mom``, converted from percent to decimal returns.

    Notes:
        The raw CSVs carry header/footer junk rows; only rows whose date field
        matches ``YYYYMMDD`` are kept.
    """
    ff3 = pd.read_csv(ff3_path, skiprows=4)
    ff3.columns = ['Date'] + list(ff3.columns[1:])
    ff3 = ff3[ff3['Date'].astype(str).str.match(r'^\d{8}$')]

    mom = pd.read_csv(mom_path, skiprows=13)
    mom = mom.iloc[:, :2]
    mom.columns = ['Date', 'Mom']
    mom = mom[mom['Date'].astype(str).str.match(r'^\d{8}$')]

    merged = ff3.merge(mom, on='Date', how='inner')
    merged['date'] = pd.to_datetime(merged['Date'], format='%Y%m%d')
    cols = ['Mkt-RF', 'SMB', 'HML', 'RF', 'Mom']
    merged[cols] = merged[cols].astype(float) / 100  #percent -> decimal
    return merged.set_index('date')[cols]


def carhart_attribution(daily_returns: pd.DataFrame, factors: pd.DataFrame, maxlags: int = 20):
    """Regress the long-short series on Mkt-RF, SMB, HML and Mom.

    Args:
        daily_returns: Wide dataframe containing a ``long_short`` column of
            daily returns (output of ``expand_to_daily_returns``).
        factors: Daily factor dataframe from ``load_french_factors``.
        maxlags: Newey-West lag length for the HAC covariance. Default 20 to
            match the longest holding period.

    Returns:
        A fitted statsmodels OLS results object. ``params['const']`` is the
        daily alpha; multiply by 252 to annualize.

    Notes:
        The long-short portfolio is self-financing, so the raw spread (not the
        excess over RF) is regressed.
    """
    combined = (daily_returns[['long_short']]
                .join(factors[['Mkt-RF', 'SMB', 'HML', 'Mom']], how='inner')
                .dropna())
    x = sm.add_constant(combined[['Mkt-RF', 'SMB', 'HML', 'Mom']])
    return sm.OLS(combined['long_short'], x).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})


if __name__ == "__main__":
    ticker_history = pd.read_pickle('tmp/back_test/test_ticker_history.pkl')
    history = ticker_history['20DaysHoldingPeriod']['TwentyDayAvgVol']
    close_data = pd.read_parquet('data/processed/processed_close.parquet')

    daily_wide = expand_to_daily_returns(history, close_data)
    factors = load_french_factors('tmp/ff3/F-F_Research_Data_Factors_daily.csv',
                                  'tmp/ff3/F-F_Momentum_Factor_daily.csv')
    model = carhart_attribution(daily_wide, factors)
    print(model.summary())
    alpha_daily = model.params['const']
    print(f"\nAnnualized alpha: {alpha_daily * 252:.2%} (NW t = {model.tvalues['const']:.2f}, "
          f"p = {model.pvalues['const']:.4f})")
