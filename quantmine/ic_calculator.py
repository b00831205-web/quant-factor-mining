import pandas as pd
import os
from scipy import stats
import numpy as np
from statsmodels.stats.multitest import multipletests

def forward_return(close:pd.DataFrame, tickers: list = None, periods: list[int] | int = None)->dict:
    """Build forward holding-period returns for each ticker.

    Args:
        close: Close price dataframe indexed by date with one column per ticker.
        tickers: Ticker list to process.
        periods: Holding periods in trading days. Can be a list or a single int.

    Returns:
        A dict mapping each holding period to a date-by-ticker dataframe of
        forward returns.

    Notes:
        The return series are shifted forward, so the value at date ``t`` is the
        return realized over the next ``period`` trading days.
    """
    if isinstance(periods, int):
        periods = [periods]
    if periods is None:
        periods = [1,5,20]  #mutable default lives inside the function so callers can't mutate it
    if tickers is None:
        cols = close.columns.tolist()
    else:
        cols = tickers
    forward_return = {}
    for period in periods:
        # one pct_change per frame instead of column-by-column inserts
        # (per-column inserts fragment the DataFrame and are very slow at 570*3 columns)
        forward_return[period] = close[cols].pct_change(period).shift(-period)
    return forward_return

def data_standarization(df: dict[str, pd.DataFrame])->dict[str, pd.DataFrame]:
    """Cross-sectionally rank-standardize each frame to the range [-1, 1].

    Args:
        df: Mapping of name to a date-by-ticker value dataframe.

    Returns:
        A mapping with the same keys, where each row is ranked across tickers
        and scaled to the interval ``[-1, 1]``.

    Notes:
        Rank-based scaling is less sensitive to outliers than raw value
        normalization.
    """
    return {
        factor_name: factor_df.rank(axis = 1, pct=True)*2-1
        for factor_name, factor_df in df.items()
    }


def TM_Information_correlation(factors: dict[str, pd.DataFrame], forward_returns: dict[int, pd.DataFrame], output_path: str)->pd.DataFrame:
    """Compute time-series information correlation for each ticker.

    Args:
        tickers: Tickers to evaluate.
        factors: Factor dataframe with ``factor_ticker`` column names.
        different_holding_period: Forward return dataframe with matching naming
            convention.
        output_path: Relative or absolute path used to save the parquet output.

    Returns:
        A dataframe of time-series IC values, saved to ``output_path``.

    Notes:
        The function groups columns by ticker first and then computes the
        correlation between factor series and holding-period return series.
    """
    result = {}
    for factor_name ,factor_df in factors.items():
        factor_ticker = list(factor_df.columns)
        for period, return_df in forward_returns.items():
            forward_returns_ticker = list(return_df.columns)
            overlap = list(set(factor_ticker) & set(forward_returns_ticker))
            ic_series = factor_df[overlap].corrwith(return_df[overlap], method= 'pearson',axis=0)
            result[(factor_name, period)] = ic_series
    TM_IC_matrix=pd.DataFrame(result)
    TM_IC_matrix.columns.names = ['factor', 'period']
    TM_IC_matrix.to_parquet(os.path.join(os.getcwd(), output_path))
    print("time Series information correlation computation complete")

def CS_Information_Correlation(factors: dict[str, pd.DataFrame], forward_returns: dict[int, pd.DataFrame], output_path: str)-> pd.DataFrame:
    """Compute cross-sectional information correlation across dates.

    Args:
        factors: Factor dataframe with ``factor_ticker`` column names.
        different_holding_period: Forward return dataframe with matching naming
            convention.
        output_path: Relative or absolute path used to save the parquet output.

    Returns:
        A dataframe of cross-sectional IC values, saved to ``output_path``.

    Notes:
        This function correlates factor values with same-date forward returns
        across the cross section of tickers.
    """
    result = {}
    for factor_name ,factor_df in factors.items():
        factor_ticker = list(factor_df.columns)
        for period, return_df in forward_returns.items():
            forward_returns_ticker = list(return_df.columns)
            overlap = list(set(factor_ticker) & set(forward_returns_ticker))
            ic_series = factor_df[overlap].corrwith(return_df[overlap], method= 'pearson',axis=1)
            result[(factor_name, period)] = ic_series
    CS_IC_matrix=pd.DataFrame(result)
    CS_IC_matrix.columns.names = ['factor', 'period']
    CS_IC_matrix.to_parquet(os.path.join(os.getcwd(), output_path))
    print("Cross-sectional information correlation computation complete")

    return CS_IC_matrix

def summary(cross_section_IC_matrix:pd.DataFrame)->pd.DataFrame:
    """Summarize an IC dataframe with mean, std, IR, sign ratio, and count.

    Args:
        cross_section_IC_matrix: IC dataframe where each column is a factor or
            factor-holding-period combination.

    Returns:
        A summary dataframe indexed by column name.

    Input : ic_df, MultiIndex column(factor, holding_period)
    Output : DataFrame, MultiIndex index (factor, holding_period)
    """
    return pd.DataFrame({
        'IC_mean' : cross_section_IC_matrix.mean(),
        'IC_std': cross_section_IC_matrix.std(),
        'IR': cross_section_IC_matrix.mean()/cross_section_IC_matrix.std(),
        'IC>0 pct': (cross_section_IC_matrix>0).mean(),
        'n': cross_section_IC_matrix.count(),
    })

def resample_summary(cross_section_IC: pd.DataFrame, periods:list|int)-> pd.DataFrame:
    """Build IC summaries from down-sampled, approximately independent samples.

    Args:
        cross_section_IC: Cross-sectional IC dataframe.
        periods: Holding periods in trading days.

    Returns:
        A concatenated dataframe of summary statistics for each holding period.

    Notes:
        The dataframe is sub-sampled using ``iloc[::period]`` to reduce overlap
        dependence when the holding period is longer than one day.
    """
    if isinstance(periods, int):
        periods = [periods]
    result = {}
    for period in periods:
        period_df = cross_section_IC.xs(period, level = 'period', axis = 1) #xs already returns a DataFrame
        summary_df = period_df.iloc[::period]
        result[f'{period}HoldingPeriodSummary']=summary(summary_df)   
    result_df = pd.concat(result.values())
    return result_df

def newey_west_summary(cross_section_IC: pd.DataFrame, lag_multiplier: int = 2)-> pd.DataFrame:
    """Compute Newey-West adjusted IC statistics for overlapping returns.

    Args:
        cross_section_IC: Cross-sectional IC dataframe.
        lag_multiplier: Multiplier used to set the Newey-West lag length.

    Returns:
        A dataframe containing IC mean, IC std, IR, sample size, lag, Newey-West
        t-statistic, and p-value.

    Notes:
        Overlapping holding periods create autocorrelation, so a plain IID t-test
        can overstate significance. When ``period == 1``, the lag becomes zero and
        the method collapses to a standard t-test.
    """
    rows = {}
    for col in cross_section_IC.columns:
        factor, period = col
        lag = lag_multiplier * max(period - 1, 0)
        s = cross_section_IC[col].dropna()
        n = len(s)
        mu = s.mean()
        e = (s - mu).values
        var = e @ e / n #with lag=0 this is the plain sample variance
        for l in range(1, min(lag, n - 1) + 1):
            w = 1 - l / (lag + 1) #Bartlett kernel weight, keeps the variance estimate non-negative
            var += 2 * w * (e[:-l] @ e[l:]) / n
        se = np.sqrt(var / n)
        rows[col] = {'IC_mean': mu, 'IC_std': s.std(), 'IR': mu / s.std(),
                     'n': n, 'lag': lag, 'NW_t': mu / se}
    nw_df = pd.DataFrame(rows).T
    nw_df.index =nw_df.index.set_names(['factor', 'period'])
    nw_df['p_value'] = stats.t.sf(nw_df['NW_t'].abs(), df=nw_df['n'] - 1) * 2
    return nw_df

def multiple_testing(summary_df:pd.DataFrame)->pd.DataFrame:
    """Apply multiple-testing corrections to IC significance results.

    Args:
        summary_df: Summary dataframe from either ``newey_west_summary`` or
            ``resample_summary``.

    Returns:
        A dataframe with raw p-values and significance flags under several
        correction schemes.

    Notes:
        If ``NW_t`` exists, the function uses the Newey-West corrected path.
        Otherwise it falls back to the IID resampled path.
    """
    if 'NW_t' in summary_df.columns: #NW path: autocorrelation-corrected t and p (output of newey_west_summary)
        significant_t = pd.DataFrame({
            "t": summary_df['NW_t'],
            "p_value": summary_df['p_value']
            })
    else: #IID path: independent-sample t-test on down-sampled ICs (output of resample_summary)
        significant_t = pd.DataFrame({
            "t":summary_df["IR"]*np.sqrt(summary_df["n"]),
            "p_value":stats.t.sf(x=abs(summary_df["IR"]*np.sqrt(summary_df["n"])), df=summary_df['n']-1)*2
            })
    significant_t['significant'] = significant_t["p_value"] < 0.05
    significant_t['Bonferroni_significant'] = significant_t["p_value"] < 0.05/len(summary_df)
    significant_t['Rank'] = significant_t["p_value"].rank(ascending=1,method='max')
    rej_bonf, _, _, _ = multipletests(significant_t['p_value'], alpha = 0.05, method='fdr_bh')
    significant_t['BH_significant'] = rej_bonf
    return significant_t

def orthogonal_analysis(factors: dict[str, pd.DataFrame]):
    """Compute average factor correlation and identify highly correlated pairs.

    Args:
        factors_ticker: Factor dataframe with ``factor_ticker`` columns.

    Returns:
        A tuple of ``(average_correlation_matrix, high_correlation_dict)``.

    Notes:
        Dates with missing factor columns are skipped. Factors whose absolute
        average correlation exceeds 0.5 are treated as highly correlated.
    """
    factor = sorted(factors.keys())
    ticker = sorted(set().union(*[tickers.columns.tolist() for tickers in factors.values()])) #* unpacks the list into separate arguments for union

    #reshape each factor into one dates*tickers wide frame with a unified column order,
    #so everything below is frame-level vectorized work
    frames = { f: factors[f].reindex(columns = ticker) for f in factor}

    #valid day: every factor has at least one non-NaN value that day
    #(same skip condition as the original day-by-day implementation)
    valid = pd.concat({f: frames[f].notna().any(axis=1) for f in factor}, axis=1).all(axis=1)
    valid_days = int(valid.sum())

    avg_corr = pd.DataFrame(1.0, index=factor, columns=factor) #diagonal is always 1
    #iterate the same sorted list on both sides: mixing dict insertion order with
    #the sorted list would mispair factors (some pairs skipped, diagonal corrupted)
    for i, fa in enumerate(factor):
        a = frames[fa]
        for fb in factor[i+1:]:
            b = frames[fb]
            #vectorized expansion of daily cross-sectional Pearson correlation:
            #each day, correlate over tickers where both factors are non-NaN
            mask = a.notna() & b.notna()
            xa, xb = a.where(mask), b.where(mask)
            n = mask.sum(axis=1)
            sa, sb = xa.sum(axis=1), xb.sum(axis=1)
            cov = (xa*xb).sum(axis=1) - sa*sb/n
            var_a = (xa*xa).sum(axis=1) - sa*sa/n
            var_b = (xb*xb).sum(axis=1) - sb*sb/n
            corr_t = cov/np.sqrt(var_a*var_b)
            #matches the original implementation: sum over valid days (NaN if any day's
            #correlation is undefined) and divide by the number of valid days
            pair_avg = corr_t[valid].to_numpy().sum()/valid_days
            avg_corr.loc[fa, fb] = pair_avg
            avg_corr.loc[fb, fa] = pair_avg

    high_corr_dict={
        column: avg_corr.index[(avg_corr[column].abs()>0.5) & (avg_corr.index!=column)].tolist() for column in avg_corr.columns
    }
    return avg_corr, high_corr_dict

def orthogonalize(factors: dict[str, pd.DataFrame], high_corr_dict: dict, ic_summary:pd.DataFrame, threshold: float = 0.03, min_period: int = 60)->dict:
    """Orthogonalize highly correlated factors using expanding regression.

    Args:
        factor_df: Factor dataframe with ``factor_ticker`` columns.
        high_corr_dict: Dictionary of highly correlated factor names produced by
            ``orthogonal_analysis``.
        ic_summary: IC summary dataframe used to estimate factor quality.
        threshold: Minimum average IR required to keep a factor. Default is 0.03.
        min_period: Minimum number of observations used in the expanding beta
            calculation.

    Returns:
        A dataframe where some factor columns may be dropped or replaced by
        orthogonalized residuals.

    Notes:
        When both factors in a correlated pair have low IR, both are dropped.
        Otherwise the weaker factor is residualized against the stronger one.
    """
    pairs=set()
    for factor, factors_corr in high_corr_dict.items():
        for factor_corr in factors_corr:
            pair = tuple(sorted([factor, factor_corr]))
            pairs.add(pair)
    
    factor_ir = ic_summary['IR'].abs().groupby(level = 'factor').mean()

    print("Average IR:")
    print({k: round(v,4) for k,v in sorted(factor_ir.items())})
    print(f"threshold={threshold}, will be dropped because it is below the threshold")

    result = factors.copy()
    drop = set()
    orthogonalized = set()
    

    for factor_a, factor_b in pairs:
        if factor_a in drop or factor_b in drop:
            continue
        ir_a = factor_ir.get(factor_a, 0) #pd.Series supports .get like a dict
        ir_b = factor_ir.get(factor_b, 0)

        if ir_a < threshold and ir_b < threshold:
            result.pop(factor_a)
            result.pop(factor_b)
            drop.update([factor_a, factor_b])
            continue
            
        keeper = factor_a if ir_a > ir_b else factor_b
        to_orthogonalize = factor_b if ir_a > ir_b else factor_a

        if ir_a < threshold:
            result.pop(factor_a)
            drop.add(factor_a)

        elif ir_b < threshold:
            
            result.pop(factor_b)
            drop.add(factor_b)
                
        if ir_a >= threshold and ir_b >= threshold:
            if to_orthogonalize in orthogonalized:
                continue
            if keeper not in result.keys() or to_orthogonalize not in result.keys():
                continue
            x = result[keeper]
            y = result[to_orthogonalize]
            
            expanding_cov = x.expanding(min_periods= min_period).cov(y)
            expanding_var = x.expanding(min_periods= min_period).var()
            beta_series = expanding_cov/expanding_var
            
            residuals = y - beta_series * x
            result[to_orthogonalize] = residuals
            orthogonalized.add(to_orthogonalize)
    return result

def time_series_stationary_test(CS_IC_matrix:pd.DataFrame, rolling_period:int =126, periods:list = None)-> pd.DataFrame:
    """Compute rolling IC, autocorrelation, and yearly IC summaries.

    Args:
        CS_IC_matrix: Cross-sectional IC dataframe indexed by date.
        rolling_period: Window size for rolling mean IC. Default is 126.
        periods: Lags used when computing autocorrelation.

    Returns:
        A tuple of ``(rolling_ic_df, acf_df, yearly_df)``.

    Notes:
        The index is converted to ``datetime64`` before grouping and rolling.
    """
    if periods is None:
        periods = [1,5,20]

    rolling_window_IC={}
    acf_ic={}
    
    CS_IC_matrix.index = pd.to_datetime(CS_IC_matrix.index)
    for col in CS_IC_matrix.columns:
        # rolling ic
        rolling_window_IC[col]=CS_IC_matrix[col].rolling(rolling_period).mean()
    rolling_ic_df = pd.DataFrame(rolling_window_IC, index = CS_IC_matrix.index)

    for col in CS_IC_matrix.columns:
        for period in periods:
                # acf_ic
                acf_ic[(col,period)] = CS_IC_matrix[col].corr(CS_IC_matrix[col].shift(period), method="pearson") #corr is for Series, corrwith is for DataFrame
    acf_df = pd.Series(acf_ic).to_frame(name = 'ACF') #dict values are scalars, so convert to a Series before building the frame
    yearly_summary={}
    # yearly IC
    for year, group in CS_IC_matrix.groupby(CS_IC_matrix.index.year):
        yearly_summary[year] = summary(group)
    yearly_df = pd.concat(yearly_summary, axis= 0)
        
    return rolling_ic_df, acf_df, yearly_df

def get_constitunents_at_date(historical_df: pd.DataFrame, date: pd.Timestamp)->set:
    """Get the set of active constituents on a specific date.

    Args:
        historical_df: Historical constituent table with ``start_date``,
            ``end_date``, and ``ticker`` columns.
        date: Target date to query.

    Returns:
        A set of tickers that are active on the given date.

    Notes:
        The date comparison is inclusive. Missing ``end_date`` values are treated
        as open-ended membership.
    """
    mask = (historical_df['start_date'] <= date) & (historical_df['end_date'].isnull() | (historical_df['end_date'] >= date))
    return set(historical_df.loc[mask, 'ticker'].str.replace('.','-',regex = False))

def train_test_analysis(cs_df: pd.DataFrame, factors: dict[str, pd.DataFrame], close: pd.DataFrame , train_end: str, test_start: str, periods: list|int = None):
    """Run the full train/test IC workflow and orthogonalization pipeline.

    Args:
        cs_df: Cross-sectional IC dataframe for all available dates.
        factors: Mapping of factor name to a date-by-ticker value dataframe.
        close: Close price dataframe used to generate holding-period returns.
        train_end: Last date included in the training sample.
        test_start: First date included in the test sample.
        periods: Holding periods in trading days.

    Returns:
        A dictionary containing training diagnostics, orthogonalized factor
        outputs, test splits, and summary statistics.

    Notes:
        The training sample is used to select significant factors and build the
        orthogonalization mapping before evaluating the test period.
    """
    if isinstance(periods, int):
        periods = [periods]
    
    if periods is None:
        periods = [1,5,20]

    train_cs_df = cs_df[cs_df.index <= train_end]
    test_cs_df = cs_df[cs_df.index >= test_start]

    forward_return_train = forward_return(close[close.index<=train_end], close.columns, periods)
    forward_return_train_stand = data_standarization(forward_return_train)
    forward_return_test = forward_return(close[close.index>=test_start], close.columns, periods)
    forward_return_test_stand = data_standarization(forward_return_test)

    train_factor_ticker = {factor_name : train_factor_ticker.loc[:train_end] for factor_name, train_factor_ticker in factors.items()}
    test_factor_ticker = {factor_name : test_factor_ticker.loc[test_start:] for factor_name, test_factor_ticker in factors.items()}

    resample_summary_train = resample_summary(train_cs_df, periods)
    print(resample_summary_train)
    print(resample_summary_train.index.tolist()[:5])

    orth_analysis,  high_corr_dict= orthogonal_analysis(train_factor_ticker)
    orthogonalize_result = orthogonalize(factors, high_corr_dict, resample_summary_train)
    orthogonalize_result.pop('excess_return')
    
    orth_train = {factor_name : orth_result.loc[:train_end] for factor_name, orth_result in orthogonalize_result.items()}
    orth_test = {factor_name : orth_result.loc[test_start: ] for factor_name, orth_result in orthogonalize_result.items()}

    cs_df_orth_train = CS_Information_Correlation(factors = orth_train,
                                                  forward_returns=forward_return_train_stand,
                                                  output_path = 'tmp/ic_test/cs_df_orth_train.parquet')
    #primary test: Newey-West on full daily ICs, correcting overlapping-holding-period autocorrelation
    nw_summary_orth_train = newey_west_summary(cs_df_orth_train)
    multiple_testing_train = multiple_testing(nw_summary_orth_train)
    print("=== Newey-West test ===")
    print(multiple_testing_train)
    print("Number of True values in BH_significant:", multiple_testing_train['BH_significant'].sum())

    #robustness control: down-sampling (iloc[::period], conservative but phase-dependent)
    resample_summary_orth_train = resample_summary(cs_df_orth_train, periods)
    multiple_testing_resample = multiple_testing(resample_summary_orth_train)
    print("=== Down-sampling control ===")
    print("Number of True values in BH_significant:", multiple_testing_resample['BH_significant'].sum())
    significant_factor = (
    multiple_testing_resample[multiple_testing_resample['BH_significant']]
    .index.get_level_values('factor')
    .unique()
    .tolist()
    )
    significant_factor_nw = (
    multiple_testing_train[multiple_testing_train['BH_significant']]
    .index.get_level_values('factor')
    .unique()
    .tolist()
)
    rolling_ic_train , acf_train, yearly_train = time_series_stationary_test(cs_df_orth_train)

    return {"resample_summary_train":resample_summary_train,
            "nw_summary_orth_train": nw_summary_orth_train,
            "multiple_testing_train": multiple_testing_train,
            "multiple_testing_resample": multiple_testing_resample,
            "orth_analysis": orth_analysis,
            "orthogonalize_result_full": orthogonalize_result,
            "orth_factors_test": orth_test,
            "high_corr_dict": high_corr_dict,
            "test_cs_df": test_cs_df,
            "test_factor_ticker": test_factor_ticker,
            "forward_return_train" : forward_return_train,
            "forward_return_train_stand" : forward_return_train_stand,
            "forward_return_test" : forward_return_test,
            "forward_return_test_stand" : forward_return_test_stand,
            "significant_factors": significant_factor,
            "significant_factors_nw": significant_factor_nw,
            "rolling_ic_train": rolling_ic_train,
            "acf_train": acf_train,
            "yearly_train" : yearly_train}

    

if __name__=='__main__':
    #Packaged train/test research workflow. Expects the processed parquet files
    #described in the README; run from the repo root as:
    #    python -m quantmine.ic_calculator
    from .datareader import MarketData
    from .factor_register import build_param_pool, calculate_all_factors
    from . import factor_mining  # noqa: F401  importing registers the built-in factors

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    close_data = pd.read_parquet("data/processed/processed_close.parquet")
    volume_data = pd.read_parquet("data/processed/processed_volume.parquet")
    os.makedirs("tmp/ic_test", exist_ok=True)

    data = MarketData(close=close_data, volume=volume_data)
    pool = build_param_pool(data, day=5, halflife=10, period=20)
    failed, factors = calculate_all_factors(pool)

    forward_returns = forward_return(close_data, periods=[1, 5, 20])
    cs_df = CS_Information_Correlation(factors=factors, forward_returns=forward_returns,
                                       output_path="tmp/ic_test/cs_df.parquet")

    #one-month embargo between train_end and test_start keeps overlapping
    #forward returns from leaking across the split
    train_test_analysis_result = train_test_analysis(cs_df=cs_df, factors=factors, close=close_data,
                                                     train_end='2023-12-31', test_start='2024-02-01')
