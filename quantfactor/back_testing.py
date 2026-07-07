import pandas as pd
import os
from quantfactor.ic_calculator import different_holding_period, train_test_analysis, CS_Information_Correlation, get_constitunents_at_date
import numpy as np
from scipy.stats import spearmanr
import pickle


def truncate_quantile(cross_section_series: list|pd.Series, part:int =5):
    """Split a sorted cross-sectional series into quantile buckets.

    Args:
        cross_section_series: A 1D sorted sequence of tickers or values.
            The input must already be sorted in ascending order before calling.
        part: Number of groups to split into. Default is 5.

    Returns:
        A list of length ``part``. Each element is a slice of the input series.

    Notes:
        This function does not sort the data. It only slices by position, so
        the input order determines the quantile grouping.
    """
    length = len(cross_section_series)
    batch_size = length // part
    batch = []
    for i in range(part):
        if i == part-1:
            quantile = cross_section_series[i*batch_size:length]
        else:
            quantile = cross_section_series[i*batch_size: (i+1)*batch_size]
        batch.append(quantile)
    return list(batch)
        

def quantile_backtest(historical_df: pd.DataFrame ,factor_ticker: pd.DataFrame,significant_factor_list:list,different_holding_period: pd.DataFrame, period: int, part:int =5):
    """Run a quantile long-short backtest for each selected factor.

    Args:
        historical_df: A dataframe describing ticker membership over time.
            It is used to filter tickers to the ones valid on each date.
        factor_ticker: Factor values indexed by date, with ticker-specific
            columns such as ``factorName_TICKER``.
        significant_factor_list: Factor names to evaluate.
        different_holding_period: Forward holding-period returns with columns
            named like ``{period}DaysHoldingPeriod_TICKER``.
        period: Holding period in trading days used to match the return table.
        part: Number of quantile groups to form. Default is 5.

    Returns:
        A dictionary mapping each factor name to a dataframe with quantile
        returns (Q1..Qn) and a ``long_short`` column.

    Notes:
        The function expects the factor dataframe and return dataframe to use
        compatible date indices and ticker suffix conventions.
    """
    all_result={}
    all_ticker_history = {}
    
    for significant_factor in significant_factor_list:
        factor = [f for f in factor_ticker.columns if f.rsplit('_',1)[0] == significant_factor]
        factor_df = factor_ticker[factor]
        holding_period = [d for d in different_holding_period.columns if d.rsplit('_',1)[0]== f'{period}DaysHoldingPeriod']
        different_holding_period_df = different_holding_period[holding_period]
        result=[]
        ticker_history = []
        for index in range(0, len(factor_ticker.index), period):
            curr_date = factor_df.index[index]
            if curr_date not in different_holding_period_df.index:
                continue
            valid_tickers = get_constitunents_at_date(historical_df=historical_df, date= curr_date)

            if factor_df.iloc[index].isnull().all():
                continue
            cross_section_factor = factor_df.iloc[index]
            cross_section_factor = cross_section_factor[cross_section_factor.index.map(lambda x: x.rsplit('_',1)[1] in valid_tickers)]
            ranked_cross_section = cross_section_factor.rank(ascending=True)
            ranked_cross_section = ranked_cross_section.sort_values()
            tickers = ranked_cross_section.index.map(lambda x: x.rsplit('_',1)[1])

            ranked_list = truncate_quantile(tickers ,part = part)
            group_list =[]
            for group in ranked_list:
                group_list.append([f"{period}DaysHoldingPeriod_{t}" for t in group])
            
            _return = {'date':curr_date}    
            _tickers = {'date':curr_date}
            for i in range(part):
                group_return = different_holding_period_df.loc[curr_date, group_list[i]]
                _return[f'Q{i+1}'] = group_return.mean()
                _tickers[f'Q{i+1}'] = set(t.rsplit('_',1)[1] for t in group_list[i])
            ticker_history.append(_tickers)
            result.append(_return)
        result_df = pd.DataFrame(result).set_index('date')
        result_df['long_short'] = result_df['Q5']-result_df['Q1']
        all_ticker_history[significant_factor] = ticker_history
        all_result[significant_factor] = result_df
    return all_result ,all_ticker_history

def expand_to_daily_returns(tickers_history:list, close_data: pd.DataFrame, cost_per_trade: float = 0.001):
    """Expand periodic rebalance snapshots into a net-of-cost daily return series.

    Between two rebalance dates the portfolio holds the quantile members
    fixed, so each daily return is the equal-weighted mean of member returns.
    Transaction costs are charged on the first trading day after each
    rebalance, scaled by the actual membership turnover of that quantile.

    Args:
        tickers_history: List of dicts produced by ``quantile_backtest``. Each
            item holds the rebalance ``date`` and the member set per quantile.
        close_data: Close price dataframe covering all member tickers.
        cost_per_trade: One-way cost per unit of turnover. The deduction is
            ``turnover * 2 * cost_per_trade`` to cover both buys and sells.

    Returns:
        A wide dataframe of daily returns with columns ``Q1``..``Q5`` and a
        ``long_short`` spread column.

    Notes:
        The price window must include the rebalance date itself: ``pct_change``
        makes the window's first row NaN, and that row is dropped. Without the
        rebalance-day anchor, each window's first daily return is lost and the
        cost deduction lands on a NaN (i.e. costs are silently never charged).
    """
    daily_records = []
    for i in range(len(tickers_history)-1):
        curr = tickers_history[i]
        next_date = tickers_history[i+1]['date']
        curr_date = curr['date']

        #窗口含起点(调仓日): pct_change后调仓日行为NaN被丢弃, 调仓日->次日的收益保留,
        #窗口右端含next_date(归属旧组合), 下一窗口从next_date+1天起, 不重不漏
        window_dates = close_data.index[(close_data.index >= curr_date) & (close_data.index <= next_date)]
        for q in [f'Q{n}' for n in range(1,6)]:
            tickers_in_group = list(curr[q])
            group_price = close_data.loc[window_dates, tickers_in_group]
            portfolio_daily_return = group_price.pct_change().mean(axis=1).iloc[1:]

            if i>0:
                prev_set = set(tickers_history[i-1][q])
                curr_set = curr[q]
                overlap = len(prev_set & curr_set)
                turnover = 1 - overlap/len(curr_set)
            else:
                turnover = 1.0 #初始建仓视为全额换手

            cost_today = turnover *2 *cost_per_trade

            for j, (d, r) in enumerate(portfolio_daily_return.items()):
                if j == 0:
                    r = r - cost_today
                daily_records.append({'date':d, 'group':q, 'return':r})
    daily_long = pd.DataFrame(daily_records)
    daily_wide = daily_long.pivot(index='date', columns='group', values='return')
    daily_wide['long_short'] = daily_wide['Q5'] - daily_wide['Q1']
    return daily_wide

def calculate_turnover(ticker_history: list, group: str)->pd.Series:
    """Compute per-rebalance membership turnover for one quantile group.

    Args:
        ticker_history: List of dicts produced by ``quantile_backtest``.
        group: Quantile column name, e.g. ``"Q1"`` or ``"Q5"``.

    Returns:
        A series aligned with the rebalance sequence. The first element is NaN
        because turnover is undefined before the initial holdings exist.

    Notes:
        Turnover is ``1 - |prev ∩ curr| / |curr|``, the fraction of the current
        portfolio that had to be newly bought at this rebalance.
    """
    turnovers = [np.nan]
    for i in range(1, len(ticker_history)):
        prev_set = ticker_history[i-1][group]
        curr_set = ticker_history[i][group]
        overlap = len(prev_set & curr_set)
        turnover = 1-(overlap/len(curr_set))
        turnovers.append(turnover)
    return pd.Series(turnovers)


def back_test_senity_test(significant_factor_list:list, factor_ticker: pd.DataFrame ,diff_holding_period: pd.DataFrame, close:pd.DataFrame, periods:list,
                          origincal_back_test: dict):
    """Run sensitivity tests against factor displacement and shuffled factors.

    Args:
        significant_factor_list: Factor names to evaluate.
        factor_ticker: Original factor table indexed by date.
        diff_holding_period: Precomputed holding-period return table.
        close: Close price table used to recompute forward returns.
        periods: List of holding periods in trading days.
        origincal_back_test: Baseline quantile backtest results used as the
            comparison reference.

    Returns:
        A tuple of ``(total_difference, displace_difference, shuffle_difference)``.

    Notes:
        This function compares the baseline backtest against two perturbations:
        shifting factor values by one period and randomly shuffling factor rows.
    """
    tickers = list(close.columns)
    factor_ticker_shifting = factor_ticker.shift(-1)
    shuffle_data = factor_ticker.copy()
    for idx in shuffle_data.index:
        shuffle_data.loc[idx] = np.random.permutation(shuffle_data.loc[idx].values)

    factor_displacement_result = {}
    period_difference = {}
    shuffle_data_return = {}

    
    for period in periods:
        #holding period return and close data shifting difference
        
        raw_return = pd.DataFrame({f'{period}DaysHoldingPeriod_{ticker}' : close[ticker].pct_change(period).shift(-period) for ticker in tickers})
        common_col = diff_holding_period.columns.intersection(raw_return.columns)
        diff = (diff_holding_period[common_col]-raw_return[common_col]).abs().sum().sum() #第一个sum得到series，第二个sum把所有series相加得到标量
        period_difference[f'{period}_holding_return'] = diff

        #factor displacement
        factor_displacement_result[f'{period}DaysHoldingPeriod'] = quantile_backtest(factor_ticker_shifting, significant_factor_list, diff_holding_period, period)

        #shuffle data
        shuffle_data_return[f'{period}DaysHoldingPeriod'] = quantile_backtest(shuffle_data, significant_factor_list, diff_holding_period, period)

    total_difference = 0
    for period, diff in period_difference.items():
        total_difference += diff
    
    displace_difference = {}
    for period, factor_result in factor_displacement_result.items():
        for factor_name, df in factor_result.items():
            displace_difference[f'{period}_{factor_name}'] = df - origincal_back_test[period][factor_name]
    
    shuffle_difference = {}
    for period, factor_result in shuffle_data_return.items():
        for factor_name, df in factor_result.items():
            shuffle_difference[f'{period}_{factor_name}'] = df - origincal_back_test[period][factor_name]
    
    for key, diff in displace_difference.items():
        long_short_diff = diff['Q5'] - diff['Q1']
        print(f'{key}: displace factor increase SHAP ratio:{long_short_diff.mean()/long_short_diff.std():.4f}')

    for key, diff in shuffle_difference.items():
        long_short_diff = diff['Q5'] - diff['Q1']
        print(f'{key}: displace factor increase SHAP ratio:{long_short_diff.mean()/long_short_diff.std():.4f}')

    return total_difference, displace_difference, shuffle_difference

def performance_summary(back_test_quantile:pd.DataFrame, periods:int):
    """Summarize quantile backtest performance.

    Args:
        back_test_quantile: Dataframe of periodic returns for each quantile.
        periods: Holding period in trading days used to annualize metrics.

    Returns:
        A tuple of ``(summary_dataframe, net_return_series)`` where the
        dataframe contains annualized return, volatility, Sharpe ratio, max
        drawdown, and win rate.

    Notes:
        The input should contain return series, not cumulative returns.
    """
    result = []
    for col in back_test_quantile.columns:
        r = back_test_quantile[col].dropna()
        net_return = (1+r).cumprod()
        
        total_return = net_return.iloc[-1] - 1
        n_years = len(r) * periods / 252 
        
        if net_return.iloc[-1] > 0:
            yearly_return = net_return.iloc[-1]**(1/n_years)-1
        else:
            yearly_return = np.nan
        volatility = r.std() * np.sqrt(252/periods)
        sharp = yearly_return / volatility if (volatility != 0 and not np.isnan(yearly_return)) else np.nan
        
        rolling_max = net_return.cummax()
        drawdown = (net_return - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        win_rate = (r>0).mean()
        summary = {
            'quantile': col,
            'yearly_return': yearly_return,
            'volatility' :volatility,
            'sharp_ratio': sharp,
            'max_drawdown':max_drawdown,
            'win_rate': win_rate
        }
        result.append(summary)
    return pd.DataFrame(result).set_index('quantile'), net_return

def monotonicity_test(result_df: pd.DataFrame, n_group: int = 5)->dict:
    """Test whether quantile returns are monotonically increasing.

    Args:
        result_df: Dataframe containing quantile columns named ``Q1`` to
            ``Qn``.
        n_group: Number of quantile groups to inspect. Default is 5.

    Returns:
        A dictionary with the mean-based Spearman correlation, its p-value,
        the average daily correlation, and the fraction of positive daily
        correlations.

    Notes:
        Rows with missing quantile values are skipped in the daily test.
    """
    quantile_cols = [f'Q{i}' for i in range(1,n_group+1)]
    ranks = list(range(1, n_group+1))
    means = result_df[quantile_cols].mean()
    corr_simple, pval_simple = spearmanr(ranks, means.values)
    
    daily_corrs = []
    for _, row in result_df[quantile_cols].iterrows():
        if row.isnull().any():
            continue
        c, _ = spearmanr(ranks, row.values)
        daily_corrs.append(c)
    return {'mean_based_corr': corr_simple,
            'mean_based_pvalue': pval_simple,
            'daily_avg_corr': pd.Series(daily_corrs).mean(),
            'daily_corr_positive_pct':(pd.Series(daily_corrs)>0).mean()}

def apply_transcation_cost(result_df: pd.DataFrame, ticker_history: list,cost_per_trade: float =0.001)-> pd.DataFrame:
    """Apply turnover-based transaction costs to quantile returns.

    Args:
        result_df: Quantile return dataframe produced by the backtest.
        ticker_history: Per-rebalance member sets from ``quantile_backtest``,
            used to compute the actual turnover of each group.
        cost_per_trade: One-way cost per unit of turnover. Default is 0.001.

    Returns:
        A new dataframe with transaction costs deducted from each quantile and
        from the ``long_short`` spread.

    Notes:
        Each group's charge is ``turnover * 2 * cost_per_trade`` (buys plus
        sells). The spread column is charged for both legs, so its turnover is
        the sum of the Q1 and Q5 turnovers.
    """
    turnover_q1 = calculate_turnover(ticker_history=ticker_history, group='Q1')
    turnover_q5 = calculate_turnover(ticker_history=ticker_history, group='Q5')
    result_after_cost = result_df.copy()
    for col in result_df.columns:
        
        if col == 'long_short':
            cost = (turnover_q5.values + turnover_q1.values)*2*cost_per_trade
            result_after_cost[col] = result_df[col].values - cost
        else:
            turnover = calculate_turnover(ticker_history=ticker_history, group = col)
            cost = turnover.values * 2 * cost_per_trade
            result_after_cost[col] = result_df[col].values - cost
    return result_after_cost

if __name__ == "__main__":
        
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    close_data=pd.read_parquet("data/processed/processed_close.parquet")
    factor_data=pd.read_parquet("tmp/factors/factors.parquet")
    #历史成分股表不在repo内(数据许可), 用环境变量覆盖默认相对路径
    historical_df = pd.read_csv(os.environ.get('SP500_MEMBERSHIP_CSV', '../sp500/sp500_ticker_start_end.csv'))
    
    back_test_path = os.path.join(os.getcwd(), 'tmp/back_test')
    os.makedirs(back_test_path, exist_ok=True)
    
    ticker_list=close_data.columns
    periods=[1,5,20]
    #原始因子算IC，用于正交化
    different_holding_period_df , _  = different_holding_period(close=close_data, tickers = close_data.columns.tolist(), periods=periods)
    # #原始因子算IC，用于正交化
    cs_df = CS_Information_Correlation(factors=factor_data, different_holding_period=different_holding_period_df, output_path="tmp/ic_test/cs_df.parquet")

    train_end = "2023-12-31" 
    test_start = '2024-01-01'
    train_test_analysis_result = train_test_analysis(cs_df= cs_df, factor_ticker=factor_data ,close = close_data, train_end=train_end, test_start= test_start)

    significant_factors = train_test_analysis_result['significant_factors']
    orth_factors_test = train_test_analysis_result['orthogonalize_result_full']
    orth_factors_test = orth_factors_test[orth_factors_test.index >= f'{test_start}']
    dhp_test = train_test_analysis_result['different_holding_period_test']

    with pd.ExcelWriter('tmp/ic_test/stationary.xlsx') as w:
        train_test_analysis_result['rolling_ic_train'].to_excel(w, sheet_name="rolling_ic")
        train_test_analysis_result['acf_train'].to_excel(w,sheet_name="acf")
        train_test_analysis_result['yearly_train'].to_excel(w, sheet_name = "yearly")
    
    test_result = {}
    test_ticker_history = {}
    for period in periods:
        result, history = quantile_backtest(historical_df ,orth_factors_test, significant_factors, dhp_test, period)
        test_result[f'{period}DaysHoldingPeriod'] = result
        test_ticker_history[f'{period}DaysHoldingPeriod'] = history
    
    with open(os.path.join(back_test_path,'test_ticker_history.pkl'), 'wb') as f:
       pickle.dump(test_ticker_history, f)

    # daily_wide = expand_to_daily_returns(test_ticker_history, close_data)
    # daily_wide.to_parquet(os.path.join(back_test_path, 'daily_wide.parquet'))


    # with open(os.path.join(back_test_path,'test_result.pkl'), 'wb') as f:
    #     pickle.dump(test_result, f)
    
    # for period_key, factor_result in test_result.items():
    #     period_num = int(period_key.replace('DaysHoldingPeriod', ''))
    #     for factor_name, df in factor_result.items():
    #         print(f'---{factor_name}{period_key} TEST indicator---')
    #         mono = monotonicity_test(df)
    #         mono_df = pd.DataFrame([mono])
    #         mono_df.to_parquet(os.path.join(back_test_path, f'{period_key}_{factor_name}_mono.parquet'))
    #         print('Monotonicity:', mono)

    #         summary_before_df, nav = performance_summary(df, period_num)
    #         summary_before_df.to_parquet(os.path.join(back_test_path, f'{period_key}_{factor_name}_summary_before_cost.parquet'))
    #         print("Before transaction cost:", summary_before_df)

    #         ticker_history = test_ticker_history[period_key][factor_name]
    #         df_after_cost = apply_transcation_cost(df, ticker_history)  
    #         summary_after_df, _ = performance_summary(df_after_cost, period_num)
    #         summary_after_df.to_parquet(os.path.join(back_test_path, f'{period_key}_{factor_name}_summary_after_cost.parquet'))
    #         print('After transaction cost:', summary_after_df)
        




    