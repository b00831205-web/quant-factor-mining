import pandas as pd
from .datareader import ConstituentsSource, MembershipTableSource
import numpy as np
from scipy.stats import spearmanr


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
        

def quantile_backtest(constituents: ConstituentsSource | pd.DataFrame | None ,factors: dict[str, pd.DataFrame], significant_factor_list:list, forward_returns: dict[int,pd.DataFrame], part:int =5):
    """Run a quantile long-short backtest for each selected factor.

    Args:
        constituents: Point-in-time universe source implementing
            ``get_constituents(date) -> set[str]``. A raw membership dataframe
            (ticker/start_date/end_date columns) is auto-wrapped in
            ``MembershipTableSource``. Pass ``None`` to use every factor
            column (survivorship-biased fixed universe).
        factors: Mapping of factor name to a date-by-ticker value dataframe.
        significant_factor_list: Factor names to evaluate.
        forward_returns: Mapping of holding period (in trading days) to a
            date-by-ticker forward return dataframe.
        part: Number of quantile groups to form. Default is 5.

    Returns:
        A tuple ``(all_result, all_ticker_history)``. Both are keyed by
        ``(factor_name, period)``; results hold Q1..Qn plus ``long_short``
        returns per rebalance date, histories hold the member set per group.
    """
    if isinstance(constituents, pd.DataFrame):
        constituents = MembershipTableSource(constituents)

    all_result={}
    all_ticker_history = {}

    for significant_factor in significant_factor_list:
        factor_df = factors[significant_factor]
        for period, forward_return_df in forward_returns.items():
            result=[]
            ticker_history = []
            for index in range(0, len(factor_df), period):
                curr_date = factor_df.index[index]
                if curr_date not in forward_return_df.index:
                    continue
                if constituents is None:
                    available_tickers = list(factor_df.columns)
                else:
                    valid_tickers = constituents.get_constituents(curr_date)
                    #按factor列序取交集: valid_tickers是set, 直接遍历顺序不确定,
                    #并列排名经稳定排序后分位边界成员会随运行漂移, 结果不可复现
                    available_tickers = [t for t in factor_df.columns if t in valid_tickers]

                if factor_df.iloc[index].isnull().all():
                    continue
                cross_section_factor = factor_df.loc[curr_date, available_tickers]
                ranked_cross_section = cross_section_factor.rank(ascending=True)
                ranked_cross_section = ranked_cross_section.sort_values()
                tickers = ranked_cross_section.index

                group_list = truncate_quantile(tickers ,part = part)
                
                _return = {'date':curr_date}    
                _tickers = {'date':curr_date}
                for i in range(part):
                    group_return = forward_return_df.loc[curr_date, group_list[i]]
                    _return[f'Q{i+1}'] = group_return.mean()
                    _tickers[f'Q{i+1}'] = set(group_list[i])
                ticker_history.append(_tickers)
                result.append(_return)
            result_df = pd.DataFrame(result).set_index('date')
            result_df['long_short'] = result_df['Q5']-result_df['Q1']
            all_ticker_history[(significant_factor,period)] = ticker_history
            all_result[(significant_factor,period)] = result_df
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

def expand_all_to_daily_returns(
    all_ticker_history: dict,
    close_data: pd.DataFrame,
    cost_per_trade: float = 0.001,
) -> dict:
    """
    对 quantile_backtest 产出的 all_ticker_history（key是(factor, period)元组），
    批量展开成逐日收益，不需要手动一个个取。

    Returns:
        dict[(factor, period), pd.DataFrame] —— 和输入结构对应，
        每个key对应一份展开后的逐日Q1-Q5+long_short收益表
    """
    all_daily_returns = {}
    for key, ticker_history in all_ticker_history.items():
        all_daily_returns[key] = expand_to_daily_returns(ticker_history, close_data, cost_per_trade)
    return all_daily_returns

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


def back_test_senity_test(constituents: ConstituentsSource | pd.DataFrame | None ,significant_factor_list:list, factors: dict[str,pd.DataFrame] ,forward_returns: dict[int,pd.DataFrame], close:pd.DataFrame, periods:list,
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
    factors_shifting = {}
    for factor_name, factor in factors.items():
        factors_shifting[factor_name] = factor.shift(-1)
    #shuffle基于原始因子, 且必须逐frame深拷贝:
    #dict.copy()是浅拷贝, in-place打乱会把factors_shifting的frame一起改掉
    shuffle_data = {factor_name: factor.copy() for factor_name, factor in factors.items()}
    for factor_name, factor in shuffle_data.items():
        for idx in factor.index:
            factor.loc[idx] = np.random.permutation(factor.loc[idx].values)
    


    period_difference = {}

    
    for period in periods:
        #holding period return and close data shifting difference
        
        raw_return = pd.DataFrame({ticker : close[ticker].pct_change(period).shift(-period) for ticker in tickers})
        common_col = forward_returns[period].columns.intersection(raw_return.columns)
        diff = (forward_returns[period][common_col]-raw_return[common_col]).abs().sum().sum() #第一个sum得到series，第二个sum把所有series相加得到标量
        period_difference[period] = diff

    #factor displacement
    factor_displacement_result , _ = quantile_backtest(constituents, factors_shifting,significant_factor_list, forward_returns)

    #shuffle data
    shuffle_data_return, _ = quantile_backtest(constituents,shuffle_data, significant_factor_list, forward_returns)

    total_difference = 0
    for period, diff in period_difference.items():
        total_difference += diff
    
    displace_difference = {}
    for (factor_name, period), df in factor_displacement_result.items():
        displace_difference[(factor_name,period)] = df - origincal_back_test[(factor_name, period)]
    
    shuffle_difference = {}
    for (factor_name, period), df in shuffle_data_return.items():
        shuffle_difference[(factor_name,period)] = df - origincal_back_test[(factor_name, period)]
    
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