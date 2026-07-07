import pandas as pd
import os
from scipy import stats
import numpy as np
from statsmodels.stats.multitest import multipletests
import warnings
import re
warnings.filterwarnings('ignore')

def different_holding_period(close:pd.DataFrame, tickers: list, periods: list | int = [1,5,20])->pd.DataFrame:
    """Build forward holding-period returns for each ticker.

    Args:
        close: Close price dataframe indexed by date with one column per ticker.
        tickers: Ticker list to process.
        periods: Holding periods in trading days. Can be a list or a single int.

    Returns:
        A tuple of ``(holding_period_dataframe, periods)``. The dataframe
        contains columns named like ``{period}DaysHoldingPeriod_TICKER``.

    Notes:
        The return series are shifted forward, so the value at date ``t`` is the
        return realized over the next ``period`` trading days.
    """
    cols = [t for t in tickers if t in close.columns]
    blocks = []
    for period in periods:
        # 整个frame一次pct_change, 替代逐列插入(逐列插入会触发DataFrame碎片化, 570*3列时极慢)
        block = close[cols].pct_change(period).shift(-period)
        block.columns = [f"{period}DaysHoldingPeriod_{t}" for t in cols]
        blocks.append(block)
    different_holding = pd.concat(blocks, axis=1)
    return different_holding, periods

def data_standarization(df:pd.DataFrame)->pd.DataFrame:
    """Cross-sectionally standardize factor columns to the range [-1, 1].

    Args:
        df: Factor dataframe with columns named as ``factor_ticker``.

    Returns:
        A dataframe with the same shape, where each factor group is ranked by
        row and scaled to the interval ``[-1, 1]``.

    Notes:
        This function assumes the column naming convention is consistent and
        uses rank-based scaling, which is less sensitive to outliers than raw
        value normalization.
    """
    factors = list(set(cols.rsplit("_",1)[0] for cols in df.columns))
    tickers = list(set(cols.rsplit("_",1)[1] for cols in df.columns))
    blocks = []
    for factor in factors:
        cols = [f'{factor}_{ticker}' for ticker in tickers]
        cols = [c for c in cols if c in df.columns]
        ranked = df[cols].rank(axis=1,pct=True)*2-1
        blocks.append(ranked)
        #最大值归一化受极端值影响大，不建议使用
    return pd.concat(blocks, axis=1) #一次concat替代逐块列插入, 避免碎片化

def TM_Information_correlation(tickers : list, factors: pd.DataFrame, different_holding_period: pd.DataFrame, output_path=str)->pd.DataFrame:
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
    TM_IC_matrix=pd.DataFrame() #时序IC
    
    def ticker_dict(tickers: list, df:pd.DataFrame)->dict:
        tickers_dict={ticker:[] for ticker in tickers}
        for ticker in tickers:
            for column in df.columns:
                if column.rsplit("_",1)[1] == ticker:
                    tickers_dict[ticker].append(column)
        return tickers_dict
    
    ticker_factor_dict={}
    for ticker, factor in ticker_dict(tickers, factors).items():
        if factor:
            df = factors[factor].copy()
            df.columns=[c.rsplit('_',1)[0] for c in factor]
            ticker_factor_dict[ticker] = df
    
    ticker_holding_dict={}
    for ticker, holding_period in ticker_dict(tickers, different_holding_period).items():
        if holding_period:
            df = different_holding_period[holding_period].copy()
            df.columns=[c.rsplit('_',1)[0] for c in holding_period]
            ticker_holding_dict[ticker] = df
    
    result=[]
    for ticker, factor in ticker_factor_dict.items():
        holding_df =ticker_holding_dict[ticker]
        for h_col in holding_df.columns:
            ic = factor.corrwith(holding_df[h_col],method="pearson",axis =0)
            ic.name=f'{ticker}_{h_col}'
            result.append(ic)
    TM_IC_matrix = pd.concat(result,axis=1)
    
    TM_IC_matrix.to_parquet(os.path.join(os.getcwd(), output_path))
    print("Time-series information correlation computation complete")

    return TM_IC_matrix

def CS_Information_Correlation(factors:pd.DataFrame, different_holding_period: pd.DataFrame, output_path: str)-> pd.DataFrame:
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
    CS_IC_matrix=pd.DataFrame() #截面IC
    
    #获取ticker和因子列表
    ticker_list=[]
    factor_list=[]
    for column in factors.columns:
        factor, ticker = column.rsplit("_",1)
        factor_list.append(factor)
        ticker_list.append(ticker)
    factor_list=list(set(factor_list))
    ticker_list=list(set(ticker_list))
    
    #获取持有期列表
    holding_period_list =[]
    for column in different_holding_period.columns:
        holding_period, ticker = column.rsplit("_",1)
        holding_period_list.append(holding_period)
    holding_period_list = list(set(holding_period_list))
    
    def factor_dict(factors: list, df:pd.DataFrame) ->dict:
        factors_dict={factor: [] for factor in factors}
        for factor in factors:
            for column in df.columns:
                if column.rsplit("_",1)[0] == factor:
                    factors_dict[factor].append(column)
        return factors_dict
    
    ticker_factor_dict={}
    for factor, ticker in factor_dict(factor_list, factors).items():
        if ticker:
            df = factors[ticker].copy()
            df.columns=[c.rsplit('_',1)[1] for c in ticker]
            ticker_factor_dict[factor] = df
    
    ticker_holding_dict={}
    for holding_period, ticker in factor_dict(holding_period_list, different_holding_period).items():
        if ticker:
            df = different_holding_period[ticker].copy()
            df.columns=[c.rsplit('_',1)[1] for c in ticker]
            ticker_holding_dict[holding_period] = df

    result=[]
    for factor, factor_df in ticker_factor_dict.items():
        for holding_period , holding_period_ticker in ticker_holding_dict.items():
            ic_series= factor_df.corrwith(holding_period_ticker[factor_df.columns], method='pearson',axis=1)
            ic_series.name=f"{factor}_{holding_period}"
            result.append(ic_series)
    CS_IC_matrix=pd.concat(result, axis=1)

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
    result = {}
    for period in periods:
        columns = [c for c in cross_section_IC.columns if c.rsplit('_',1)[1]==f"{period}DaysHoldingPeriod"]
        period_df = cross_section_IC[columns]
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
        period = int(re.search(r'(\d+)DaysHoldingPeriod$', col).group(1))
        lag = lag_multiplier * max(period - 1, 0)
        s = cross_section_IC[col].dropna()
        n = len(s)
        mu = s.mean()
        e = (s - mu).values
        var = e @ e / n #lag=0时就是普通样本方差
        for l in range(1, min(lag, n - 1) + 1):
            w = 1 - l / (lag + 1) #Bartlett核权重, 保证方差估计非负
            var += 2 * w * (e[:-l] @ e[l:]) / n
        se = np.sqrt(var / n)
        rows[col] = {'IC_mean': mu, 'IC_std': s.std(), 'IR': mu / s.std(),
                     'n': n, 'lag': lag, 'NW_t': mu / se}
    nw_df = pd.DataFrame(rows).T
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
    if 'NW_t' in summary_df.columns: #NW路径: 用自相关修正后的t和p(newey_west_summary的输出)
        significant_t = pd.DataFrame({
            "t": summary_df['NW_t'],
            "p_value": summary_df['p_value']
            })
    else: #iid路径: 抽样降频后的独立样本t检验(resample_summary的输出)
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

def orthogonal_analysis(factors_ticker: pd.DataFrame):
    """Compute average factor correlation and identify highly correlated pairs.

    Args:
        factors_ticker: Factor dataframe with ``factor_ticker`` columns.

    Returns:
        A tuple of ``(average_correlation_matrix, high_correlation_dict)``.

    Notes:
        Dates with missing factor columns are skipped. Factors whose absolute
        average correlation exceeds 0.5 are treated as highly correlated.
    """
    factors = sorted(set(col.rsplit('_',1)[0] for col in factors_ticker.columns))
    tickers = sorted(set(col.rsplit('_',1)[1] for col in factors_ticker.columns))

    #每个因子重排成一张 dates*tickers 的宽表, 列顺序统一, 后续全部是frame级向量化操作
    frames = {}
    for f in factors:
        cols = [c for c in factors_ticker.columns if c.rsplit('_',1)[0] == f]
        block = factors_ticker[cols]
        block.columns = [c.rsplit('_',1)[1] for c in cols]
        frames[f] = block.reindex(columns=tickers)

    #有效日: 每个因子当天至少有一个非NaN值(与原逐日实现的跳过条件一致)
    valid = pd.concat({f: frames[f].notna().any(axis=1) for f in factors}, axis=1).all(axis=1)
    valid_days = int(valid.sum())

    avg_corr = pd.DataFrame(1.0, index=factors, columns=factors) #对角线恒为1
    for i, fa in enumerate(factors):
        a = frames[fa]
        for fb in factors[i+1:]:
            b = frames[fb]
            #逐日截面Pearson相关的向量化展开: 每天在两因子共同非NaN的ticker上算相关
            mask = a.notna() & b.notna()
            xa, xb = a.where(mask), b.where(mask)
            n = mask.sum(axis=1)
            sa, sb = xa.sum(axis=1), xb.sum(axis=1)
            cov = (xa*xb).sum(axis=1) - sa*sb/n
            var_a = (xa*xa).sum(axis=1) - sa*sa/n
            var_b = (xb*xb).sum(axis=1) - sb*sb/n
            corr_t = cov/np.sqrt(var_a*var_b)
            #与原实现一致: 对有效日求和(若某日相关无法计算则整体为NaN)后除以有效日数
            pair_avg = corr_t[valid].to_numpy().sum()/valid_days
            avg_corr.loc[fa, fb] = pair_avg
            avg_corr.loc[fb, fa] = pair_avg

    high_corr_dict={
        column: avg_corr.index[(avg_corr[column].abs()>0.5) & (avg_corr.index!=column)].tolist() for column in avg_corr.columns
    }
    return avg_corr, high_corr_dict

def orthogonalize(factor_df: pd.DataFrame, high_corr_dict: dict, ic_summary:pd.DataFrame, threshold: float = 0.03, min_period: int = 60)->pd.DataFrame:
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
    
    factor_ir = {}
    for raw in ic_summary.index:
        factor = re.sub(r'_\d+DaysHoldingPeriod$','',raw)
        ir = abs(ic_summary.loc[raw, "IR"])
        if factor not in factor_ir:
            factor_ir[factor] = []
        factor_ir[factor].append(ir)
    factor_ir = {f: sum(irs)/len(irs) for f, irs in factor_ir.items()}

    print("Average IR:")
    print({k: round(v,4) for k,v in sorted(factor_ir.items())})
    print(f"threshold={threshold}, will be dropped because it is below the threshold")

    result_df = factor_df.copy()
    drop = set()
    orthogonalized = set()
    
    def drop_columns(result_df:pd.DataFrame, factor_a: str, factor_b:str, drop:set)->pd.DataFrame:
        col_a = [c for c in result_df.columns if c.rsplit("_",1)[0]==factor_a]
        result_df = result_df.drop(columns=col_a)
        drop.add(factor_a)
        return result_df

    for factor_a, factor_b in pairs:
        if factor_a in drop or factor_b in drop:
            continue
        ir_a = factor_ir.get(factor_a, 0)
        ir_b = factor_ir.get(factor_b, 0)

        if ir_a < threshold and ir_b < threshold:
            col_a = [c for c in result_df.columns if c.rsplit('_',1)[0]==factor_a]
            col_b = [c for c in result_df.columns if c.rsplit('_',1)[0]==factor_b]
            result_df = result_df.drop(columns=col_a + col_b)
            drop.update([factor_a, factor_b])
            continue
            
        keeper = factor_a if ir_a > ir_b else factor_b
        to_orthogonalize = factor_b if ir_a > ir_b else factor_a

        if ir_a < threshold:
            result_df = drop_columns(result_df, factor_a, factor_b, drop)

        elif ir_b < threshold:
            
            result_df = drop_columns(result_df, factor_b, factor_a, drop)
                
        if ir_a >= threshold and ir_b >= threshold:
            if to_orthogonalize in orthogonalized:
                continue
            tickers = list(set([t.rsplit("_",1)[1] for t in result_df.columns]))

            print(f"Number of columns before orthogonalization: {len(result_df.columns)}")
            for ticker in tickers:
                x_col = f"{keeper}_{ticker}"
                y_col = f"{to_orthogonalize}_{ticker}"
                if x_col not in result_df.columns or y_col not in result_df.columns:
                    continue
                x = result_df[x_col]
                y = result_df[y_col]
                
                expanding_cov = x.expanding(min_periods= min_period).cov(y)
                expanding_var = x.expanding(min_periods= min_period).var()
                beta_series = expanding_cov/expanding_var
                
                residuals = y - beta_series * x
                if y_col in result_df.columns:
                    result_df = result_df.drop(columns = y_col)
                result_df[y_col] = residuals
            orthogonalized.add(to_orthogonalize)
    return result_df

def time_series_stationary_test(CS_IC_matrix:pd.DataFrame, rolling_period:int =126, periods:list = [1,7,15,20])-> pd.DataFrame:
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
                acf_ic[f"{col}_{period}_ACF"] = CS_IC_matrix[col].corr(CS_IC_matrix[col].shift(period), method="pearson") #corr是对series, corrwith是对dataframe
    acf_df = pd.Series(acf_ic).to_frame(name = 'ACF') #字典的每个value是一个标量，不能直接用pd.DataFrame，需要先转成Series
    yearly_summary={}
    # 分段IC    
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
    start = pd.to_datetime(historical_df['start_date'])
    end = pd.to_datetime(historical_df['end_date'])
    #注意运算符优先级: end条件必须整体括起来, 否则 end>=date 的行会绕过 start<=date 的检查,
    #导致尚未加入指数的股票被提前计入截面
    mask = (start <= date) & (end.isnull() | (end >= date))
    return set(historical_df.loc[mask, 'ticker'])

def train_test_analysis(cs_df: pd.DataFrame, factor_ticker: pd.DataFrame, close: pd.DataFrame , train_end: str, test_start: str, periods: list|int = [1,5,20]):
    """Run the full train/test IC workflow and orthogonalization pipeline.

    Args:
        cs_df: Cross-sectional IC dataframe for all available dates.
        factor_ticker: Full factor dataframe with ``factor_ticker`` columns.
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
    train_cs_df = cs_df[cs_df.index <= train_end]
    test_cs_df = cs_df[cs_df.index >= test_start]

    different_holding_period_train, _ = different_holding_period(close[close.index<=train_end], close.columns, periods)
    different_holding_period_train =data_standarization(different_holding_period_train)
    different_holding_period_test, _ = different_holding_period(close[close.index>=test_start], close.columns, periods)

    train_factor_ticker = factor_ticker[factor_ticker.index <= train_end]
    test_factor_ticker = factor_ticker[factor_ticker.index >= test_start]

    resample_summary_train = resample_summary(train_cs_df, periods)
    print(resample_summary_train)
    print(resample_summary_train.index.tolist()[:5])

    orth_analysis,  high_corr_dict= orthogonal_analysis(train_factor_ticker)
    orthogonalize_result = orthogonalize(factor_ticker, high_corr_dict, resample_summary_train)
    excess_cols = [c for c in orthogonalize_result.columns if c.rsplit("_",1)[0] == "ExcessReturn"]
    orthogonalize_result = orthogonalize_result.drop(columns=excess_cols)
    orth_test = orthogonalize_result[orthogonalize_result.index >= test_start]

    cs_df_orth_train = CS_Information_Correlation(factors = orthogonalize_result[orthogonalize_result.index <= train_end],
                                                  different_holding_period=different_holding_period_train,
                                                  output_path = 'tmp/ic_test/cs_df_orth_train.parquet')
    #主检验: Newey-West(全日频IC, 修正重叠持有期自相关)
    nw_summary_orth_train = newey_west_summary(cs_df_orth_train)
    multiple_testing_train = multiple_testing(nw_summary_orth_train)
    print("=== Newey-West test ===")
    print(multiple_testing_train)
    print("Number of True values in BH_significant:", multiple_testing_train['BH_significant'].sum())

    #稳健性对照: 抽样降频(iloc[::period], 保守但有相位依赖)
    resample_summary_orth_train = resample_summary(cs_df_orth_train, periods)
    multiple_testing_resample = multiple_testing(resample_summary_orth_train)
    print("=== Down-sampling control ===")
    print("Number of True values in BH_significant:", multiple_testing_resample['BH_significant'].sum())
    significant_factor = list(set([idx.rsplit('_',1)[0] for idx in multiple_testing_train.index if multiple_testing_train.loc[idx, 'BH_significant'] == True]))

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
            "different_holding_period_train" : different_holding_period_train,
            "different_holding_period_test" : different_holding_period_test,
            "significant_factors": significant_factor,
            "rolling_ic_train": rolling_ic_train,
            "acf_train": acf_train,
            "yearly_train" : yearly_train}

    

if __name__=='__main__':
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    close_data=pd.read_parquet("data/processed/processed_close.parquet")
    factor_data=pd.read_parquet("tmp/factors/factors.parquet")
    
    ticker_list=close_data.columns
    periods=[1,5,20]

    different_holding_period_df , _  = different_holding_period(close=close_data, tickers = close_data.columns.tolist(), periods=periods)
    # #原始因子算IC，用于正交化
    cs_df = CS_Information_Correlation(factors=factor_data, different_holding_period=different_holding_period_df, output_path="tmp/ic_test/cs_df.parquet")

    train_end = '2023-12-31'
    test_start = '2024-01-01'

    train_test_analysis_result = train_test_analysis(cs_df= cs_df, factor_ticker=factor_data ,close = close_data, train_end=train_end, test_start= test_start)

    # with pd.ExcelWriter('tmp/ic_test/stationary.xlsx') as w:
    #     train_test_analysis_result['rolling_ic_train'].to_excel(w, sheet_name="rolling_ic")
    #     train_test_analysis_result['acf_train'].to_excel(w,sheet_name="acf")
    #     train_test_analysis_result['yearly_train'].to_excel(w, sheet_name = "yearly")
    
    # train_test_analysis_result['multiple_testing_train'].to_parquet('tmp/ic_test/significant_test.parquet')
    # print("Done !!!")
    # resample_summary_train = resample_summary(train_cs_df, periods)
    # print(resample_summary_train.index.tolist()[:5])

    
