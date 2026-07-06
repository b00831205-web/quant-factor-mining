import pandas as pd


def daily_return(df:pd.DataFrame, tickers:list)->pd.DataFrame:
    """Compute daily percentage returns for a list of tickers.

    Args:
        df: Price dataframe indexed by date.
        tickers: Tickers to compute returns for.

    Returns:
        A dataframe of daily returns with one column per ticker.

    Notes:
        The first row for each ticker will be ``NaN`` because percentage change
        requires a previous observation.
    """
    cols = [t for t in tickers if t in df.columns]
    return df[cols].pct_change()

def excess_return(daily_return: pd.DataFrame) -> pd.DataFrame:
    """Compute excess returns over the market proxy SPY.

    Args:
        daily_return: Dataframe of daily returns that includes an ``SPY`` column.

    Returns:
        A dataframe of excess returns for all non-SPY columns.

    Notes:
        This function assumes ``SPY`` is the benchmark and removes it from the
        output.
    """
    market_return=daily_return['SPY']
    excess_return=daily_return.drop(columns=['SPY']).sub(market_return, axis=0)
    return excess_return

def momentum(df: pd.DataFrame, tickers: list, day:int =2) -> pd.DataFrame:
    """Compute a simple momentum factor for each ticker.

    Args:
        df: Price dataframe indexed by date.
        tickers: Tickers to process.
        day: Lookback window used to measure momentum.

    Returns:
        A dataframe with columns named like ``{day}DayMomentum_TICKER``.

    Notes:
        The factor is based on the relative price change between the current
        price and the price ``day - 1`` periods ago.
    """
    cols = [t for t in tickers if t in df.columns]
    mmt = df[cols].pct_change(day - 1)
    mmt.columns = [f"{day}DayMomentum_{t}" for t in cols]
    return mmt
    
def EWMA(window_data, wk: float)->float: #做.apply()时由于切分下来的nparray会直接传入第一个参数，因此要把window_data写在最前面
    """Compute an exponential weighted moving average over a window.

    Args:
        window_data: One-dimensional numpy array passed by ``rolling.apply``.
        wk: Exponential decay factor.

    Returns:
        The weighted sum for the supplied window.

    Notes:
        The function is designed for use with ``Series.rolling(...).apply(...)``
        and expects the newest observation to contribute the most weight.
    """
    ewma=0
    period = len(window_data)
    for i in range(period):
        ewma += wk**(i)*window_data[period-i-1]
    return ewma

def ShortTermReversal(excess_return: pd.DataFrame,tickers: list, halflife: int, period: int)->pd.DataFrame:
    """Compute a short-term reversal factor from lagged excess returns.

    Args:
        excess_return: Dataframe of excess returns.
        tickers: Tickers to process.
        halflife: Halflife used to define the exponential decay weight.
        period: Rolling window length.

    Returns:
        A dataframe with one short-term reversal column per ticker.

    Notes:
        The signal is negated because high recent returns are assumed to mean
        lower near-term future returns.
    """
    cols = [t for t in tickers if t in excess_return.columns]
    data = excess_return[cols]
    wk = 0.5 ** (1 / halflife)
    # 固定权重FIR滤波: sum_i wk^i * x.shift(i), 每个lag对整个frame做一次shift,
    # 等价于rolling(period).apply(EWMA)但避免了逐窗口的Python循环
    # (O(dates*tickers*period)次Python调用 -> period次向量化操作)。
    # NaN语义一致: 窗口内任一观测缺失则结果为NaN。
    ewma = data * 1.0
    for i in range(1, period):
        ewma = ewma + (wk ** i) * data.shift(i)
    return -ewma #反转因子信号是负的——过去收益率高，预期未来会回落

'''
def get_short_term_reversal_ewma(excess_return, ticker, period, halflife):
    if ticker not in excess_return.columns:
        return None
    
    price = excess_return[ticker]
    
    def ewma_func(window):
        weights = np.array([0.5**(1/halflife) ** (len(window) - 1 - i) for i in range(len(window))])
        weights /= weights.sum() #做权重归一化，不做归一化计算出来的EWMA值会受窗口期长短影响——窗口越长，权重之和越大，算出来的值越大，不同窗口期的值没有可比性
        return -np.dot(weights, window)
    
    return price.rolling(period).apply(ewma_func, raw=True)
    '''

def TwentyDayVolatility(daily_return:pd.DataFrame, tickers:list)->pd.DataFrame: #获取20日波动率
    """Compute 20-day rolling volatility for each ticker.

    Args:
        daily_return: Dataframe of daily returns.
        tickers: Tickers to process.

    Returns:
        A dataframe of 20-day rolling standard deviations.
    """
    cols = [t for t in tickers if t in daily_return.columns]
    return daily_return[cols].rolling(20).std()

def TwentyDayNegVotality(daily_return:pd.DataFrame, tickers:list)->pd.DataFrame: #获取20日负收益波动率
    """Compute 20-day rolling volatility using only negative returns.

    Args:
        daily_return: Dataframe of daily returns.
        tickers: Tickers to process.

    Returns:
        A dataframe of rolling standard deviations calculated from negative
        returns only.

    Notes:
        Non-negative returns are converted to ``NaN`` before rolling.
    """
    cols = [t for t in tickers if t in daily_return.columns]
    neg_return = daily_return[cols].where(daily_return[cols] < 0) #满足cond的保留，不满足的变成NaN
    return neg_return.rolling(window=20, min_periods=1).std()

def TwentyDayAvgVol(volume:pd.DataFrame, tickers:list)->pd.DataFrame: #20日平均成交量因子
    """Compute 20-day average trading volume for each ticker.

    Args:
        volume: Volume dataframe indexed by date.
        tickers: Tickers to process.

    Returns:
        A dataframe of 20-day rolling mean volume values.
    """
    cols = [t for t in tickers if t in volume.columns]
    return volume[cols].rolling(20).mean()

def VolPriceCorr(volume:pd.DataFrame, daily_return:pd.DataFrame, tickers:list)->pd.DataFrame: #20日量价相关系数
    """Compute 20-day rolling correlation between returns and volume.

    Args:
        volume: Volume dataframe indexed by date.
        daily_return: Daily return dataframe indexed by date.
        tickers: Tickers to process.

    Returns:
        A dataframe of rolling correlation values for each ticker.

    Notes:
        This factor measures the relationship between trading activity and
        price movement over a 20-day window.
    """
    cols = [t for t in tickers if t in daily_return.columns and t in volume.columns]
    return daily_return[cols].rolling(20).corr(volume[cols])