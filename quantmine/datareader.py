from dataclasses import dataclass
import pandas as pd
from typing import TypeAlias, Protocol
from abc import ABC, abstractmethod

@dataclass
class MarketData:
    close: pd.DataFrame = None
    volume: pd.DataFrame = None
    market_cap : pd.DataFrame = None
    def require(self, *fields: str)->None:
        for field in fields:
            if getattr(self, field) is None:
                raise ValueError(f'You have not provided the data for "{field}"')
            
class DataSource(Protocol):
    def load(self, tickers: list[str], start: str, end: str )-> MarketData:
        ... #ellipsis: 占位符，约定俗成这个函数体是空的，只是展位

class LocalFileSource(ABC):
    def __init__(self, paths: dict[str]):
        self.paths = paths
    @abstractmethod
    def _read_file(self, path:str)->pd.DataFrame:
        ...
    def load(self, tickers: list[str] =None, start: str=None, end:str= None)-> MarketData:
        loaded = {}
        for filed_name, path in self.paths.items():
            df = self._read_file(path)
            if tickers:
                loaded[filed_name] = df.loc[start:end, tickers]
            else: 
                loaded[filed_name] = df.loc[start:end]
        return MarketData(**loaded)

class YFinanceSource:
    def __init__(self, batch_size: int =50, max_retries: int=3, wait: int=60, checkpoint_dir:str="tmp/checkpoint"):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.wait = wait
        self.checkpoint_dir = checkpoint_dir
    def load(self, tickers: list[str], start: str, end: str) ->MarketData:
        #惰性import: 只用本地文件源的用户不必安装yfinance依赖链
        from data_acquisition import data_acquisition
        close, volume = data_acquisition(tickers=tickers, start_date = start, end_date = end, batch_size= self.batch_size,
                                    max_retries= self.max_retries, wait= self.wait, checkpoint_dir=self.checkpoint_dir)
        return MarketData(close = close, volume = volume)

class ParquetSource(LocalFileSource):
    def _read_file(self, path:str) -> pd.DataFrame:
        return pd.read_parquet(path)

class CSVSource(LocalFileSource):
    def _read_file(self, path):
        return pd.read_csv(path, index_col = 0, parse_dates= True)

class ExcelSource(LocalFileSource):
    def _read_file(self, path):
        return pd.read_excel(path, index_col=0, parse_dates= True)

class ConstituentsSource(Protocol):
    """Point-in-time股票池数据源协议。

    回测在每个调仓日调用get_constituents(date)获取当日有效宇宙,
    任何实现了该方法的对象都可以接入(静态名单/区间表/实时API)。
    """
    def get_constituents(self, date: pd.Timestamp) -> set[str]:
        ...

class StaticUniverse:
    """固定股票池: 没有point-in-time成分股数据的用户的最简实现。

    注意: 固定宇宙意味着回测带有幸存者偏差, 结论解读时需要声明。
    """
    def __init__(self, tickers: list[str] | set[str]):
        self._tickers = set(tickers)
    def get_constituents(self, date: pd.Timestamp) -> set[str]:
        return self._tickers

class MembershipTableSource:
    """区间表实现: 从 (ticker, start_date, end_date) 表构造point-in-time宇宙。

    end_date为空表示至今仍在指数内。日期解析在构造时一次完成
    (每次查询重复parse是逐调仓日的全表开销), 查询结果按日期缓存。

    Args:
        table: 含ticker/start_date/end_date三列的DataFrame。
        normalize: True时把ticker里的'.'替换成'-'(如BRK.B -> BRK-B),
            与yfinance列名对齐; 若你的行情数据用'.'写法则传False。
    """
    def __init__(self, table: pd.DataFrame, ticker_col: str = 'ticker',
                 start_col: str = 'start_date', end_col: str = 'end_date',
                 normalize: bool = True):
        self._start = pd.to_datetime(table[start_col])
        self._end = pd.to_datetime(table[end_col])
        tickers = table[ticker_col].astype(str)
        if normalize:
            tickers = tickers.str.replace('.', '-', regex=False)
        self._tickers = tickers
        self._cache: dict[pd.Timestamp, set[str]] = {}

    def get_constituents(self, date: pd.Timestamp) -> set[str]:
        date = pd.Timestamp(date)
        if date not in self._cache:
            #end条件必须整体括起来, 否则end>=date的行会绕过start<=date的检查
            mask = (self._start <= date) & (self._end.isnull() | (self._end >= date))
            self._cache[date] = set(self._tickers[mask])
        return self._cache[date]

FactorDict: TypeAlias = dict[str, pd.DataFrame]
ForwardReturn: TypeAlias = dict[int, pd.DataFrame]




if __name__ == "__main__":
    yfinance_data = YFinanceSource()
    apple_data = yfinance_data.load(tickers = ['AAPL'],start = '2026-07-02', end = '2026-07-08')
    print(apple_data)