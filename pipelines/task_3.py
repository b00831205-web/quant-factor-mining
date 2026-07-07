import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  #以脚本方式执行时定位repo根目录

from quantfactor import factor_mining
import pandas as pd
import os
import argparse

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="calculate quantitative factors")
    parse.add_argument("--date", type= str, required=True)
    parse.add_argument("--batch", type= str, required=True)
    parse.add_argument("--days", type=int, required=False, default = 5)
    parse.add_argument("--halflife", type=int, required=False, default = 10)
    parse.add_argument("--period", type=int, required=False, default =20)
    args = parse.parse_args()
    

    close_path = os.path.join(os.getcwd(),'data/processed', "processed_close.parquet")
    volume_path = os.path.join(os.getcwd(),'data/processed', "processed_volume.parquet")
    final_path = os.path.join(os.getcwd(),'tmp/factors', "factors.parquet")

    def calculate_factors(close:pd.DataFrame, volume:pd.DataFrame, tickers:list ,days: int, half_life:int , period: int )->pd.DataFrame:
        batch = pd.DataFrame()
        daily_return=factor_mining.daily_return(close, tickers)
        excess_return=factor_mining.excess_return(daily_return)
        momentum=factor_mining.momentum(close, tickers, day=days)
        ShortTermReversal = factor_mining.ShortTermReversal(excess_return=excess_return, tickers=tickers, halflife=half_life, period=period)
        TwentyDayVolatility = factor_mining.TwentyDayVolatility(daily_return=daily_return, tickers=tickers)
        TwentyDayNegVotality = factor_mining.TwentyDayNegVotality(daily_return=daily_return, tickers=tickers)
        TwentyDayAvgVol = factor_mining.TwentyDayAvgVol(volume=volume, tickers=tickers)
        VolPriceCorr=factor_mining.VolPriceCorr(volume=volume,daily_return=daily_return,tickers=tickers)
        batch = pd.concat([daily_return.add_prefix("DailyReturn_"), 
                           excess_return.add_prefix("ExcessReturn_"), 
                           momentum, 
                           ShortTermReversal.add_prefix("ShortTermReversal_"), 
                           TwentyDayVolatility.add_prefix("TwentyDayVolatility_"), 
                           TwentyDayNegVotality.add_prefix("TwentyDayNegVolatility_"), 
                           TwentyDayAvgVol.add_prefix("TwentyDayAvgVol_"), 
                           VolPriceCorr.add_prefix("VolPriCorr_")],
                           axis=1)
        return batch

    
    def data_proceeding(close_path: str, volume_path: str, final_path:str) -> pd.DataFrame: #因子计算需要历史窗口,不能只取新数据。
        if not os.path.exists(close_path) or not os.path.exists(volume_path):
            return "file not found"
        else:
            close=pd.read_parquet(close_path)
            volume=pd.read_parquet(volume_path)
            tickers=close.columns
            all_factors=calculate_factors(close=close,volume=volume,tickers=tickers, days=args.days, half_life=args.halflife, period=args.period)
            all_factors.to_parquet(final_path)
            print("data_proceeding_complete")
        return all_factors
    
    data_proceeding(close_path=close_path, volume_path=volume_path, final_path=final_path )