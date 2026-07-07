import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  #以脚本方式执行时定位repo根目录

from quantfactor.data_acquisition import data_acquisition
import argparse
import datetime
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    parse=argparse.ArgumentParser(description="archive raw data using start date and the batch")
    parse.add_argument("--date", type=str, required=True)
    parse.add_argument("--batch", type=str, required=True)
    args=parse.parse_args()

    base_dir=os.getcwd()
    tmp_dir = os.path.join(base_dir, "data")
    os.makedirs(tmp_dir, exist_ok=True)
    raw_close_path=os.path.join(tmp_dir, 'raw',"raw_close.parquet")
    raw_volume_path=os.path.join(tmp_dir, 'raw',"raw_volume.parquet")
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    
    close_path=os.path.join(tmp_dir, 'processed',"close.parquet")
    volume_path=os.path.join(tmp_dir, 'processed',"volume.parquet")
    
    r = requests.get(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        headers={'User-Agent': 'Mozilla/5.0'}
    )

    print("close_path:", close_path)
    print("exists:", os.path.exists(close_path))
    if os.path.exists(close_path):
        existing = pd.read_parquet(close_path)
        print("existing date range:", existing.index.min(), "-", existing.index.max())
    historical_data = pd.read_csv('/mnt/e/Handout/project/project_quant/sp500/sp500_ticker_start_end.csv')
    historical_data['start_date'] = pd.to_datetime(historical_data['start_date'])
    historical_data['end_date'] = pd.to_datetime(historical_data['end_date'])

    ANALYSIS_START = pd.Timestamp('2015-01-01')

# 只保留在分析窗口内相关的股票（仍在指数 或 end_date在分析起点之后）
    relevant = historical_data[
        historical_data['end_date'].isnull() |
        (historical_data['end_date'] >= ANALYSIS_START)
    ]
    tickers = set(relevant['ticker'].unique())
    # Yahoo Finance 格式：BRK.B → BRK-B
    tickers = [t.replace('.', '-') for t in tickers]
    tickers.append('SPY')  # 保留基准，删掉这行如果不需要
    print(f"Loaded {len(tickers)} tickers")

    if os.path.exists(close_path) and os.path.exists(volume_path):
        existing_close = pd.read_parquet(close_path)
        existing_volume = pd.read_parquet(volume_path)
        last_date = min(existing_close.index.max(),existing_volume.index.max())
        start_date=((last_date + datetime.timedelta(days=1))).strftime("%Y-%m-%d")
    else:
        start_date="2015-01-01"
    
    if start_date >= end_date:
        print(f"{args.batch} is newest, no need to be updated")
    else:   
        close, volume =data_acquisition(tickers = tickers, start_date=start_date, end_date=end_date, batch_size=50)
        close.to_parquet(close_path)
        volume.to_parquet(volume_path)
        print(f"data has been download to temp path {raw_close_path}, {raw_volume_path}")