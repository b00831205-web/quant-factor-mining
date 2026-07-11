import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  #locate the repo root when executed as a script

from quantmine.data_acquisition import retry_batches, merge_checkpoints
import argparse
import datetime
import os
import pandas as pd


if __name__ == "__main__":
    parse=argparse.ArgumentParser(description="archive raw data using start date and the batch")
    parse.add_argument("--date", type=str, required=True)
    parse.add_argument("--batch", type=str, required=True)
    args=parse.parse_args()

    tmp_dir=os.path.join(os.getcwd(),"tmp")
    checkpoint_dir = os.path.join(tmp_dir, "checkpoint")
    raw_close_path = os.path.join(tmp_dir, "raw_close.parquet")
    raw_volume_path = os.path.join(tmp_dir, "raw_volume.parquet")


    if os.path.exists(raw_close_path): #resume from the last failed checkpoint: start where the previous run stopped
        existing = pd.read_parquet(raw_close_path)
        start_date = (existing.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d") 
    else:
        start_date="2021-01-01"
    close, volume =retry_batches(start_date=start_date, end_date=args.date, max_retries=3)
    if close is not None and volume is not None:
        if os.path.exists(raw_close_path) and os.path.exists(raw_volume_path):
            curr_close = pd.read_parquet(raw_close_path)
            curr_volume = pd.read_parquet(raw_volume_path)
            pd.concat([curr_close, close]).loc[~pd.concat([curr_close, close]).index.duplicated(keep="last")].to_parquet(raw_close_path)
            pd.concat([curr_volume, volume]).loc[~pd.concat([curr_volume, volume]).index.duplicated(keep="last")].to_parquet(raw_volume_path)
        else:
            close.to_parquet(raw_close_path)
            volume.to_parquet(raw_volume_path)
        print(f"retry date save to {raw_close_path}, {raw_volume_path}")
    else:
        print("no new data from retry")


