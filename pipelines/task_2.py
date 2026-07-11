import pandas as pd
import os
import argparse

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="clean and merge raw data")
    parse.add_argument("--date", type=str, required=True)
    parse.add_argument("--batch", type=str, required=True)
    parse.add_argument("--missing_threshold", type=float, default=0.3)
    args = parse.parse_args()

    tmp_dir = os.path.join(os.getcwd(), "data/processed")

    # 定义一个处理函数，避免 close/volume 重复写两遍
    def process(raw_name, final_name):
        raw_path = os.path.join(tmp_dir, raw_name)
        final_path = os.path.join(tmp_dir, final_name)

        if not os.path.exists(raw_path):
            print(f"[{args.batch}] {raw_name} 无新数据，跳过")
            return

        new_data = pd.read_parquet(raw_path)
        missing_pct = new_data.isnull().mean()
        usable_cols = missing_pct[missing_pct < args.missing_threshold].index.tolist()
        dropped_cols = missing_pct[missing_pct >= args.missing_threshold].index.tolist()

        new_data = new_data[usable_cols]
        new_data = new_data.ffill()

        if os.path.exists(final_path):
            existing = pd.read_parquet(final_path)
            combined = pd.concat([existing, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = new_data

        combined.to_parquet(final_path)
        os.remove(raw_path)
        print(f"[{args.batch}] {final_name} 合并完成")

    # 分别处理 close 和 volume
    process("close.parquet", "processed_close.parquet")
    process("volume.parquet", "processed_volume.parquet")