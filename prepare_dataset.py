"""
Downloads the PolySpeech-100 Parquet dataset from Hugging Face 
and restores the original audio (.wav) and text (.txt) files.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download

def restore_parquet(parquet_file, restore_dir):
    lang_name = Path(parquet_file).stem  # e.g., "lang=eng_Latn"
    lang_output_dir = os.path.join(restore_dir, lang_name)
    
    print(f"Restoring: {lang_name} ...")
    df = pd.read_parquet(parquet_file, engine='pyarrow')
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=lang_name):
        rel_path = row['relative_path']
        content = row['file_content']
        
        full_output_path = os.path.join(lang_output_dir, rel_path)
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        
        with open(full_output_path, "wb") as f:
            f.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and restore PolySpeech-100 dataset.")
    parser.add_argument("--repo_id", type=str, default="youngseng/PolySpeech-100-v1",
                        help="Hugging Face dataset repository ID")
    parser.add_argument("--output_dir", type=str, default="./Restored-PolySpeech",
                        help="Directory to save the restored files")
    parser.add_argument("--cache_dir", type=str, default="./hf_cache",
                        help="Temporary directory for downloaded parquet files")
    
    # 【新增】控制特定语言下载的超参数
    parser.add_argument("--lang", type=str, default="all",
                        help="Specify a language code to download (e.g., 'eng_Latn'). Use 'all' for the full dataset.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # 1. 配置下载模式
    if args.lang.lower() == "all":
        download_pattern = "*.parquet"
        print(f"Downloading the ENTIRE dataset from Hugging Face: {args.repo_id}...")
    else:
        # 只匹配特定语言的 parquet 文件，例如 "lang=eng_Latn.parquet"
        download_pattern = f"lang={args.lang}.parquet"
        print(f"Downloading specific language ({args.lang}) from: {args.repo_id}...")

    # 2. 从 Hugging Face 下载
    # 使用 allow_patterns 可以确保只下载我们需要的文件，节省大量带宽
    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.cache_dir,
        allow_patterns=[download_pattern]
    )

    # 3. 查找并还原文件
    if args.lang.lower() == "all":
        parquet_files = list(Path(local_dir).glob("*.parquet"))
    else:
        parquet_files = list(Path(local_dir).glob(f"lang={args.lang}.parquet"))

    if not parquet_files:
        print(f"No .parquet files found matching '{download_pattern}'. Please check the language code.")
    else:
        print(f"Found {len(parquet_files)} Parquet file(s). Starting restoration...")
        for pf in parquet_files:
            restore_parquet(pf, args.output_dir)
            
        print(f"\nAll files successfully restored to: {args.output_dir}")