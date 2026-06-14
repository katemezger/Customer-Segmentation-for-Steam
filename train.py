"""
Training entry point. Reads steam-200k.csv, fits the full pipeline, and serializes it.

Usage:
    python train.py
    python train.py --data path/to/steam-200k.csv --out custom_pipeline.pkl
"""

import argparse
import sys
import pandas as pd
from pipeline import SteamSegmentationPipeline


def main():
    parser = argparse.ArgumentParser(description="Train the Steam segmentation pipeline.")
    parser.add_argument("--data", default="steam-200k.csv", help="Path to raw CSV (default: steam-200k.csv)")
    parser.add_argument("--out", default=None, help="Output path for serialized pipeline (default: steam_pipeline.pkl)")
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    try:
        # no header row in the raw dataset — pipeline assigns column names internally
        df = pd.read_csv(args.data, header=None)
    except FileNotFoundError:
        print(f"Error: '{args.data}' not found. Download it from Kaggle and place it in the project root.")
        sys.exit(1)

    print(f"Loaded {len(df):,} rows. Fitting pipeline ...")
    pipe = SteamSegmentationPipeline()
    pipe.fit(df)

    # pass path only if explicitly provided; otherwise save_pipeline uses its default
    save_kwargs = {"path": args.out} if args.out else {}
    pipe.save_pipeline(**save_kwargs)


if __name__ == "__main__":
    main()
