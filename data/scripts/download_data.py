#!/usr/bin/env python
"""
Data Download Script

Downloads and preprocesses DDI datasets from TDC (Therapeutics Data Commons).
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def download_drugbank(save_dir: str = 'data'):
    """Download DrugBank DDI dataset."""
    from tdc.multi_pred import DDI

    print("Downloading DrugBank DDI dataset...")
    data = DDI(name='DrugBank')

    # Get data and splits
    df = data.get_data()
    splits = data.get_split()

    # Create directories
    raw_dir = Path(save_dir) / 'raw'
    processed_dir = Path(save_dir) / 'processed'
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data
    df.to_csv(raw_dir / 'drugbank_full.csv', index=False)
    print(f"Saved full dataset: {len(df)} pairs")

    # Save splits
    for split_name, split_df in splits.items():
        split_df.to_csv(processed_dir / f'drugbank_{split_name}.csv', index=False)
        print(f"Saved {split_name} split: {len(split_df)} pairs")

    # Save label names
    if 'Y_Name' in df.columns:
        label_names = df[['Y', 'Y_Name']].drop_duplicates()
        label_dict = dict(zip(label_names['Y'].astype(str), label_names['Y_Name']))
        with open(processed_dir / 'label_names.json', 'w') as f:
            json.dump(label_dict, f, indent=2)
        print(f"Saved label names: {len(label_dict)} interaction types")

    print("DrugBank download complete!")
    return df


def download_twosides(save_dir: str = 'data'):
    """Download TWOSIDES dataset."""
    from tdc.multi_pred import DDI

    print("Downloading TWOSIDES dataset...")
    data = DDI(name='TWOSIDES')

    df = data.get_data()
    splits = data.get_split()

    # Create directories
    raw_dir = Path(save_dir) / 'raw'
    processed_dir = Path(save_dir) / 'processed'
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    df.to_csv(raw_dir / 'twosides_full.csv', index=False)
    print(f"Saved full dataset: {len(df)} pairs")

    for split_name, split_df in splits.items():
        split_df.to_csv(processed_dir / f'twosides_{split_name}.csv', index=False)
        print(f"Saved {split_name} split: {len(split_df)} pairs")

    print("TWOSIDES download complete!")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download DDI datasets")
    parser.add_argument(
        '--dataset',
        type=str,
        default='drugbank',
        choices=['drugbank', 'twosides', 'all'],
        help='Dataset to download',
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='data',
        help='Directory to save data',
    )

    args = parser.parse_args()

    print("DDI Dataset Downloader")
    print("=" * 50)

    if args.dataset == 'drugbank' or args.dataset == 'all':
        download_drugbank(args.save_dir)

    if args.dataset == 'twosides' or args.dataset == 'all':
        download_twosides(args.save_dir)

    print("\nDownload complete!")


if __name__ == '__main__':
    main()
