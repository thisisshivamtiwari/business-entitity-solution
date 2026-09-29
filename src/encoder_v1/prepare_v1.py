"""Encoder v1 data step: normalised records of all three sources (row order = source 1, 2, 3) and the
ground-truth links. 5% of the S1 (CRC32 hash, seed 42) are left out of encoder training."""
import argparse
import zlib
from pathlib import Path

import pandas as pd

from common_v1 import load_records, read_tsv


def fold_of(s1_id, val_frac, seed):
    h = zlib.crc32(f"{seed}:{s1_id}".encode()) / 2**32
    return "val" if h < val_frac else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        out = work / f"{split}_records.parquet"
        if out.exists():
            print(f"[prepare] {out} exists, skipping")
            continue
        df = load_records(args.data_dir, split)
        df.to_parquet(out, index=False)
        print(f"[prepare] {split}: {len(df):,} records, {df.addr_missing.mean():.2%} missing address -> {out}")

    if "train" in args.splits:
        gt = read_tsv(Path(args.data_dir) / "train" / "train_ground_truth.tsv")
        gt["fold"] = [fold_of(s, args.val_frac, args.seed) for s in gt.source1_entity_id]
        gt.loc[gt.fold == "val", "source1_entity_id"].to_csv(work / "val_s1_ids.txt", index=False, header=False)
        pairs = gt.assign(match_id=gt.matched_entity_ids.str.split(",")).explode("match_id")
        pairs = pairs[pairs.match_id.str.len() > 0]
        pairs = pairs.rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "match_id", "fold"]]
        pairs.to_parquet(work / "train_pairs.parquet", index=False)
        print(f"[prepare] {len(pairs):,} links; {(gt.fold=='val').sum():,} S1 entities held out")


if __name__ == "__main__":
    main()
