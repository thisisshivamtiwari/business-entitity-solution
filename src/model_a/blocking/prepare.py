"""Stage 1: clean the raw TSVs, build cluster labels, mark the shared validation fold.

Outputs (in --work-dir):
  {split}_records.parquet   cleaned records, all 3 sources, row order used by every later stage
  {split}_ids.parquet       entity_id only, same row order (deliverable next to the .npy files)
  train_pairs.parquet       (s1_id, match_id, fold)  one row per ground-truth link
  val_s1_ids.txt            fold-0 S1 entities: polars hash(seed=7) % 5 == 0  (expected 442,001)
"""
import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from common import load_records, read_tsv  # model_a/blocking/common.py

EXPECTED_FOLD0 = 442_001


def fold_column(ids):
    """Team-wide validation fold. Depends on the polars hash, hence the pinned version."""
    return (pl.DataFrame({"entity_id": list(ids)})
              .select((pl.col("entity_id").hash(seed=7) % 5).alias("fold"))["fold"].to_list())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--drop-s1", default=None, help="file of S1 ids removed from the training pool (test-like world): "
                    "their links are not written, so no model trains on them and their S2/S3 copies act as decoys")
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
        df[["entity_id"]].to_parquet(work / f"{split}_ids.parquet", index=False)
        print(f"[prepare] {split}: {len(df):,} records, {df.addr_missing.mean():.2%} missing address, "
              f"{df.indic_name.mean():.2%} Indic-script names -> {out}")

    if "train" in args.splits and not (work / "train_pairs.parquet").exists():
        gt = read_tsv(Path(args.data_dir) / "train" / "train_ground_truth.tsv")
        gt["fold"] = fold_column(gt.source1_entity_id)
        n0 = int((gt.fold == 0).sum())
        flag = "" if n0 == EXPECTED_FOLD0 else f"  (WARNING: expected {EXPECTED_FOLD0:,}; polars version differs?)"
        print(f"[prepare] fold-0 validation S1 entities: {n0:,}{flag}")
        gt.loc[gt.fold == 0, "source1_entity_id"].to_csv(work / "val_s1_ids.txt", index=False, header=False)
        pairs = gt.assign(match_id=gt.matched_entity_ids.str.split(",")).explode("match_id")
        pairs = pairs[pairs.match_id.str.len() > 0]
        pairs = pairs.rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "match_id", "fold"]]
        if args.drop_s1:
            rem = set(Path(args.drop_s1).read_text().split())
            hit0 = int(gt.loc[gt.fold == 0, "source1_entity_id"].isin(rem).sum())
            assert hit0 == 0, f"{hit0} removed S1 are in fold 0; the validation fold must stay intact"
            n = len(pairs); pairs = pairs[~pairs.s1_id.isin(rem)]
            print(f"[prepare] test-like world: {len(rem):,} S1 removed from the pool, {n - len(pairs):,} links dropped, fold 0 untouched")
        pairs.to_parquet(work / "train_pairs.parquet", index=False)
        print(f"[prepare] {len(pairs):,} links, {(pairs.fold != 0).sum():,} usable for training")


if __name__ == "__main__":
    main()
