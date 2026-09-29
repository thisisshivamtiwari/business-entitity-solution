"""Matching stage 1: turn candidate lists into labelled pairs.

For every S1 entity in a candidate file, each candidate becomes one row
(s1_id, cand_id, rank, label) with label = 1 when the ground truth lists it.
Training entities are split into a 'ce' group (cross-encoder training) and a
'gbm' group (scored by the cross-encoder, then used to train the tree model),
so the tree never sees over-confident scores on the encoder's own training data.

Outputs: {name}_pairs.parquet
"""
import argparse
import sys
import zlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blocking"))
from common import read_tsv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--ground-truth", default=None, help="omit for the test set (no label column)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-split", type=float, default=0.0,
                    help="if >0, assign this share of S1 entities to group 'gbm', the rest to 'ce'")
    args = ap.parse_args()

    cands = read_tsv(args.candidates)
    truth = None
    if args.ground_truth:
        gt = read_tsv(args.ground_truth)
        truth = {s1: set(x for x in l.split(",") if x) for s1, l in zip(gt.source1_entity_id, gt.matched_entity_ids)}

    rows = []
    for s1, lst in zip(cands.source1_entity_id, cands.candidate_entity_ids):
        t = truth.get(s1, set()) if truth is not None else None
        for rank, c in enumerate(x for x in lst.split(",") if x):
            rows.append((s1, c, rank, int(c in t)) if t is not None else (s1, c, rank))
    df = pd.DataFrame(rows, columns=["s1_id", "cand_id", "rank"] + (["label"] if truth is not None else []))
    if args.group_split > 0:
        ent = df.s1_id.unique()
        h = {e: zlib.crc32(f"grp:{e}".encode()) / 2**32 for e in ent}
        df["group"] = df.s1_id.map(lambda e: "gbm" if h[e] < args.group_split else "ce")
    df.to_parquet(args.out, index=False)
    n_ent = df.s1_id.nunique()
    print(f"[pairs] {len(df):,} pairs from {n_ent:,} S1 entities" + (f", positives {df.label.mean():.3%}" if "label" in df else "")
          + (f", groups {df.groupby('group').s1_id.nunique().to_dict()}" if args.group_split > 0 else "")
          + f" -> {args.out}")


if __name__ == "__main__":
    main()
