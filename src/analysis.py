"""Error analysis on the validation fold (run after train.py).

Answers "where is F0.5 being lost, and what do the mistakes look like?":
  1. F0.5 lost per error category (missed matches, empty predictions, false matches, ...)
  2. missed true pairs: never found by blocking vs. found but rejected by the model
  3. false matches: record that belongs to another S1 vs. an unmatched look-alike (decoy)
  4. a printed sample of false positives and false negatives, side by side

Usage:  python analysis.py [n_examples=15]
"""
import pickle
import sys

import polars as pl

from config import SEED, WORK_DIR, norm_path
from evaluate import per_s1_f05, truth_rids
from train import select

N_VAL = 150_000  # must match the n_val used by train.py


def load_validation():
    with open(WORK_DIR / "model.pkl", "rb") as f:
        threshold = pickle.load(f)["thr"]
    pred = pl.read_parquet(WORK_DIR / "val_pred.parquet")
    truth, s1 = truth_rids()
    # Same validation sample as train.py (fold 0, same seed and size).
    val_s1 = s1.filter(pl.col("fold") == 0).sample(N_VAL, seed=SEED)
    val_truth = truth.join(val_s1, on="r1", how="semi")
    return pred, select(pred, threshold), truth, val_truth, val_s1, threshold


def loss_by_category(chosen, val_truth, val_s1):
    x = per_s1_f05(chosen, val_truth, val_s1)
    category = (
        pl.when(pl.col("nt") == 0).then(pl.lit("no true match, but we predicted one"))
        .when(pl.col("np") == 0).then(pl.lit("has true matches, we predicted none"))
        .when(pl.col("np") > pl.col("tp")).then(pl.lit("some predicted matches are wrong"))
        .otherwise(pl.lit("all predictions right, some matches missed"))
    )
    n = x.height
    print(f"macro F0.5 = {x['f'].mean():.4f}  (F0.5 points lost = {1 - x['f'].mean():.4f})")
    print(
        x.filter(pl.col("f") < 1).with_columns(category=category)
        .group_by("category").agg(S1=pl.len(), points_lost=((1 - pl.col("f")).sum() / n).round(4))
        .sort("points_lost", descending=True)
    )


def show_pairs(title, pairs, norm, n):
    print(f"\n--- {title}")
    sample = pairs.sample(min(n, pairs.height), seed=1)
    left = norm.select(r1="rid", name1="name_raw", addr1="addr_raw")
    right = norm.select(r2="rid", name2="name_raw", addr2="addr_raw")
    for r in sample.join(left, on="r1").join(right, on="r2").iter_rows(named=True):
        print(f"p={r['p']:.2f}  {r['name1']} | {r['addr1']}\n        <> {r['name2']} | {r['addr2']}")


def main(n_examples=15):
    pred, chosen, truth, val_truth, val_s1, threshold = load_validation()
    print(f"validation: {val_s1.height:,} S1, threshold {threshold:.2f}\n")
    loss_by_category(chosen, val_truth, val_s1)

    missed = val_truth.join(chosen, on=["r1", "r2"], how="anti")
    in_candidates = missed.join(pred, on=["r1", "r2"], how="semi")
    print(f"\nmissed true pairs: {missed.height:,} of {val_truth.height:,}")
    print(f"  never found by blocking : {missed.height - in_candidates.height:,}")
    print(f"  found, rejected by model: {in_candidates.height:,}")

    wrong = chosen.join(val_truth, on=["r1", "r2"], how="anti")
    owner = wrong.join(truth.rename({"r1": "true_owner"}), on="r2", how="left")
    print(f"\nfalse matches: {wrong.height:,} of {chosen.height:,} predicted")
    print(f"  record belongs to another S1: {owner['true_owner'].is_not_null().sum():,}")
    print(f"  record matches no S1 (decoy): {owner['true_owner'].is_null().sum():,}")

    norm = pl.read_parquet(norm_path("train"), columns=["rid", "name_raw", "addr_raw"])
    show_pairs("false positives (S1 <> wrongly matched record)",
               wrong.join(pred, on=["r1", "r2"]), norm, n_examples)
    show_pairs("false negatives the model rejected (S1 <> true match)",
               in_candidates.join(pred, on=["r1", "r2"]), norm, n_examples)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
