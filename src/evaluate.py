"""Local evaluation: the evaluation metric (macro F0.5) and blocking recall.

Metric: F0.5 is computed per S1 entity and averaged over
ALL S1 entities (macro average). An S1 with no true matches scores 1 if we predict
nothing and 0 otherwise.

Validation split: S1 entities with hash(entity_id) % 5 == 0 form the held-out fold.
Their candidates still come from the full training pool, so the amount of look-alike
"decoy" records is the same as in the real task.

Usage:  python evaluate.py      # blocking recall report for work/cand/train.parquet
"""
import polars as pl

from config import WORK_DIR, gt_pq_path, norm_path

NFOLD = 5


def fold_expr(col="entity_id"):
    return (pl.col(col).hash(seed=7) % NFOLD).cast(pl.UInt8)


def truth_rids():
    """Return (truth, s1):
        truth  true pairs as (r1, r2) rids
        s1     every training S1 as (r1, fold)
    """
    norm = pl.read_parquet(norm_path("train"), columns=["rid", "entity_id", "src"])
    ids = norm.select("entity_id", "rid")
    s1 = norm.filter(pl.col("src") == 1).select(r1="rid", fold=fold_expr())
    truth = (
        pl.read_parquet(gt_pq_path())
        .join(ids.rename({"entity_id": "s1", "rid": "r1"}), on="s1")
        .join(ids.rename({"entity_id": "cand", "rid": "r2"}), on="cand")
        .select("r1", "r2")
    )
    return truth, s1


def per_s1_f05(pred, truth, s1):
    """Per-S1 counts and F0.5. pred / truth: (r1, r2) pairs; s1: the S1 set being scored."""
    tp = pred.join(truth, on=["r1", "r2"], how="semi").group_by("r1").len("tp")
    n_pred = pred.group_by("r1").len("np")
    n_true = truth.group_by("r1").len("nt")
    x = (
        s1.select("r1")
        .join(tp, on="r1", how="left").join(n_pred, on="r1", how="left")
        .join(n_true, on="r1", how="left").fill_null(0)
    )
    precision = pl.col("tp") / pl.col("np")
    recall = pl.col("tp") / pl.col("nt")
    f = (
        pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))  # singleton
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise(1.25 * precision * recall / (0.25 * precision + recall))
    )
    return x.with_columns(f=f)


def f05(pred, truth, s1):
    """Macro F0.5 over every S1 in `s1` (the evaluation metric)."""
    return per_s1_f05(pred, truth, s1)["f"].mean()


def blocking_report(cand, truth, s1):
    """Share of true pairs present among the top-k candidates, for several k."""
    n_s1 = s1.height
    n_true = truth.join(s1, on="r1", how="semi").height
    cand = cand.join(s1, on="r1", how="semi")
    print(f"  {n_s1:,} S1, {n_true:,} true pairs, {cand.height:,} candidates "
          f"({cand.height / n_s1:.1f} per S1)")
    for k in (1, 3, 5, 8, 10, 12, 15, 20, 30):
        top = cand.filter(pl.col("brank") < k)
        hits = top.join(truth, on=["r1", "r2"], how="semi").height
        print(f"  top-{k:<2}: recall {hits / n_true:.4f}, {top.height / n_s1:.2f} candidates per S1")


if __name__ == "__main__":
    truth, s1 = truth_rids()
    print("Blocking recall on all training S1:")
    blocking_report(pl.read_parquet(WORK_DIR / "cand" / "train.parquet"), truth, s1)
