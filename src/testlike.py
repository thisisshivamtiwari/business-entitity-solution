"""Build a "test-like" training world: same records, but as many decoys per S1 as the test set.

Why: the EDA found ~2.3 decoy S2/S3 records per S1 in test but only ~1.2 in train, so the
ordinary validation understates how often a look-alike record can be wrongly matched.

How: we remove a random share of Source-1 records from the *query side* in each country, but keep
all their S2/S3 records. Those records now match no S1, i.e. they become decoys. Removed S1 are
drawn only from folds 1-4, so the validation fold (fold 0) keeps all of its clusters intact.

The world is written as work/norm/trainlike.parquet: identical to norm/train.parquet except that
removed S1 records get src = 0, so blocking ignores them (rid still equals the row number).
Candidates for it are then built with the normal pipeline steps:

    python testlike.py                     # build the world (seconds)
    python block.py trainlike              # candidate search in that world
    python rerank.py apply trainlike       # learned ranker (uses the trained ranker)
    python validate.py <tag> trainlike     # score the trained matcher on the validation fold
"""
import json

import polars as pl

from config import DATA_DIR, SEED, WORK_DIR, norm_path
from evaluate import fold_expr

MEAN_CLUSTER = 3.46  # mean S2/S3 matches per S1 in train (identical for US and India)


def decoys_per_s1(split):
    """(S2 + S3) / S1 - mean cluster size, per country, from the raw source files."""
    counts = {}
    for n in (1, 2, 3):
        df = pl.read_csv(DATA_DIR / split / f"{split}_source{n}.tsv", separator="\t", quote_char=None,
                         infer_schema=False, columns=["country"])
        for c, k in df["country"].value_counts().rows():
            counts.setdefault(c, [0, 0])[0 if n == 1 else 1] += k
    return {c: s23 / s1 - MEAN_CLUSTER for c, (s1, s23) in counts.items()}


def main():
    target = decoys_per_s1("test")
    norm = pl.read_parquet(norm_path("train"))
    s1 = norm.filter(pl.col("src") == 1).select("rid", "country", fold=fold_expr())
    plan, drop = [], []
    for country, grp in s1.group_by("country"):
        country = country[0]
        n_s1 = grp.height
        n_s23 = norm.filter((pl.col("country") == country) & (pl.col("src") > 1)).height
        keep = round(n_s23 / (MEAN_CLUSTER + target[country]))
        n_drop = n_s1 - keep
        pool = grp.filter(pl.col("fold") != 0).sort("rid")
        assert 0 < n_drop < pool.height, (country, n_drop)
        drop.append(pool.sample(n_drop, seed=SEED).select("rid"))
        plan.append({"country": country, "S1": n_s1, "S1_removed": n_drop, "S1_kept": keep,
                     "decoys_per_S1_before": round(n_s23 / n_s1 - MEAN_CLUSTER, 2),
                     "decoys_per_S1_after": round(n_s23 / keep - MEAN_CLUSTER, 2),
                     "test_target": round(target[country], 2)})
    drop = pl.concat(drop)
    # Every iteration must be measured in the SAME world: reuse the saved list of removed S1.
    saved = WORK_DIR / "results" / "testlike_removed_s1.parquet"
    if saved.exists():
        drop = pl.read_parquet(saved)
        print(f"reusing the saved set of {drop.height:,} removed S1 ({saved})")
    else:
        saved.parent.mkdir(parents=True, exist_ok=True)
        drop.write_parquet(saved)
    drop = drop.with_columns(removed=pl.lit(True))
    world = (norm.join(drop, on="rid", how="left")
             .with_columns(src=pl.when(pl.col("removed")).then(pl.lit(0, pl.UInt8)).otherwise(pl.col("src")))
             .drop("removed").sort("rid"))
    assert world["rid"].to_list() == list(range(world.height))
    world.write_parquet(norm_path("trainlike"))
    (WORK_DIR / "results").mkdir(exist_ok=True)
    (WORK_DIR / "results" / "testlike_world.json").write_text(json.dumps(plan, indent=1))
    for p in plan:
        print(p)
    print(f"test-like world -> {norm_path('trainlike')}")


if __name__ == "__main__":
    main()
