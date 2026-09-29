"""Step 4b - train the matcher and choose the decision threshold.

1. Split the training S1 entities into 5 folds (evaluate.fold_expr). Fold 0 is held out
   for validation; the model is trained on a sample of folds 1-4.
2. Build pair features for their candidates (features.py) and label each pair with the
   ground truth.
3. Fit a gradient-boosted tree classifier: P(pair is a true match). BER_MODEL selects it:
   "hgb" = scikit-learn HistGradientBoosting (iterations 1-4), "lgbm" = LightGBM (faster;
   needs the OpenMP runtime: `brew install libomp` on macOS).
4. On the validation fold, try thresholds and keep the one with the best macro F0.5,
   the project's evaluation metric.

Usage:  python train.py [n_train_S1=400000] [n_val_S1=150000]
Writes: work/model.pkl (model, feature list, threshold), work/val_pred.parquet
"""
import os
import pickle
import sys
import time

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

import embeddings as EMB
import features as F
from config import SEED, WORK_DIR, norm_path
from cluster_features import CLUSTER_COLS, cluster_features
from evaluate import f05, truth_rids
from memguard import start_memory_guard

TOPK_MODEL = 12   # candidates per S1 scored by the matcher (= rerank.KEEP)
MODEL = os.environ.get("BER_MODEL", "lgbm")     # "hgb" reproduces iterations 1-4
CLUSTER = os.environ.get("BER_CLUSTER", "0") == "1"   # second pass with cluster-consistency features
NAMESEARCH = os.environ.get("BER_NAMESEARCH", "0") == "1"  # add name-only candidates for no-address records
RESCUE = float(os.environ.get("BER_RESCUE", "0"))        # >0: S1 left empty get their best candidate if p >= RESCUE
# World the matcher is TRAINED in. "trainlike" = the test-like world (testlike.py): ~19% of the S1 in folds 1-4 removed,
# their S2/S3 records kept as orphan decoys, so training sees the test decoy density (~2.3 per S1 instead of ~1.2).
# Validation always uses the ordinary world here; validate.py scores the test-like one.
TRAIN_WORLD = os.environ.get("BER_TRAIN_WORLD", "train")
S1_PER_CHUNK = 250_000

# For these features, also give the gap to the best candidate of the same S1:
# "is this the most similar candidate this S1 has?" is very informative.
RELATIVE_COLS = ["n_tset", "n_ratio", "cc_ratio", "a_tset", "nm_wi", "ad_wi", "num_inter"]


def add_relative(x):
    return x.with_columns(
        (pl.col(c) - pl.col(c).max().over("r1")).alias(f"{c}_gap1") for c in RELATIVE_COLS)


def featurize(pairs, norm, idf_core, idf_addr):
    """Full feature frame, built chunk by chunk (each chunk holds complete S1 groups)."""
    r1s = pairs["r1"].unique().sort()
    parts = []
    for i in range(0, len(r1s), S1_PER_CHUNK):
        chunk = pairs.join(pl.DataFrame({"r1": r1s[i:i + S1_PER_CHUNK]}), on="r1", how="semi")
        parts.append(add_relative(F.build(chunk, norm, idf_core, idf_addr)))
    return pl.concat(parts)


def select(pred, threshold, rescue=None):
    """Final decision rule: keep pairs with p >= threshold, and give each S2/S3 record to
    at most one S1 (the most probable one) - in the data no record belongs to two S1s.

    rescue (default BER_RESCUE): an S1 left with no match (which scores 0 unless it truly has none)
    gets its best still-unassigned candidate if that candidate's p >= rescue.
    """
    rescue = RESCUE if rescue is None else rescue
    chosen = (
        pred.filter(pl.col("p") >= threshold)
        .sort(["p", "r1"], descending=[True, False])
        .unique("r2", keep="first", maintain_order=True)
        .select("r1", "r2")
    )
    if rescue > 0:
        extra = (pred.filter(pl.col("p") >= rescue)
                 .join(chosen.select("r1"), on="r1", how="anti")        # S1 still empty
                 .join(chosen.select("r2"), on="r2", how="anti")        # record still free
                 .sort(["p", "r2"], descending=[True, False])
                 .unique("r1", keep="first", maintain_order=True)
                 .sort(["p", "r1"], descending=[True, False])
                 .unique("r2", keep="first", maintain_order=True)
                 .select("r1", "r2"))
        chosen = pl.concat([chosen, extra])
    return chosen


def fit_model(train_frame, feat_cols):
    """Fit the matcher chosen by BER_MODEL on a labelled feature frame."""
    X, y = train_frame.select(feat_cols).to_numpy(), train_frame["y"].to_numpy()
    if MODEL == "lgbm":
        import lightgbm as lgb
        # early-stopping set: 5% of the training S1 records (whole records, so no S1 is on both sides)
        held = (train_frame["r1"].hash(seed=11) % 20 == 0).to_numpy()
        model = lgb.LGBMClassifier(
            n_estimators=3000, learning_rate=0.05, num_leaves=127, min_child_samples=100,
            reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=SEED, deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
        model.fit(X[~held], y[~held], eval_set=[(X[held], y[held])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        return model
    model = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.08, max_leaf_nodes=127, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.05,
        n_iter_no_change=30, random_state=SEED,
    )
    return model.fit(X, y)


def n_trees(model):
    return getattr(model, "best_iteration_", None) or getattr(model, "n_iter_", "?")


def predict_proba(bundle, x, norm, world="train"):
    """P(match) for each row of a feature frame, with one or two passes (cluster features)."""
    if "stage1" in bundle:
        p1 = bundle["stage1"].predict_proba(x.select(bundle["feat1"]).to_numpy())[:, 1]
        x = x.hstack(cluster_features(x.select("r1", "r2").with_columns(p1=p1), norm, world=world))
    return bundle["model"].predict_proba(x.select(bundle["feat_cols"]).to_numpy())[:, 1]


def fit_two_pass(train_frame, feat_cols, norm, first_model, world="train"):
    """Second pass: out-of-fold first-pass scores -> cluster features -> second model.

    Training S1 records get p1 from a model that never saw them (2-fold split by S1), so the
    second model learns from the same kind of scores it will see on validation and test.
    """
    half = (train_frame["r1"].hash(seed=3) % 2 == 0).to_numpy()
    p1 = np.zeros(train_frame.height)
    for h in (True, False):
        m = fit_model(train_frame.filter(pl.Series(half == h)), feat_cols)
        p1[half != h] = m.predict_proba(train_frame.filter(pl.Series(half != h)).select(feat_cols).to_numpy())[:, 1]
    frame2 = train_frame.hstack(cluster_features(train_frame.select("r1", "r2").with_columns(p1=p1), norm, world=world))
    feat2 = feat_cols + CLUSTER_COLS
    return {"model": fit_model(frame2, feat2), "feat_cols": feat2, "stage1": first_model, "feat1": feat_cols}


def load_candidates(split):
    cand = pl.read_parquet(WORK_DIR / "cand" / f"{split}.parquet").filter(pl.col("brank") < TOPK_MODEL)
    cand = cand.with_columns(name_sim=pl.lit(0.0, pl.Float32), from_namesearch=pl.lit(0.0, pl.Float32))
    path = WORK_DIR / "cand_name" / f"{split}.parquet"
    if NAMESEARCH and path.exists():
        # name-only candidates for no-address records (namesearch.py) that blocking did not find
        name = pl.read_parquet(path)
        cand = cand.join(name.select("r1", "r2", ns=pl.col("name_sim")), on=["r1", "r2"], how="left") \
                   .with_columns(name_sim=pl.col("ns").fill_null(0.0).cast(pl.Float32)).drop("ns")
        extra = name.join(cand.select("r1", "r2"), on=["r1", "r2"], how="anti").with_columns(
            from_namesearch=pl.lit(1.0, pl.Float32), brank=pl.lit(TOPK_MODEL, pl.UInt32),
            name_sim=pl.col("name_sim").cast(pl.Float32))
        cand = pl.concat([cand, extra], how="diagonal_relaxed").fill_null(0)
    if EMB.ENABLED:
        # name-search extras have no embedding features yet: compute them for every final candidate
        cand = cand.drop([c for c in ("cos", "erank", "rev_rank", "rev_gap_cos") if c in cand.columns])
        cand = cand.join(EMB.pair_features(cand.select("r1", "r2"), split), on=["r1", "r2"], how="left")
        cand = EMB.within_s1(cand.with_columns(pl.col("in_emb", "in_raw").fill_null(0.0)
                                               if "in_emb" in cand.columns else []))
    return F.add_block_context(cand)


def main(n_train=400_000, n_val=150_000):
    t0 = time.time()
    truth, s1 = truth_rids()
    val_s1 = s1.filter(pl.col("fold") == 0).sample(n_val, seed=SEED)
    pool = s1.filter(pl.col("fold") != 0)
    if TRAIN_WORLD != "train":   # only S1 that still exist in that world (removed ones have src 0)
        kept = pl.read_parquet(norm_path(TRAIN_WORLD), columns=["rid", "src"]).filter(pl.col("src") == 1).select(r1="rid")
        pool = pool.join(kept, on="r1", how="semi")
    train_s1 = pool.sample(n_train, seed=SEED)
    print(f"training world: {TRAIN_WORLD} ({pool.height:,} eligible S1 in folds 1-4)", flush=True)

    # --- features + labels for the training sample (TRAIN_WORLD) and the validation sample (ordinary world)
    frames, norms = {}, {}
    for name, sample, world in (("train", train_s1, TRAIN_WORLD), ("val", val_s1, "train")):
        cand = load_candidates(world).join(sample.select("r1"), on="r1", how="semi")
        norm = F.load_norm(world)
        norms[name] = norm
        idf_core, idf_addr = F.idf_table(norm, "core"), F.idf_table(norm, "addr")
        x = featurize(cand, norm, idf_core, idf_addr)
        del cand
        positives = truth.join(sample, on="r1", how="semi").select("r1", "r2", y=pl.lit(True))
        x = x.join(positives, on=["r1", "r2"], how="left").with_columns(pl.col("y").fill_null(False))
        frames[name] = x.sort("r1", "r2")  # fixed row order -> reproducible model
        print(f"{name}: {x.height:,} pairs, {x['y'].mean():.1%} true matches "
              f"({time.time() - t0:.0f}s)", flush=True)

    # --- fit
    feat_cols = [c for c in frames["train"].columns if c not in ("r1", "r2", "y")]
    model = fit_model(frames["train"], feat_cols)
    bundle = {"model": model, "feat_cols": feat_cols}
    print(f"model fitted ({MODEL}): {n_trees(model)} trees, {len(feat_cols)} features "
          f"({time.time() - t0:.0f}s)", flush=True)
    if CLUSTER:
        bundle = fit_two_pass(frames["train"], feat_cols, norms["train"], model, world=TRAIN_WORLD)
        print(f"second pass fitted: {n_trees(bundle['model'])} trees, {len(bundle['feat_cols'])} features "
              f"({time.time() - t0:.0f}s)", flush=True)

    # --- choose the threshold on the validation fold
    val = frames["val"]
    pred = val.select("r1", "r2", "y").with_columns(p=predict_proba(bundle, val, norms["val"]))
    val_truth = truth.join(val_s1, on="r1", how="semi")
    best_score, best_thr = 0.0, 0.5
    for thr in np.arange(0.20, 0.96, 0.05):
        score = f05(select(pred, thr), val_truth, val_s1)
        print(f"  threshold {thr:.2f}: macro F0.5 {score:.4f}")
        if score > best_score:
            best_score, best_thr = score, float(thr)
    ceiling = f05(pred.filter("y").select("r1", "r2"), val_truth, val_s1)
    print(f"VALIDATION macro F0.5 = {best_score:.4f} at threshold {best_thr:.2f} "
          f"(ceiling with these candidates: {ceiling:.4f})")

    with open(WORK_DIR / "model.pkl", "wb") as f:
        pickle.dump({**bundle, "thr": best_thr}, f)
    pred.write_parquet(WORK_DIR / "val_pred.parquet")


if __name__ == "__main__":
    start_memory_guard("train")
    main(*[int(a) for a in sys.argv[1:]])
