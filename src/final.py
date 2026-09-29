"""Final training: fit the ranker and the matcher on ALL training data.

Experiments (train.py) fit on a 400k-S1 sample of folds 1-4 and keep fold 0 for validation.
For the final run every training S1 record is used (~2.2M S1, ~26M candidate pairs), with the
settings chosen by the experiments (see EXPERIMENTS.md):

    python final.py ranker            # candidate ranker on S1 from all folds
    python final.py matcher <tag>     # matcher on all S1; threshold = best test-like threshold of <tag>

Memory: features are built in S1 chunks and packed straight into one float32 matrix (~10 GB)
instead of keeping ~25 GB of intermediate tables; cluster features are computed chunk by chunk.
A 2% slice of S1 records is set aside only to decide when LightGBM stops adding trees.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import polars as pl

import features as F
import rerank as R
from boosting import BoosterModel
from cluster_features import CLUSTER_COLS, cluster_features
from config import SEED, WORK_DIR
from evaluate import truth_rids
from memguard import start_memory_guard
from train import CLUSTER, TRAIN_WORLD, featurize, load_candidates

S1_CHUNK = 200_000
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------------------
# Candidate ranker on every fold
# ---------------------------------------------------------------------------------------

def fit_ranker(n_s1=600_000):
    from sklearn.ensemble import HistGradientBoostingClassifier
    truth, s1 = truth_rids()
    sample = s1.sample(n_s1, seed=SEED)                       # all folds
    cand = R.load_pool("train", sample)
    x = (R.add_within_s1(cand)
         .join(truth.select("r1", "r2", y=pl.lit(True)), on=["r1", "r2"], how="left")
         .with_columns(pl.col("y").fill_null(False)).sort("r1", "r2"))
    del cand
    cols = R.feature_columns(x)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                           min_samples_leaf=200, random_state=SEED)
    model.fit(R.to_matrix(x, cols), x["y"].to_numpy())
    with open(R.MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "cols": cols}, f)
    log(f"ranker fitted on {n_s1:,} S1 from all folds ({x.height:,} pairs) -> {R.MODEL_PATH}")


# ---------------------------------------------------------------------------------------
# Matcher on every training S1
# ---------------------------------------------------------------------------------------

# Same settings as train.fit_model's LGBMClassifier, in LightGBM's native parameter names.
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100,
              "lambda_l2": 1.0, "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8,
              "seed": SEED, "deterministic": True, "force_row_wise": True, "verbose": -1}


def fit_subset(dataset, fit_rows, stop_rows):
    """Train on dataset rows `fit_rows`, stop early on `stop_rows`. Subsets share the parent's
    binned data, so the ~10 GB feature matrix is never copied."""
    import lightgbm as lgb
    booster = lgb.train(LGB_PARAMS, dataset.subset(fit_rows), num_boost_round=3000,
                        valid_sets=[dataset.subset(stop_rows)],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    return BoosterModel(booster)


def build_matrix(cand, norm, truth):
    """Features for every candidate pair, packed into one float32 matrix (S1 chunk by chunk)."""
    idf_core, idf_addr = F.idf_table(norm, "core"), F.idf_table(norm, "addr")
    r1s = cand["r1"].unique().sort()
    labels = truth.select("r1", "r2", y=pl.lit(True))
    X = cols = None
    keys, ys, row = [], [], 0
    for i in range(0, len(r1s), S1_CHUNK):
        part = cand.join(pl.DataFrame({"r1": r1s[i:i + S1_CHUNK]}), on="r1", how="semi")
        x = featurize(part, norm, idf_core, idf_addr).join(labels, on=["r1", "r2"], how="left") \
            .with_columns(pl.col("y").fill_null(False)).sort("r1", "r2")
        if X is None:
            cols = [c for c in x.columns if c not in ("r1", "r2", "y")]
            X = np.empty((cand.height, len(cols)), dtype=np.float32)
        X[row:row + x.height] = x.select(cols).to_numpy().astype(np.float32)
        keys.append(x.select("r1", "r2"))
        ys.append(x["y"].to_numpy())
        row += x.height
        log(f"features: {min(i + S1_CHUNK, len(r1s)):,}/{len(r1s):,} S1, {row:,} pairs")
    return X[:row], np.concatenate(ys), pl.concat(keys), cols


def fit_matcher(tag):
    thr = json.loads((WORK_DIR / "results" / f"{tag}__trainlike.json").read_text())["best_threshold"]
    truth, _ = truth_rids()
    # every training S1 of TRAIN_WORLD (BER_TRAIN_WORLD): "trainlike" drops the fixed set of removed S1, whose
    # S2/S3 records stay in as orphan decoys (the test condition); top-12 candidates
    cand = load_candidates(TRAIN_WORLD)
    norm = F.load_norm(TRAIN_WORLD)
    log(f"training world: {TRAIN_WORLD}, {cand['r1'].n_unique():,} S1")
    X, y, keys, cols = build_matrix(cand, norm, truth)
    del cand
    import lightgbm as lgb
    r1 = keys["r1"]
    stop = (r1.hash(seed=11) % 50 == 0).to_numpy()            # 2% of S1: early stopping only
    log(f"matrix {X.shape[0]:,} x {X.shape[1]} ({X.nbytes / 2**30:.1f} GB), {y.mean():.1%} true")

    # One binned dataset for every fit. free_raw_data=True: LightGBM keeps only its binned copy
    # (~2.5 GB); our own X stays alive just long enough to compute the first-pass scores.
    data = lgb.Dataset(X, label=y, free_raw_data=True, params={"verbose": -1}).construct()
    first = fit_subset(data, np.where(~stop)[0], np.where(stop)[0])
    log(f"first pass: {first.best_iteration_} trees")
    bundle = {"model": first, "feat_cols": cols, "thr": thr}
    if CLUSTER:
        # out-of-fold first-pass scores: each half of the S1 records is scored by a model trained
        # on the other half, so the second pass learns from scores like those it will see on test
        half = (r1.hash(seed=3) % 2 == 0).to_numpy()
        p1 = np.zeros(len(y))
        for h in (True, False):
            m = fit_subset(data, np.where((half == h) & ~stop)[0], np.where((half == h) & stop)[0])
            p_all = m.predict_proba(X)[:, 1]
            p1[half != h] = p_all[half != h]
            log(f"out-of-fold half fitted ({m.best_iteration_} trees)")
        scored = keys.with_columns(p1=p1)
        C = np.empty((len(y), len(CLUSTER_COLS)), dtype=np.float32)
        r1s = r1.unique().sort()
        row_of = scored.with_row_index("row")
        for i in range(0, len(r1s), S1_CHUNK):                 # cluster features only look within an S1
            part = row_of.join(pl.DataFrame({"r1": r1s[i:i + S1_CHUNK]}), on="r1", how="semi")
            C[part["row"].to_numpy()] = cluster_features(part.select("r1", "r2", "p1"), norm, world=TRAIN_WORLD).to_numpy()
        del X                                                   # raw matrix no longer needed
        # append the cluster columns to the binned dataset (no 10 GB hstack of raw matrices)
        data.add_features_from(lgb.Dataset(C, free_raw_data=True, params={"verbose": -1}).construct())
        del C
        second = fit_subset(data, np.where(~stop)[0], np.where(stop)[0])
        log(f"second pass: {second.best_iteration_} trees")
        bundle = {"model": second, "feat_cols": cols + CLUSTER_COLS, "stage1": first, "feat1": cols, "thr": thr}
    with open(WORK_DIR / "model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    log(f"final matcher saved (threshold {thr:.2f} from {tag}, test-like world) -> {WORK_DIR / 'model.pkl'}")


if __name__ == "__main__":
    start_memory_guard("final")
    if sys.argv[1] == "ranker":
        fit_ranker()
    else:
        fit_matcher(sys.argv[2])
