"""Step 3 - learned candidate ranker: keep the best KEEP of the ~100 raw candidates per S1.

Blocking ranks candidates by a hand-made score (sum of 1/key-frequency), which pushes
about half of the true matches it finds below the top 20. This step learns a better
ranking from the training labels using only cheap blocking statistics:
    - evidence per key type (s_nn, s_na, ... from block.py)
    - how the candidate compares with the other candidates of the same S1
    - how many S1 records compete for the same S2/S3 record (each belongs to at most one)
Measured on the validation fold, 10 learned candidates recover more true matches than 20
crude ones, so the matcher gets a shorter AND better list.

Usage:
    python rerank.py fit                 # train on folds 1-4, print recall on fold 0
    python rerank.py apply train test    # write work/cand/{split}.parquet (top KEEP per S1)
"""
import pickle
import sys

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

import embeddings as EMB
from block import KEY_TYPES
from config import SEED, WORK_DIR
from evaluate import truth_rids
from memguard import start_memory_guard

KEEP = int(__import__("os").environ.get("BER_KEEP", 12))   # candidates per S1 passed on to the matcher
S1_PER_CHUNK = 300_000     # features are built for this many S1 at a time (memory bound)
TYPE_COLS = [f"s_{t}" for t in KEY_TYPES]
MODEL_PATH = WORK_DIR / "ranker.pkl"


def load_raw(split):
    return pl.read_parquet(WORK_DIR / "cand_raw" / f"{split}.parquet")


def load_pool(split, s1=None):
    """Ranker input: keyword candidates with competition features; with BER_EMB=1 also the embedding
    neighbours and embedding features (embeddings.merge_pool). s1: optional S1 subset (r1 column)."""
    cand = add_competition(load_raw(split))
    if s1 is not None:
        cand = cand.join(s1.select("r1"), on="r1", how="semi")
    if EMB.ENABLED:
        cand = EMB.merge_pool(cand, split)
    return cand


def add_competition(cand):
    """Features that look across S1 records, so they need the FULL candidate table:
    for each S2/S3 record, how many S1 records claim it and how this S1 ranks among them."""
    return cand.with_columns(
        rrank=pl.col("bscore").rank("ordinal", descending=True).over("r2").cast(pl.Float32),
        n_rev=pl.len().over("r2").cast(pl.Float32),
        rev_gap=pl.col("bscore") / pl.col("bscore").max().over("r2"),
    )


def add_within_s1(cand):
    """Features comparing a candidate with the other candidates of the same S1.
    Only needs whole S1 groups, so it can run chunk by chunk."""
    return cand.with_columns(
        b_gap=pl.col("bscore") / pl.col("bscore").max().over("r1"),
        n_cand=pl.len().over("r1").cast(pl.Float32),
        n_types=pl.sum_horizontal([(pl.col(t) > 0).cast(pl.UInt8) for t in TYPE_COLS]),
        **{f"{t}_gap": pl.col(t) - pl.col(t).max().over("r1") for t in TYPE_COLS},
    ).pipe(lambda c: EMB.within_s1(c) if "cos" in c.columns else c)


def feature_columns(frame):
    return [c for c in frame.columns if c not in ("r1", "r2", "y")]


def s1_chunks(cand, size=S1_PER_CHUNK):
    """Yield sub-tables that each contain complete S1 groups."""
    r1s = cand["r1"].unique().sort()
    for i in range(0, len(r1s), size):
        yield cand.join(pl.DataFrame({"r1": r1s[i:i + size]}), on="r1", how="semi")


def to_matrix(frame, cols):
    return frame.select(cols).to_numpy().astype(np.float32)


def fit(n_train=300_000, n_val=150_000):
    truth, s1 = truth_rids()
    train_s1 = s1.filter(pl.col("fold") != 0).sample(n_train, seed=SEED)
    val_s1 = s1.filter(pl.col("fold") == 0).sample(n_val, seed=SEED)
    cand = load_pool("train", pl.concat([train_s1, val_s1]))
    labels = truth.select("r1", "r2", y=pl.lit(True))

    def labelled(sample):
        x = cand.join(sample.select("r1"), on="r1", how="semi")
        x = add_within_s1(x).join(labels, on=["r1", "r2"], how="left")
        return x.with_columns(pl.col("y").fill_null(False)).sort("r1", "r2")

    xtr, xva = labelled(train_s1), labelled(val_s1)
    del cand

    cols = feature_columns(xtr)
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
        random_state=SEED)
    model.fit(to_matrix(xtr, cols), xtr["y"].to_numpy())

    # Recall of the true pairs among the top-k candidates: crude score vs learned score.
    xva = xva.with_columns(rscore=model.predict_proba(to_matrix(xva, cols))[:, 1])
    xva = xva.sort(["rscore", "r2"], descending=[True, False]).with_columns(
        lrank=pl.int_range(pl.len()).over("r1"))
    n_true = truth.join(val_s1, on="r1", how="semi").height
    for k in (5, 10, KEEP, 20, 30):
        crude = xva.filter(pl.col("brank") < k)["y"].sum() / n_true
        learned = xva.filter(pl.col("lrank") < k)["y"].sum() / n_true
        print(f"  top-{k:<2} recall: crude {crude:.4f} -> learned {learned:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "cols": cols}, f)
    print(f"ranker saved -> {MODEL_PATH}")


def apply(split, keep=KEEP):
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model, cols = bundle["model"], bundle["cols"]

    cand = add_competition(load_raw(split))       # competition features need the full table
    kept = []
    for part in s1_chunks(cand):
        if EMB.ENABLED:
            part = EMB.merge_pool(part, split)     # embedding channel, merged chunk by chunk (memory)
        part = add_within_s1(part)
        part = part.with_columns(rscore=pl.Series(
            model.predict_proba(to_matrix(part, cols))[:, 1], dtype=pl.Float32))
        kept.append(
            part.select("r1", "r2", "bscore", "nkeys", "minn1", *TYPE_COLS, "rscore",
                        *[c for c in EMB.EMB_COLS if c in part.columns])
            .sort(["rscore", "r2"], descending=[True, False])
            .with_columns(brank=pl.int_range(pl.len(), dtype=pl.UInt32).over("r1"))
            .filter(pl.col("brank") < keep)
        )
    out = pl.concat(kept)

    path = WORK_DIR / "cand" / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    print(f"{split}: kept {out.height:,} candidates ({out.height / out['r1'].n_unique():.1f} per S1)"
          f" -> {path}")


if __name__ == "__main__":
    start_memory_guard("rerank")
    if len(sys.argv) < 2 or sys.argv[1] not in ("fit", "apply"):
        sys.exit(__doc__)
    if sys.argv[1] == "fit":
        fit()
    else:
        for split in sys.argv[2:] or ["train", "test"]:
            apply(split)
