"""Entity embeddings as an extra candidate channel and as features (iteration 10, BER_EMB=1).

The embeddings come from a fine-tuned multilingual-e5-small encoder (384-d, L2-normalised, float16),
one vector per record with row i = record id (rid) i:
    train (and the test-like world, same records)  Embeddings/train_emb.npy
    test                                            work/emb/test_emb.npy
Encoder input (v1 recipe): "query: name: {name} | address: {address} | country: {country}".

Nearest neighbours are computed on a GPU machine (src/gpu/knn.py, see src/gpu/README.md), per country, and
stored as
    work/emb/knn_<world>_fwd.parquet   S1 -> its 20 nearest S2/S3 records   (r1, r2, cos, erank)
    work/emb/knn_<world>_rev.parquet   S2/S3 -> its 10 nearest S1 records  (r2, r1, rcos, rev_rank)
where <world> is train, trainlike (test-like world: removed S1 excluded) or test.

Why (eda/EmbBlocking.ipynb, clean 22k-S1 evaluation): a ranker that merges our keyword candidates with
the embedding neighbours reaches 99.1% candidate recall at 8 per S1 (96.0% today at 12), mostly on
no-address and Indian-script records, without adding look-alike decoys.
"""
import os

import numpy as np
import polars as pl

from config import PROJECT_DIR, WORK_DIR

ENABLED = os.environ.get("BER_EMB", "0") == "1"
RAW_KEEP = 30        # keyword candidates per S1 that enter the merged pool (top by blocking score)
K_FWD, K_REV = 20, 10
EMB_FILES = {"train": PROJECT_DIR / "Embeddings" / "train_emb.npy",
             "trainlike": PROJECT_DIR / "Embeddings" / "train_emb.npy",
             "test": WORK_DIR / "emb" / "test_emb.npy"}
# features added to the ranker pool and to the matcher's candidates
EMB_COLS = ["cos", "erank", "rev_rank", "rev_gap_cos", "in_emb", "in_raw"]


def _emb(world):
    path = EMB_FILES[world]
    E = np.load(path, mmap_mode="r")
    n = pl.scan_parquet(WORK_DIR / "norm" / f"{world}.parquet").select(pl.len()).collect().item()
    assert E.shape[0] == n, f"{path} has {E.shape[0]:,} rows but the {world} records table has {n:,}"
    return E


def pair_cosine(world, r1, r2, chunk=500_000):
    """Cosine for pairs of record ids, read from the memory-mapped embeddings chunk by chunk."""
    E = _emb(world)
    r1, r2 = np.asarray(r1), np.asarray(r2)
    out = np.empty(len(r1), np.float32)
    for i in range(0, len(r1), chunk):
        a, b = r1[i:i + chunk], r2[i:i + chunk]
        ua, ia = np.unique(a, return_inverse=True)
        ub, ib = np.unique(b, return_inverse=True)
        va = np.asarray(E[ua], dtype=np.float32)
        vb = np.asarray(E[ub], dtype=np.float32)
        out[i:i + chunk] = np.einsum("ij,ij->i", va[ia], vb[ib])
    return out


_KNN = {}


def knn(world, direction):
    """Neighbour list, read once per process and cached (merge_pool is called chunk by chunk)."""
    key = (world, direction)
    if key not in _KNN:
        _KNN[key] = pl.read_parquet(WORK_DIR / "emb" / f"knn_{world}_{direction}.parquet")
    return _KNN[key]


def pair_features(pairs, world):
    """cos, erank, rev_rank, rev_gap_cos for (r1, r2) pairs, from the stored neighbour lists."""
    fwd = knn(world, "fwd").select("r1", "r2", erank=pl.col("erank").cast(pl.Float32))
    rev = knn(world, "rev")
    top = rev.filter(pl.col("rev_rank") == 0).select("r2", rev_top=pl.col("rcos"))
    x = (pairs.join(fwd, on=["r1", "r2"], how="left")
         .join(rev.select("r1", "r2", rev_rank=pl.col("rev_rank").cast(pl.Float32)), on=["r1", "r2"], how="left")
         .join(top, on="r2", how="left"))
    x = x.with_columns(cos=pl.Series(pair_cosine(world, x["r1"].to_numpy(), x["r2"].to_numpy()), dtype=pl.Float32))
    return x.with_columns(
        pl.col("erank").fill_null(float(K_FWD)),             # not among the S1's 20 nearest records
        pl.col("rev_rank").fill_null(float(K_REV)),          # S1 not among the record's 10 nearest S1
        rev_gap_cos=(pl.col("cos") - pl.col("rev_top").fill_null(pl.col("cos"))).cast(pl.Float32),
    ).drop("rev_top")


def merge_pool(raw, world):
    """Ranker pool = keyword top-RAW_KEEP (with competition features already added) + embedding top-20."""
    raw = raw.filter(pl.col("brank") < RAW_KEEP).with_columns(in_raw=pl.lit(1.0, pl.Float32))
    fwd = knn(world, "fwd").join(raw.select("r1").unique(), on="r1", how="semi")   # the S1 being ranked
    new = fwd.select("r1", "r2").join(raw.select("r1", "r2"), on=["r1", "r2"], how="anti")
    pool = pl.concat([raw, new], how="diagonal_relaxed")
    num = [c for c in raw.columns if c not in ("r1", "r2", "brank")]
    pool = pool.with_columns(pl.col(num).fill_null(0), pl.col("brank").fill_null(100))
    pool = pool.join(fwd.select("r1", "r2", in_emb=pl.lit(1.0, pl.Float32)), on=["r1", "r2"], how="left") \
        .with_columns(pl.col("in_emb").fill_null(0.0))
    feats = pair_features(pool.select("r1", "r2"), world)
    return pool.join(feats, on=["r1", "r2"], how="left")


def within_s1(cand):
    """Embedding features relative to the other candidates of the same S1."""
    return cand.with_columns(
        cos_gap1=pl.col("cos") - pl.col("cos").max().over("r1"),
        cos_rank1=pl.col("cos").rank("ordinal", descending=True).over("r1").cast(pl.Float32),
    )
