"""Cluster-consistency features (item 3c of the improvement plan): a second pass.

Why (eda/Errors_iter2.ipynb, eda/Funnel.ipynb): the matcher rejects true pairs whose evidence
against the S1 record is weak: no address (48% of its misses), only a partial address, or an
unrelated name (address-only matches). Yet the other records of the same business usually DO match
the S1 confidently, and they resemble each other (same name spelling, same address, same house
number). So we ask a second question:

    "does this candidate look like the records we are already confident belong to this S1?"

How:
  1. A first-pass model gives p1 for every candidate pair (out-of-fold for training S1 records, so
     the second model never sees leaked, over-confident scores).
  2. For each S1, its confident matches are candidates with p1 >= CONFIDENT.
  3. Every candidate is compared with the S1's OTHER confident matches (never with itself):
     best name similarity, best address similarity, shared house number, and how many there are.
  4. p1 itself, its rank within the S1, and its gap to the S1's best p1 are added too.
The second-pass model is trained on base + these features.
"""
import os
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz

import embeddings as EMB

CONFIDENT = 0.9
# BER_CLUSTER_EMB=1 (needs BER_EMB=1): also compare each candidate's EMBEDDING with the S1's other confident matches.
# Names and addresses of a no-address record say little, but its embedding sits close to the right branch's cluster.
CLUSTER_EMB = os.environ.get("BER_CLUSTER_EMB", "0") == "1" and EMB.ENABLED
CLUSTER_COLS = ["p1", "p1_rank", "p1_gap", "n_conf_others", "conf_name_sim", "conf_addr_sim",
                "conf_house_eq", "conf_same_source"]
EMB_CLUSTER_COLS = ["conf_emb_max", "conf_emb_mean", "conf_emb_gap"]
if CLUSTER_EMB:
    CLUSTER_COLS = CLUSTER_COLS + EMB_CLUSTER_COLS


def _sims(args):
    n1, n2, a1, a2 = args
    out = np.empty((len(n1), 2), dtype=np.float32)
    for i in range(len(n1)):
        out[i, 0] = fuzz.token_set_ratio(n1[i], n2[i])
        out[i, 1] = fuzz.token_set_ratio(a1[i], a2[i]) if a1[i] and a2[i] else -1
    return out


def cluster_features(scored, norm, procs=12, block=200_000, world="train"):
    """scored: r1, r2, p1 for every candidate of the S1 records in scope (same row order is kept).

    norm: normalised records of the world (rid == row number). Returns CLUSTER_COLS, row-aligned.
    world: which embedding table to read for the embedding columns (train, trainlike or test).
    """
    scored = scored.with_row_index("row")
    conf = scored.filter(pl.col("p1") >= CONFIDENT).select("r1", r2c="r2")
    x = (scored.select("row", "r1", "r2").join(conf, on="r1")
         .filter(pl.col("r2") != pl.col("r2c")))                     # candidate x OTHER confident matches
    fields = norm.select("core", "addr", "src", house=pl.col("addr").str.extract(r"(\d+)"))
    a = fields[x["r2"].to_numpy()]
    b = fields[x["r2c"].to_numpy()]
    data = [a["core"].to_list(), b["core"].to_list(), a["addr"].to_list(), b["addr"].to_list()]
    jobs = [[col[i:i + block] for col in data] for i in range(0, x.height, block)]
    if jobs:
        with Pool(procs) as pool:
            sims = np.vstack(pool.map(_sims, jobs, chunksize=1))
    else:
        sims = np.zeros((0, 2), dtype=np.float32)
    x = x.with_columns(
        name_sim=pl.Series(sims[:, 0]), addr_sim=pl.Series(sims[:, 1]),
        house_eq=(a["house"] == b["house"]).fill_null(False).to_numpy(),
        same_source=(a["src"] == b["src"]).to_numpy(),
    )
    if CLUSTER_EMB:                                   # cosine to each of the S1's other confident matches
        x = x.with_columns(emb_sim=pl.Series(EMB.pair_cosine(world, x["r2"].to_numpy(), x["r2c"].to_numpy())))
    emb_aggs = [pl.col("emb_sim").max().alias("conf_emb_max"), pl.col("emb_sim").mean().alias("conf_emb_mean")] \
        if CLUSTER_EMB else []
    agg = x.group_by("row").agg(
        *emb_aggs,
        n_conf_others=pl.len(),
        conf_name_sim=pl.col("name_sim").max(),
        conf_addr_sim=pl.col("addr_sim").max(),
        conf_house_eq=pl.col("house_eq").any(),
        conf_same_source=pl.col("same_source").sum(),
    )
    out = (scored.select("row", "r1", "p1")
           .with_columns(p1_rank=pl.col("p1").rank("ordinal", descending=True).over("r1"),
                         p1_gap=pl.col("p1") - pl.col("p1").max().over("r1"))
           .join(agg, on="row", how="left").sort("row")
           .with_columns(pl.col("n_conf_others", "conf_same_source").fill_null(0),
                         pl.col("conf_name_sim", "conf_addr_sim").fill_null(-1),
                         pl.col("conf_house_eq").fill_null(False)))
    if CLUSTER_EMB:
        # no confident other match -> neutral value; gap = this candidate's cluster similarity minus the best
        # cluster similarity among the S1's candidates ("is it the candidate closest to the cluster?")
        out = out.with_columns(pl.col("conf_emb_max", "conf_emb_mean").fill_null(-1.0)).with_columns(
            conf_emb_gap=pl.col("conf_emb_mean") - pl.col("conf_emb_mean").max().over("r1"))
    return out.select(pl.col(CLUSTER_COLS).cast(pl.Float32))
