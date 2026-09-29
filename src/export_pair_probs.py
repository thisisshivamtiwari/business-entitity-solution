"""Model B -> the stacker: write Model B's pair probabilities in the format the ensemble reads.

    python export_pair_probs.py fold0    # after `train.py 400000 442001`: every fold-0 candidate, out of sample
    python export_pair_probs.py test     # after the final run: every test candidate, from the all-data model

Writes work/ensemble/model_b/{fold0,test}_pair_probs.parquet with columns
    s1_id, cand_id, rank, ber_prob, emb_cosine, cand_addr_missing, from_namesearch, selected (+ label for fold 0)
    rank            Model B's ranker position (0-11); 12 for candidates added by the name-only search
    ber_prob        Model B's match probability (second pass)
    selected        1 if Model B's own decision rule (threshold 0.85, one S1 per record, rescue 0.5) keeps the pair
"""
import sys

import polars as pl

import embeddings as EMB
from config import WORK_DIR, norm_path
from evaluate import truth_rids
from memguard import start_memory_guard
from train import select

OUT = WORK_DIR / "ensemble" / "model_b"
THR, RESCUE = 0.85, 0.5
TOPK = 12


def pair_table(pred, split):
    """pred: (r1, r2, p[, y]) -> the exported columns, sorted by S1 and Model B's rank."""
    n = pl.read_parquet(norm_path(split), columns=["rid", "entity_id", "f_addr_empty"])
    brank = pl.read_parquet(WORK_DIR / "cand" / f"{split}.parquet", columns=["r1", "r2", "brank"]).filter(pl.col("brank") < TOPK)
    chosen = select(pred.select("r1", "r2", "p"), THR, rescue=RESCUE).with_columns(selected=pl.lit(1, pl.Int8))
    x = (pred.join(brank, on=["r1", "r2"], how="left").join(chosen, on=["r1", "r2"], how="left")
         .with_columns(from_namesearch=pl.col("brank").is_null().cast(pl.Int8), rank=pl.col("brank").fill_null(TOPK).cast(pl.Int64),
                       selected=pl.col("selected").fill_null(0)))
    x = x.with_columns(emb_cosine=pl.Series(EMB.pair_cosine(split, x["r1"].to_numpy(), x["r2"].to_numpy()), dtype=pl.Float32))
    x = (x.join(n.select(r1="rid", s1_id="entity_id"), on="r1")
         .join(n.select(r2="rid", cand_id="entity_id", cand_addr_missing=pl.col("f_addr_empty").cast(pl.Int8)), on="r2"))
    cols = ["s1_id", "cand_id", "rank", pl.col("p").cast(pl.Float32).alias("ber_prob"), "emb_cosine", "cand_addr_missing",
            "from_namesearch", "selected"]
    if "y" in x.columns:
        cols.append(pl.col("y").cast(pl.Int64).alias("label"))
    return x.sort("r1", "rank", "r2").select(cols)


def main(which):
    start_memory_guard("export_pair_probs")
    OUT.mkdir(parents=True, exist_ok=True)
    if which == "fold0":
        _, s1 = truth_rids()
        f0 = s1.filter(pl.col("fold") == 0).select("r1")
        pred = pl.read_parquet(WORK_DIR / "val_pred.parquet")
        assert pred.join(f0, on="r1", how="anti").height == 0 and pred["r1"].n_unique() > 0.99 * f0.height, \
            "work/val_pred.parquet must hold all fold-0 S1: run `train.py 400000 442001` first"
        t = pair_table(pred, "train")
    else:
        t = pair_table(pl.read_parquet(WORK_DIR / "test_pred.parquet"), "test")
    t.write_parquet(OUT / f"{which}_pair_probs.parquet")
    print(f"{which}: {t.height:,} pairs, {t['s1_id'].n_unique():,} S1 -> {OUT / f'{which}_pair_probs.parquet'}")


if __name__ == "__main__":
    main(sys.argv[1])
