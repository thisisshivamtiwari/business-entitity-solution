"""Run on the GPU instance: exact cosine nearest neighbours within each country.

  python knn.py <split> [--exclude-s1 file]
    forward: every S1  -> top-20 S2/S3 records     -> knn_<tag>_fwd.parquet  (r1, r2, cos, erank)
    reverse: every S2/S3 -> top-10 S1 records      -> knn_<tag>_rev.parquet  (r2, r1, rcos, rev_rank)
  --exclude-s1: S1 rids to leave out of the S1 side (the test-like world), tag = <split>like
Record ids (rid) = row numbers of <split>_emb.npy; meta_<split>.parquet gives rid, src, country.
"""
import argparse, time
from pathlib import Path
import numpy as np, polars as pl, torch

W = Path("/workspace/emb")
K_FWD, K_REV = 20, 10
dev = torch.device("cuda")


def topk_search(E, q_ids, db_ids, k, q_block=4096, d_block=200_000):
    """Exact top-k by cosine: queries x database, both given as rid arrays into E."""
    db = torch.from_numpy(np.asarray(E[db_ids])).to(dev)             # whole country database on GPU (fp16)
    out_i = np.empty((len(q_ids), k), np.int64); out_s = np.empty((len(q_ids), k), np.float32)
    for j in range(0, len(q_ids), q_block):
        q = torch.from_numpy(np.asarray(E[q_ids[j:j + q_block]])).to(dev)
        best_s = torch.full((len(q), k), -9.0, device=dev, dtype=torch.float16)
        best_i = torch.zeros((len(q), k), dtype=torch.int64, device=dev)
        for i in range(0, len(db_ids), d_block):
            s = q @ db[i:i + d_block].T                    # fp16 tile: 4096 x 200k = 1.6 GB
            v, ix = torch.topk(s, min(k, s.shape[1]), dim=1)
            allv = torch.cat([best_s, v], 1); alli = torch.cat([best_i, ix + i], 1)
            best_s, o = torch.topk(allv, k, dim=1); best_i = torch.gather(alli, 1, o)
        out_i[j:j + len(q)] = best_i.cpu().numpy(); out_s[j:j + len(q)] = best_s.float().cpu().numpy()
    del db; torch.cuda.empty_cache()
    return db_ids[out_i], out_s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("--exclude-s1")
    a = ap.parse_args()
    tag = a.split + ("like" if a.exclude_s1 else "")
    E = np.load(W / f"{a.split}_emb.npy", mmap_mode="r")
    meta = pl.read_parquet(W / f"meta_{a.split}.parquet")
    assert meta.height == E.shape[0], "meta rows != embedding rows"
    s1_meta = meta.filter(pl.col("src") == 1)
    if a.exclude_s1:
        s1_meta = s1_meta.join(pl.read_parquet(a.exclude_s1).select("rid"), on="rid", how="anti")
    fwd, rev = [], []
    t0 = time.time()
    for c in meta["country"].unique().sort().to_list():
        s1 = np.sort(s1_meta.filter(pl.col("country") == c)["rid"].to_numpy())
        s23 = np.sort(meta.filter((pl.col("country") == c) & (pl.col("src") > 1))["rid"].to_numpy())
        r2, cs = topk_search(E, s1, s23, K_FWD)
        fwd.append(pl.DataFrame({"r1": np.repeat(s1, K_FWD).astype(np.uint32), "r2": r2.ravel().astype(np.uint32),
                                 "cos": cs.ravel(), "erank": np.tile(np.arange(K_FWD, dtype=np.uint8), len(s1))}))
        print(f"[{tag}] {c}: forward {len(s1):,} S1 x {len(s23):,} records ({time.time() - t0:.0f}s)", flush=True)
        r1, cs = topk_search(E, s23, s1, K_REV)
        rev.append(pl.DataFrame({"r2": np.repeat(s23, K_REV).astype(np.uint32), "r1": r1.ravel().astype(np.uint32),
                                 "rcos": cs.ravel(), "rev_rank": np.tile(np.arange(K_REV, dtype=np.uint8), len(s23))}))
        print(f"[{tag}] {c}: reverse {len(s23):,} records x {len(s1):,} S1 ({time.time() - t0:.0f}s)", flush=True)
    pl.concat(fwd).write_parquet(W / f"knn_{tag}_fwd.parquet")
    pl.concat(rev).write_parquet(W / f"knn_{tag}_rev.parquet")
    print(f"DONE {tag} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
