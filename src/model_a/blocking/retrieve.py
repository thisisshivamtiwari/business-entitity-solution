"""Stage 4: candidate generation. Per country, score every S1 record against every S2+S3
record (exact top-k by chunked matmul on the GPU when CUDA is present, else FAISS HNSW),
take each S1 record's top-k neighbours on the full_text view, union
the top-k' neighbours on the name_text view (helps when one side lacks an address) and
any same-country record with the identical name key, and write candidate_pairs.tsv.
Candidates are written in rank order: full_text neighbours first, best first.
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def exact_topk_torch(emb, db_rows, q_rows, k, min_sim, q_chunk=2048, db_chunk=1_000_000, dev="cuda"):
    """Exact inner-product top-k on the GPU: no index, no approximation.
    Returns {query_row: [(db_position, sim), ...]} in rank order."""
    import torch
    db = torch.from_numpy(np.ascontiguousarray(emb[db_rows])).to(dev)          # float16, N x d
    out = {}
    k = min(k, len(db_rows))
    for qs in range(0, len(q_rows), q_chunk):
        qr = q_rows[qs:qs + q_chunk]
        q = torch.from_numpy(np.ascontiguousarray(emb[qr])).to(dev)             # float16
        best_s = torch.full((len(qr), k), -2.0, device=dev, dtype=torch.float32)
        best_i = torch.zeros((len(qr), k), device=dev, dtype=torch.int64)
        for ds in range(0, len(db_rows), db_chunk):
            sims = q.float() @ db[ds:ds + db_chunk].float().T                   # Q x chunk, fp32 math
            s_c, i_c = sims.topk(min(k, sims.shape[1]), dim=1)
            cat_s = torch.cat([best_s, s_c], dim=1)
            cat_i = torch.cat([best_i, i_c + ds], dim=1)
            best_s, sel = cat_s.topk(k, dim=1)
            best_i = torch.gather(cat_i, 1, sel)
        bs, bi = best_s.cpu().numpy(), best_i.cpu().numpy()
        for j, r in enumerate(qr):
            keep = bs[j] >= min_sim
            out[r] = list(zip(bi[j][keep].tolist(), bs[j][keep].tolist()))
    del db
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def build_index(xb, kind, nprobe, ef_search):
    d, n = xb.shape[1], xb.shape[0]
    if kind == "auto":
        kind = "flat" if n <= 300_000 else "hnsw"
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = ef_search
    elif kind == "ivf":
        nlist = max(64, int(4 * math.sqrt(n)))
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(xb[np.random.choice(n, min(n, 50 * nlist), replace=False)])
        index.nprobe = nprobe
    else:
        raise ValueError(kind)
    if kind != "hnsw" and hasattr(faiss, "StandardGpuResources"):
        try:
            if faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_all_gpus(index)
        except Exception:
            pass
    index.add(xb)
    return index, kind


def search(index, emb, q_rows, k, min_sim, chunk):
    """Return {query_row: [(db_position, sim), ...]} in rank order."""
    out = {}
    for start in range(0, len(q_rows), chunk):
        qr = q_rows[start:start + chunk]
        xq = np.ascontiguousarray(emb[qr], dtype=np.float32)
        sims, nn = index.search(xq, k)
        for i, r in enumerate(qr):
            keep = (nn[i] >= 0) & (sims[i] >= min_sim)
            out[r] = list(zip(nn[i][keep].tolist(), sims[i][keep].tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--s1-ids", default=None, help="optional file restricting queried S1 ids (validation)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=25, help="neighbours on the full_text view")
    ap.add_argument("--name-k", type=int, default=10, help="neighbours on the name_text view (0 = off)")
    ap.add_argument("--min-sim", type=float, default=0.0)
    ap.add_argument("--index", default="auto", choices=["auto", "exact", "flat", "ivf", "hnsw"],
                    help="auto = exact GPU search when CUDA is available, else hnsw")
    ap.add_argument("--ef-search", type=int, default=128)
    ap.add_argument("--nprobe", type=int, default=48)
    ap.add_argument("--exact-name-cap", type=int, default=30)
    ap.add_argument("--query-chunk", type=int, default=50_000)
    ap.add_argument("--shard", default=None, help="i/n: handle the i-th contiguous slice of the queried S1 rows (one process per GPU)")
    ap.add_argument("--gpu-q-chunk", type=int, default=2048, help="exact search: queries per GPU batch (lower = less GPU memory)")
    ap.add_argument("--gpu-db-chunk", type=int, default=1_000_000, help="exact search: database rows per GPU batch")
    args = ap.parse_args()
    work = Path(args.work_dir)
    np.random.seed(0)
    if args.index in ("auto", "exact"):
        import shutil
        import subprocess
        has_cuda = bool(shutil.which("nvidia-smi")) and subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True).returncode == 0      # no torch import next to faiss
        if has_cuda:
            args.index = "exact"
        else:
            print("[retrieve] no CUDA: falling back to hnsw" if args.index == "exact" else
                  "[retrieve] no CUDA: using hnsw")
            args.index = "hnsw"

    rec = pd.read_parquet(work / f"{args.split}_records.parquet",
                          columns=["entity_id", "source", "country", "name_key"])
    views = {"full_text": np.load(work / f"{args.split}_full_text.npy", mmap_mode="r")}
    if args.name_k:
        views["name_text"] = np.load(work / f"{args.split}_name_text.npy", mmap_mode="r")
    for v in views.values():
        assert len(v) == len(rec), "embeddings and records out of sync; re-run encode"
    ids = rec.entity_id.to_numpy()
    source = rec.source.to_numpy()
    country = rec.country.to_numpy()

    s1_mask = source == 1
    if args.s1_ids:
        keep = set(Path(args.s1_ids).read_text().split())
        s1_mask = s1_mask & rec.entity_id.isin(keep).to_numpy()
    s1_rows = np.flatnonzero(s1_mask)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        s1_rows = np.array_split(s1_rows, n)[i]
        print(f"[retrieve] shard {i}/{n}: {len(s1_rows):,} S1 rows", flush=True)
    cands = {}
    stats = {"k": args.k, "name_k": args.name_k, "index": args.index, "countries": {}}

    for c in pd.unique(country[s1_rows]):
        q_rows = s1_rows[country[s1_rows] == c]
        db_rows = np.flatnonzero((source != 1) & (country == c))
        if len(db_rows) == 0:
            for r in q_rows:
                cands[r] = []
            print(f"[retrieve] {c}: no S2/S3 records, {len(q_rows):,} S1 left empty")
            continue
        merged = {r: {} for r in q_rows}                  # row -> {db_row: best_sim}, insertion = rank
        cstat = {"queries": int(len(q_rows)), "db": int(len(db_rows))}
        for view, emb in views.items():
            k = args.k if view == "full_text" else args.name_k
            t0 = time.time()
            if args.index == "exact":
                print(f"[retrieve] {c}/{view}: {len(q_rows):,} queries vs {len(db_rows):,} db (exact, GPU)", flush=True)
                hits = exact_topk_torch(emb, db_rows, q_rows, k, args.min_sim, q_chunk=args.gpu_q_chunk, db_chunk=args.gpu_db_chunk)
                index = xb = None
            else:
                xb = np.ascontiguousarray(emb[db_rows], dtype=np.float32)
                index, kind = build_index(xb, args.index, args.nprobe, args.ef_search)
                print(f"[retrieve] {c}/{view}: {len(q_rows):,} queries vs {len(db_rows):,} db "
                      f"({kind}, built in {time.time()-t0:.0f}s)", flush=True)
                hits = search(index, emb, q_rows, k, args.min_sim, args.query_chunk)
            added = 0
            for r, lst in hits.items():
                for pos, sim in lst:
                    dbr = int(db_rows[pos])
                    if dbr not in merged[r]:
                        merged[r][dbr] = sim; added += 1
            cstat[f"{view}_added"] = added
            print(f"[retrieve] {c}/{view}: searched in {time.time()-t0:.0f}s, +{added:,} candidates", flush=True)
            del index, xb
        if args.exact_name_cap:
            block = defaultdict(list)
            nk = rec.name_key.to_numpy()
            for r in db_rows:
                if nk[r]:
                    block[nk[r]].append(int(r))
            added = 0
            for r in q_rows:
                blk = block.get(nk[r], ())
                if 0 < len(blk) <= args.exact_name_cap:
                    for dbr in blk:
                        if dbr not in merged[r]:
                            merged[r][dbr] = 0.0; added += 1
            cstat["exact_name_added"] = added
        for r in q_rows:
            cands[r] = list(merged[r])
        stats["countries"][c] = cstat

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for r in s1_rows:
            f.write(f"{ids[r]}\t{','.join(ids[x] for x in cands.get(r, []))}\n")
    total = sum(len(v) for v in cands.values())
    stats.update(total_candidates=total, s1_rows=int(len(s1_rows)))
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[retrieve] {len(s1_rows):,} S1 rows, {total:,} candidates "
          f"({total/max(1,len(s1_rows)):.1f} per S1) -> {out}")


if __name__ == "__main__":
    main()
