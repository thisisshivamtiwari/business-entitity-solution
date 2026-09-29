"""Stage 3: embed every record of a split in two views with the fine-tuned encoder.

Writes, in --work-dir (row i == row i of {split}_records.parquet / {split}_ids.parquet):
  {split}_full_text.npy   float16, L2-normalised   "name | address"  (name only if no address)
  {split}_name_text.npy   float16, L2-normalised   name only
  meta.json
Resumable per chunk.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from common import PREFIX, full_text, name_text, pick_device

VIEWS = ("full_text", "name_text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--shard", default=None, help="i/n: encode only every n-th chunk starting at i (one process per GPU)")
    ap.add_argument("--init-only", type=int, default=0, help="create the output files for this embedding dimension and exit")
    args = ap.parse_args()
    work = Path(args.work_dir)
    sh_i, sh_n = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)

    rec = pd.read_parquet(work / f"{args.split}_records.parquet", columns=["name_clean", "addr_clean"])
    n = len(rec)
    if args.init_only:
        for view in args.views:
            out_path = work / f"{args.split}_{view}.npy"
            if not out_path.exists():
                np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, args.init_only)).flush()
        print(f"[encode] output files ready for {n:,} records x {args.init_only} dims")
        return
    device = pick_device()
    model = SentenceTransformer(args.model_dir, device=device)
    model.max_seq_length = args.max_seq_len
    if device == "cuda":
        model.half()
    dim = model.get_sentence_embedding_dimension()
    (work / "meta.json").write_text(json.dumps({
        "model": args.model_dir, "dim": dim, "prefix": PREFIX, "normalized": True,
        "max_length": args.max_seq_len, "dtype": "float16", "views": list(VIEWS)}, indent=2))

    for view in args.views:
        out_path = work / f"{args.split}_{view}.npy"
        done_path = work / f"{args.split}_{view}.done"
        done = set(done_path.read_text().split()) if done_path.exists() else set()
        emb = np.lib.format.open_memmap(out_path, mode="r+" if out_path.exists() else "w+",
                                        dtype=np.float16, shape=(n, dim))
        print(f"[encode] {args.split}/{view}: {n:,} records, dim={dim}, device={device}")
        for ci, start in enumerate(range(0, n, args.chunk)):
            if str(start) in done or ci % sh_n != sh_i:
                continue
            end = min(start + args.chunk, n)
            sub = rec.iloc[start:end]
            if view == "full_text":
                texts = [full_text(a, b) for a, b in zip(sub.name_clean, sub.addr_clean)]
            else:
                texts = [name_text(a) for a in sub.name_clean]
            with torch.inference_mode():
                vec = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
            emb[start:end] = vec.astype(np.float16)
            emb.flush()
            with done_path.open("a") as f:
                f.write(f"{start}\n")
            print(f"[encode] {view} {end:,}/{n:,}", flush=True)
        del emb
        print(f"[encode] -> {out_path}")


if __name__ == "__main__":
    main()
