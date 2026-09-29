"""Encoder v1: embed every record of a split. Writes {split}_emb.npy (float16, L2-normalised);
row i is row i of {split}_records.parquet, which is the record order of the raw files
(source 1, then 2, then 3). Resumable per chunk."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from common_v1 import serialise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=200_000)
    args = ap.parse_args()
    work = Path(args.work_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec = pd.read_parquet(work / f"{args.split}_records.parquet", columns=["name_norm", "addr_norm", "country"])
    model = SentenceTransformer(args.model_dir, device=device)
    model.max_seq_length = args.max_seq_len
    if device == "cuda":
        model.half()
    out, done_path = work / f"{args.split}_emb.npy", work / f"{args.split}_emb.done"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    emb = np.lib.format.open_memmap(out, mode="r+" if out.exists() else "w+", dtype=np.float16,
                                    shape=(len(rec), model.get_sentence_embedding_dimension()))
    for start in range(0, len(rec), args.chunk):
        if str(start) in done:
            continue
        sub = rec.iloc[start:start + args.chunk]
        texts = [serialise(a, b, c) for a, b, c in zip(sub.name_norm, sub.addr_norm, sub.country)]
        with torch.inference_mode():
            vec = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        emb[start:start + len(sub)] = vec.astype(np.float16); emb.flush()
        with done_path.open("a") as f:
            f.write(f"{start}\n")
        print(f"[encode-v1] {min(start + args.chunk, len(rec)):,}/{len(rec):,}", flush=True)
    print(f"[encode-v1] -> {out}")


if __name__ == "__main__":
    main()
