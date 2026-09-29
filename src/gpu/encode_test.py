"""Run on the GPU instance: encode test texts with sc_encoder (v1 recipe), float16, L2-normalised,
row i = test rid i. Resumable: writes chunks, then one test_emb.npy. First checks it reproduces
the stored training vectors on 5,000 records."""
import sys, time
from pathlib import Path
import numpy as np, polars as pl, torch
from sentence_transformers import SentenceTransformer

W = Path("/workspace/emb")
CHUNK = 500_000

def main():
    m = SentenceTransformer(str(W / "sc_encoder"), device="cuda")
    m.half()
    chk = pl.read_parquet(W / "train_check_text.parquet")["text"].to_list()
    ref = np.load(W / "train_check_emb.npy").astype(np.float32)
    v = m.encode(chk, batch_size=1024, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    c = (v * ref).sum(1)
    print(f"check vs stored train vectors: mean {c.mean():.5f} min {c.min():.5f}", flush=True)
    if c.min() < 0.995:
        sys.exit("encoder output does not match the stored training embeddings - stopping")
    texts = pl.read_parquet(W / "test_text.parquet")["text"].to_list()
    n = len(texts); (W / "chunks").mkdir(exist_ok=True); t0 = time.time()
    for i in range(0, n, CHUNK):
        f = W / "chunks" / f"{i:09d}.npy"
        if f.exists():
            continue
        e = m.encode(texts[i:i + CHUNK], batch_size=1024, normalize_embeddings=True, convert_to_numpy=True)
        np.save(f, e.astype(np.float16))
        done = min(i + CHUNK, n)
        print(f"{done:,}/{n:,} encoded ({done / (time.time() - t0):,.0f}/s)", flush=True)
    out = np.lib.format.open_memmap(W / "test_emb.npy", mode="w+", dtype=np.float16, shape=(n, 384))
    for i in range(0, n, CHUNK):
        out[i:i + CHUNK] = np.load(W / "chunks" / f"{i:09d}.npy")
    out.flush()
    print(f"DONE test_emb.npy {out.shape} in {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
