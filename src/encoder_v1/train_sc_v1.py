"""Encoder v1: fine-tune intfloat/multilingual-e5-small with a supervised contrastive loss.
Each ground-truth cluster (an S1 record and its matches) is one class. A batch holds
P clusters x K records; every other record in the batch is a negative.
Training uses the 95% of S1 marked 'train' by prepare_v1.py.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from common_v1 import serialise


def supcon_loss(emb, labels, temperature):
    sim = emb @ emb.T / temperature
    eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, -1e4)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    per_anchor = -(log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)
    return per_anchor[pos.any(1)].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--base-model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--clusters-per-batch", type=int, default=128)
    ap.add_argument("--records-per-cluster", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=96)
    ap.add_argument("--addr-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    work = Path(args.work_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rec = pd.read_parquet(work / "train_records.parquet", columns=["entity_id", "name_norm", "addr_norm", "country"]).set_index("entity_id")
    pairs = pd.read_parquet(work / "train_pairs.parquet")
    pairs = pairs[pairs.fold == "train"]
    clusters = {s1: [s1] + m for s1, m in pairs.groupby("s1_id").match_id.apply(list).items()}
    keys = list(clusters)
    print(f"[train-v1] device={device} clusters={len(keys):,}")

    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_seq_len
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.06, anneal_strategy="linear")
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    tok = getattr(model, "preprocess", None) or model.tokenize

    def text_of(eid, anchor):
        r = rec.loc[eid]
        addr = "" if (not anchor and r.addr_norm and random.random() < args.addr_dropout) else r.addr_norm
        return serialise(r.name_norm, addr, r.country)

    model.train(); t0 = time.time()
    for step in range(1, args.steps + 1):
        texts, labels = [], []
        for ci, s1 in enumerate(random.sample(keys, min(args.clusters_per_batch, len(keys)))):
            members = clusters[s1]
            for eid in [s1] + random.sample(members[1:], min(args.records_per_cluster - 1, len(members) - 1)):
                texts.append(text_of(eid, anchor=(eid == s1))); labels.append(ci)
        feats = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tok(texts).items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            emb = model(feats)["sentence_embedding"]
        loss = supcon_loss(F.normalize(emb.float(), dim=-1), torch.tensor(labels, device=device), args.temperature)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 50 == 0 or step == 1:
            print(f"[train-v1] step {step}/{args.steps} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    model.save(args.out_dir)
    print(f"[train-v1] saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
