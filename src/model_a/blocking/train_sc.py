"""Stage 2: SC-Block training. Fine-tune a multilingual bi-encoder with a supervised
contrastive loss where each ground-truth cluster (S1 record + its matches) is a class.

Batches are P clusters x K records. A share of the P clusters are drawn as *hard
groups*: several different businesses from the same country sharing the same name key
(e.g. the 85 distinct "Twisted Nails"), so the model must separate them by address.
Only fold 1-4 pairs are used; fold 0 is the shared validation set.
"""
import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from common import full_text, pick_device


def supcon_loss(emb, labels, temperature):
    sim = emb @ emb.T / temperature
    n = sim.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, -1e4)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    per_anchor = -(log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)
    return per_anchor[pos.any(1)].mean()


def build_clusters(work, rec, max_clusters):
    pairs = pd.read_parquet(work / "train_pairs.parquet")
    pairs = pairs[pairs.fold != 0]
    clusters = pairs.groupby("s1_id").match_id.apply(list).to_dict()
    clusters = {s1: [s1] + m for s1, m in clusters.items()}
    keys = list(clusters)
    if max_clusters and len(keys) > max_clusters:
        random.shuffle(keys)
        keys = keys[:max_clusters]
        clusters = {k: clusters[k] for k in keys}
    groups = defaultdict(list)                       # (country, name_key) -> [s1 ids]
    for s1 in keys:
        groups[(rec.country[s1], rec.name_key[s1])].append(s1)
    hard = [g for g in groups.values() if len(g) >= 2]
    return clusters, keys, hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--base-model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--clusters-per-batch", type=int, default=128)
    ap.add_argument("--records-per-cluster", type=int, default=2)
    ap.add_argument("--hard-frac", type=float, default=0.4,
                    help="share of clusters per batch drawn from same-name groups (hard negatives)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--addr-dropout", type=float, default=0.05)
    ap.add_argument("--max-clusters", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    work = Path(args.work_dir)
    device = pick_device()
    print(f"[train] device={device} base={args.base_model}")

    rec = pd.read_parquet(work / "train_records.parquet",
                          columns=["entity_id", "name_clean", "addr_clean", "country", "name_key"])
    rec = rec.set_index("entity_id")
    clusters, keys, hard = build_clusters(work, rec, args.max_clusters)
    print(f"[train] {len(keys):,} clusters, {sum(len(v) for v in clusters.values()):,} records, "
          f"{len(hard):,} same-name hard groups")

    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_seq_len
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.06, anneal_strategy="linear")
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    tok = getattr(model, "preprocess", None) or model.tokenize

    def text_of(eid, anchor):
        r = rec.loc[eid]
        addr = r.addr_clean
        if not anchor and addr and random.random() < args.addr_dropout:
            addr = ""
        return full_text(r.name_clean, addr)

    def sample_batch():
        chosen, seen = [], set()
        n_hard = int(args.clusters_per_batch * args.hard_frac) if hard else 0
        while len(chosen) < n_hard:
            for s1 in random.sample(random.choice(hard), 2):
                if s1 not in seen:
                    chosen.append(s1); seen.add(s1)
        while len(chosen) < args.clusters_per_batch:
            s1 = random.choice(keys)
            if s1 not in seen:
                chosen.append(s1); seen.add(s1)
        return chosen[:args.clusters_per_batch]

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        texts, labels = [], []
        for ci, s1 in enumerate(sample_batch()):
            members = clusters[s1]
            picked = [s1] + random.sample(members[1:], min(args.records_per_cluster - 1, len(members) - 1))
            for eid in picked:
                texts.append(text_of(eid, anchor=(eid == s1)))
                labels.append(ci)
        feats = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tok(texts).items()}
        labels = torch.tensor(labels, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            emb = model(feats)["sentence_embedding"]
        emb = F.normalize(emb.float(), dim=-1)
        loss = supcon_loss(emb, labels, args.temperature)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % args.log_every == 0 or step == 1:
            print(f"[train] step {step}/{args.steps} loss {loss.item():.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)

    model.save(args.out_dir)
    print(f"[train] saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
