"""Matching stage 3: the judge. A multilingual cross-encoder reads both records in one
sequence  ("name | address"  [SEP]  "name | address")  and outputs p(same business).

  train : fine-tune on labelled pairs (BCE) with Ditto-style augmentation
  score : add a ce_score column to a pairs parquet

Backbone: FacebookAI/xlm-roberta-base (278M parameters, MIT licence), bf16 autocast on CUDA.
Fold 0 never enters training: the pairs file for training is built from folds 1-4
candidates only.
"""
import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blocking"))
from common import pick_device  # noqa: E402


# ----------------------------------------------------------------------------- data
def load_texts(work, split, pairs):
    rec = pd.read_parquet(work / f"{split}_records.parquet",
                          columns=["entity_id", "name_clean", "addr_clean"]).set_index("entity_id")
    txt = (rec.name_clean + " | " + rec.addr_clean).str.rstrip(" |")
    return txt.loc[pairs.s1_id].to_numpy(), txt.loc[pairs.cand_id].to_numpy()


def augment(a, b, p, addr_drop_p=0.0):
    """Ditto-style perturbations. First, with probability addr_drop_p (label-dependent, see
    train), the whole address of one side is removed (name-only record). Then, with
    probability p, one of five random perturbations: drop a token, drop an address, swap
    sides, change case, shuffle address parts. Labels are unchanged by any of them."""
    if addr_drop_p and random.random() < addr_drop_p:
        side = random.randrange(2)
        if side == 0 and " | " in a: a = a.split(" | ")[0]
        elif " | " in b: b = b.split(" | ")[0]
        elif " | " in a: a = a.split(" | ")[0]
    if random.random() >= p:
        return a, b
    op = random.randrange(5)
    if op == 0:                                   # drop one random token from one side
        side = random.randrange(2); toks = (a if side == 0 else b).split()
        if len(toks) > 2:
            toks.pop(random.randrange(len(toks)))
            if side == 0: a = " ".join(toks)
            else: b = " ".join(toks)
    elif op == 1:                                 # drop the address of one side
        side = random.randrange(2)
        if side == 0 and " | " in a: a = a.split(" | ")[0]
        elif side == 1 and " | " in b: b = b.split(" | ")[0]
    elif op == 2:                                 # swap sides
        a, b = b, a
    elif op == 3:                                 # case change on one side
        side = random.randrange(2); f = random.choice((str.upper, str.lower, str.title))
        if side == 0: a = f(a)
        else: b = f(b)
    else:                                         # shuffle address parts of one side
        side = random.randrange(2); s = a if side == 0 else b
        if " | " in s:
            nm, ad = s.split(" | ", 1); parts = ad.split(", "); random.shuffle(parts)
            s = nm + " | " + ", ".join(parts)
            if side == 0: a = s
            else: b = s
    return a, b


def encode(tok, a, b, max_len, device):
    enc = tok(list(a), list(b), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def ddp_setup():
    """torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; single-process runs see world=1."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        torch.distributed.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return world, int(os.environ["RANK"]), f"cuda:{local}"
    return 1, 0, pick_device()


def amp_dtype(device, mode="auto"):
    """auto = bf16 on CUDA when supported; fp32 = no autocast (TF32 matmul kept on)."""
    if not device.startswith("cuda") or mode == "fp32":
        return None
    if mode == "fp16":
        return torch.float16
    if mode == "bf16" or torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ---------------------------------------------------------------------------- train
def train(args):
    world, rank, device = ddp_setup()
    work = Path(args.work_dir)
    pairs = pd.read_parquet(args.pairs)
    if "fold" in pairs.columns:
        assert (pairs.fold != 0).all(), "fold-0 pairs found in training data"
    if "group" in pairs.columns and args.group:
        pairs = pairs[pairs.group == args.group]
    pos, neg = pairs[pairs.label == 1], pairs[pairs.label == 0]
    if args.neg_per_pos and len(neg) > args.neg_per_pos * len(pos):
        neg = neg.sample(n=int(args.neg_per_pos * len(pos)), random_state=args.seed)
    pairs = pd.concat([pos, neg]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if args.max_pairs and len(pairs) > args.max_pairs:
        pairs = pairs.iloc[:args.max_pairs]
    a, b = load_texts(work, "train", pairs)
    y = pairs.label.to_numpy().astype(np.float32)
    # label-aware address dropping: raise the share of name-only positives to the target
    both_addr = np.array([(" | " in x) and (" | " in z) for x, z in zip(a, b)])
    nat_pos = 1 - both_addr[y == 1].mean() if (y == 1).any() else 0.0
    nat_neg = 1 - both_addr[y == 0].mean() if (y == 0).any() else 0.0
    drop_pos = max(0.0, (args.addr_drop_pos - nat_pos) / max(1e-9, 1 - nat_pos))
    drop_neg = max(0.0, (args.addr_drop_neg - nat_neg) / max(1e-9, 1 - nat_neg))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dtype = amp_dtype(device, args.amp)
    if rank == 0:
        print(f"[ce-train] world={world} device={device} amp={dtype} pairs={len(pairs):,} positives={y.mean():.3%} "
              f"base={args.base_model} epochs={args.epochs} augment_p={args.augment_p}", flush=True)
        print(f"[ce-train] name-only side: natural share positives {nat_pos:.1%} / negatives {nat_neg:.1%}; "
              f"targets {args.addr_drop_pos:.0%} / {args.addr_drop_neg:.0%} -> extra drop prob {drop_pos:.2f} / {drop_neg:.2f}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1).to(device)
    if args.freeze_embeddings:            # 192M of the 278M weights are the token table; freezing it cuts the per-step sync
        for p_ in model.base_model.embeddings.word_embeddings.parameters():
            p_.requires_grad_(False)
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    if rank == 0:
        print(f"[ce-train] trainable weights {n_train/1e6:.0f}M, batch per GPU {args.batch_size}, total batch {args.batch_size*world}, lr {args.lr}", flush=True)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(device.split(":")[1])])
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=args.lr, weight_decay=0.01)
    per_epoch = math.ceil(len(pairs) / (args.batch_size * world))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    if args.time_budget_min > 0:
        # Speed probe: 30 real steps with learning rate 0 (weights do not move), timed after warm-up.
        # If the requested epochs do not fit the budget, the epoch count is lowered before training starts.
        for g in opt.param_groups:
            g["lr"] = 0.0
        probe = np.random.RandomState(args.seed + 999).permutation(len(pairs))[rank::world]
        model.train()
        for i in range(30):
            if i == 5:
                torch.cuda.synchronize() if device.startswith("cuda") else None
                tp = time.time()
            idx = np.take(probe, np.arange(i * args.batch_size, (i + 1) * args.batch_size), mode="wrap")
            enc = encode(tok, a[idx], b[idx], args.max_len, device)
            with torch.autocast(device_type="cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
                lg = model(**enc).logits.squeeze(-1)
            ls = loss_fn(lg.float(), torch.from_numpy(y[idx]).to(device))
            opt.zero_grad(set_to_none=True); ls.backward(); opt.step()
        torch.cuda.synchronize() if device.startswith("cuda") else None
        dt = (time.time() - tp) / 25
        fit = max(1, min(args.epochs, int(args.time_budget_min * 60 / (dt * per_epoch))))
        ep_t = torch.tensor([fit], device=device)
        if world > 1:
            torch.distributed.broadcast(ep_t, src=0)                     # every rank follows rank 0
        if rank == 0:
            print(f"[ce-train] speed probe: {dt:.3f} s/step, {per_epoch:,} steps per epoch = {dt*per_epoch/60:.1f} min per epoch; "
                  f"budget {args.time_budget_min:.0f} min -> training {int(ep_t.item())} epoch(s) (requested {args.epochs})", flush=True)
        args.epochs = int(ep_t.item())
        if args.probe_only:
            if world > 1:
                torch.distributed.destroy_process_group()
            return
        opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=args.lr, weight_decay=0.01)   # fresh optimizer state
    steps = per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05,
                                                anneal_strategy="linear")
    use_scaler = dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    model.train(); t0 = time.time(); step = 0; skipped = 0
    for ep in range(args.epochs):
        order = np.random.RandomState(args.seed + ep).permutation(len(pairs))[rank::world]   # each rank: its slice
        for s in range(0, per_epoch * args.batch_size, args.batch_size):
            idx = order[s:s + args.batch_size]
            if len(idx) == 0:
                idx = order[:args.batch_size]                                            # keep ranks in step
            ab = [augment(a[i], b[i], args.augment_p, drop_pos if y[i] == 1 else drop_neg) for i in idx]
            enc = encode(tok, [x for x, _ in ab], [z for _, z in ab], args.max_len, device)
            target = torch.from_numpy(y[idx]).to(device)
            with torch.autocast(device_type="cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
                logits = model(**enc).logits.squeeze(-1)
            loss = loss_fn(logits.float(), target)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):                      # never let a NaN batch touch the weights
                skipped += 1; sched.step(); step += 1
                if skipped in (1, 10, 100) and rank == 0:
                    print(f"[ce-train] WARNING non-finite loss at step {step} (skipped {skipped} so far)", flush=True)
                continue
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if rank == 0 and (step % args.log_every == 0 or step == 1):
                el = time.time() - t0
                print(f"[ce-train] epoch {ep+1} step {step}/{steps} loss {loss.item():.4f} {el:.0f}s "
                      f"(left {el / step * (steps - step) / 60:.0f} min)", flush=True)
        if rank == 0:
            (model.module if world > 1 else model).save_pretrained(args.out_dir); tok.save_pretrained(args.out_dir)
            print(f"[ce-train] epoch {ep+1} done, checkpoint saved -> {args.out_dir}", flush=True)
        if world > 1:
            torch.distributed.barrier()
    if rank == 0:
        print(f"[ce-train] finished in {time.time()-t0:.0f}s, non-finite batches skipped: {skipped}")
    if world > 1:
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------- score
def score(args):
    device = pick_device()
    work = Path(args.work_dir)
    pairs = pd.read_parquet(args.pairs)
    if args.only_group and "group" in pairs.columns:
        pairs = pairs[pairs.group == args.only_group].reset_index(drop=True)
        print(f"[ce-score] scoring only group '{args.only_group}': {len(pairs):,} pairs")
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        lo, hi = i * len(pairs) // n, (i + 1) * len(pairs) // n
        pairs = pairs.iloc[lo:hi].reset_index(drop=True)
        print(f"[ce-score] shard {i}/{n}: rows {lo:,}-{hi:,}")
    a, b = load_texts(work, args.split, pairs)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = amp_dtype(device, args.amp)
    if dtype is not None:
        model.to(dtype)
    out = np.empty(len(pairs), dtype=np.float32)
    t0 = time.time()
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])     # length-sorted batches: less padding
    with torch.inference_mode():
        for n, s in enumerate(range(0, len(pairs), args.batch_size)):
            idx = order[s:s + args.batch_size]
            enc = encode(tok, a[idx], b[idx], args.max_len, device)
            out[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            if n % 200 == 0:
                print(f"[ce-score] {min(s + args.batch_size, len(pairs)):,}/{len(pairs):,} {time.time()-t0:.0f}s", flush=True)
    pairs["ce_score"] = out
    pairs.to_parquet(args.out, index=False)
    print(f"[ce-score] -> {args.out} ({time.time()-t0:.0f}s, {len(pairs)/max(1,time.time()-t0):.0f} pairs/s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--work-dir", required=True); t.add_argument("--pairs", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--base-model", default="FacebookAI/xlm-roberta-base")
    t.add_argument("--epochs", type=int, default=2); t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=2e-5); t.add_argument("--max-len", type=int, default=128)
    t.add_argument("--neg-per-pos", type=float, default=4.0, help="cap negatives at this multiple of positives (0 = all)")
    t.add_argument("--augment-p", type=float, default=0.3, help="share of training pairs given a random perturbation")
    t.add_argument("--addr-drop-pos", type=float, default=0.30, help="target share of TRUE pairs with a name-only side (natural ~4%%; the rest is created by dropping an address)")
    t.add_argument("--addr-drop-neg", type=float, default=0.0, help="target share of NEGATIVE pairs with a name-only side (0 = keep the natural ~33%%)")
    t.add_argument("--group", default="ce", help="if the pairs file has a 'group' column, train on this group")
    t.add_argument("--max-pairs", type=int, default=0); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--log-every", type=int, default=200)
    t.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    t.add_argument("--freeze-embeddings", type=int, default=0, help="1 = do not update the token embedding table")
    t.add_argument("--probe-only", type=int, default=0, help="1 = run the speed probe and exit (needs --time-budget-min > 0)")
    t.add_argument("--time-budget-min", type=float, default=0, help="if >0: time 30 steps first and lower --epochs so training fits this many minutes")
    s = sub.add_parser("score")
    s.add_argument("--work-dir", required=True); s.add_argument("--split", default="train")
    s.add_argument("--pairs", required=True); s.add_argument("--model-dir", required=True)
    s.add_argument("--out", required=True); s.add_argument("--batch-size", type=int, default=512)
    s.add_argument("--max-len", type=int, default=128)
    s.add_argument("--only-group", default=None)
    s.add_argument("--shard", default=None, help="i/n: score the i-th contiguous slice of n (one GPU each)")
    s.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    args = ap.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    (train if args.cmd == "train" else score)(args)


if __name__ == "__main__":
    main()
