"""Score a candidate_pairs.tsv against the training ground truth.

Reports the recall ceiling (pair-level and macro per S1 entity), recall@10/20/50
(candidate lists are written in rank order), reduction ratio, list-size distribution,
recall by country / source / missing address / Indic-script name, candidate load on
true singletons, and a sample of missed pairs.

Usage:
  python evaluate_blocking.py --candidates work/val_candidates.tsv \
      --ground-truth dataset/train/train_ground_truth.tsv \
      --records work/train_records.parquet        # or: --data-dir dataset
Only S1 entities present in the candidate file are scored.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import load_records, read_tsv  # noqa: E402

CUTOFFS = (10, 20, 25, 40, 50)


def read_id_lists(path, col):
    df = read_tsv(path)
    return {s1: [x for x in lst.split(",") if x] for s1, lst in zip(df.source1_entity_id, df[col])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--records", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--show-misses", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    cands = read_id_lists(args.candidates, "candidate_entity_ids")
    truth = read_id_lists(args.ground_truth, "matched_entity_ids")
    truth = {s1: v for s1, v in truth.items() if s1 in cands}
    if not truth:
        raise SystemExit("no overlap between candidate file and ground truth S1 ids")

    cols = ["entity_id", "source", "country", "addr_missing", "indic_name", "business_name", "business_address"]
    rec = pd.read_parquet(args.records, columns=cols) if args.records else load_records(args.data_dir, "train")[cols]
    rec = rec.set_index("entity_id")
    country, addr_missing, indic = rec.country, rec.addr_missing, rec.indic_name

    fmt = Counter()
    for lst in cands.values():
        if len(lst) != len(set(lst)): fmt["duplicate ids in a list"] += 1
        if any(x.startswith("S1-") for x in lst): fmt["S1 ids in a list"] += 1

    n_links = n_found = fully = 0
    per_entity = []
    at = Counter()
    grp = {}                                   # name -> [found, total]
    def bump(name, ok):
        g = grp.setdefault(name, [0, 0]); g[0] += ok; g[1] += 1
    misses = []
    for s1, tm in truth.items():
        if not tm:
            continue
        lst = cands[s1]
        rank = {x: i for i, x in enumerate(lst)}
        hit = [m for m in tm if m in rank]
        n_links += len(tm); n_found += len(hit)
        per_entity.append(len(hit) / len(tm))
        fully += len(hit) == len(tm)
        for m in tm:
            ok = m in rank
            for c in CUTOFFS:
                at[c] += ok and rank[m] < c
            bump(f"country={country.get(s1, '?')}", ok)
            bump(f"source={m[:2]}", ok)
            bump("match address missing" if addr_missing.get(m, False) else "match address present", ok)
            bump("Indic-script name (either side)" if (indic.get(m, False) or indic.get(s1, False))
                 else "Latin-script names", ok)
            if not ok:
                misses.append((s1, m))

    sizes = np.array([len(v) for v in cands.values()])
    singles = [s1 for s1, tm in truth.items() if not tm]
    single_sizes = np.array([len(cands[s1]) for s1 in singles]) if singles else np.array([0])
    s1_country = Counter(country.get(s, "?") for s in cands)
    db_country = Counter(country[rec.source != 1])
    full_pairs = sum(n * db_country.get(c, 0) for c, n in s1_country.items())
    total_c = int(sizes.sum())

    print("=" * 72)
    print(f"Candidate file : {args.candidates}")
    print(f"S1 scored      : {len(cands):,}  (with true matches: {len(per_entity):,}, singletons: {len(singles):,})")
    for k, v in fmt.items():
        print(f"FORMAT ISSUE   : {k}: {v:,} rows")
    print("-" * 72)
    print(f"Pair recall (recall ceiling)      : {n_found/n_links:.4f}   ({n_found:,}/{n_links:,} links)")
    print("Pair recall @ rank cutoff         : " + "  ".join(f"@{c}={at[c]/n_links:.4f}" for c in CUTOFFS))
    print(f"Macro per-entity recall           : {np.mean(per_entity):.4f}")
    print(f"Entities fully covered            : {fully/len(per_entity):.4f}")
    print(f"Entities with zero candidates     : {(sizes==0).mean():.4f}")
    print("-" * 72)
    print(f"Total candidates                  : {total_c:,}")
    print(f"Candidates per S1 mean/median/p95/max : {sizes.mean():.1f} / {np.median(sizes):.0f} / "
          f"{np.percentile(sizes,95):.0f} / {sizes.max()}")
    print(f"  on true singletons              : {single_sizes.mean():.1f}")
    print(f"Reduction ratio vs country cross-product : {1 - total_c/max(1,full_pairs):.6f}")
    print("-" * 72)
    for name, (f, t) in sorted(grp.items()):
        print(f"  {name:36s} recall {f/t:.4f}  (n={t:,})")
    print("-" * 72)
    if misses and args.show_misses:
        print(f"Sample of {min(args.show_misses, len(misses))} missed links (S1 -> missed match):")
        for s1, m in random.sample(misses, min(args.show_misses, len(misses))):
            a, b = rec.loc[s1], rec.loc[m]
            print(f"  {s1}: {a.business_name} | {a.business_address}")
            print(f"  {m}: {b.business_name} | {b.business_address}")
    print("=" * 72)


if __name__ == "__main__":
    main()
