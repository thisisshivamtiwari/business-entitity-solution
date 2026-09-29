"""Matching stage 2: pairwise features for every (S1, candidate) pair.

Name: fuzzy ratios, token and char-trigram Jaccard, exact name-key, length ratio,
      script mismatch, legal-suffix agreement.
Address: house number, postal code (IN 6-digit / US 5-digit / FR 5-digit), token Jaccard,
         fuzzy ratio, missing flags.
Blocking: cosine on both embedding views and the candidate rank.
Missing values stay NaN (LightGBM handles them natively); a flag says why.

Output: {pairs}_feat.parquet with the pair keys, label and every feature column.
"""
import argparse
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blocking"))
from common import has_indic_script  # noqa: E402

_NUM = re.compile(r"\d+")
_PIN = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_ZIP = re.compile(r"(?<!\d)(\d{5})(?!\d)")
_TOK = re.compile(r"[^\W_]+", re.UNICODE)
SUFFIXES = {"inc", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company",
            "llp", "plc", "sarl", "sas", "sa", "gmbh", "pc", "lp", "sci", "eurl"}


def toks(s):
    return set(t.lower() for t in _TOK.findall(s))


def tri(s):
    s = re.sub(r"\s+", "", s.lower())
    return {s[i:i + 3] for i in range(len(s) - 2)}


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else np.nan


def house_no(addr):
    m = _NUM.search(addr)
    return m.group(0).lstrip("0") if m else ""


def postal(addr):
    m = _PIN.search(addr) or _ZIP.search(addr)
    return m.group(1) if m else ""


def suffix_set(name_tokens):
    return name_tokens & SUFFIXES


def pair_features(args):
    n1, a1, n2, a2 = args
    t1, t2 = toks(n1), toks(n2)
    core1, core2 = t1 - SUFFIXES, t2 - SUFFIXES
    l1, l2 = n1.lower(), n2.lower()
    f = {
        "name_ratio": fuzz.ratio(l1, l2) / 100,
        "name_token_set": fuzz.token_set_ratio(l1, l2) / 100,
        "name_partial": fuzz.partial_ratio(l1, l2) / 100,
        "name_tok_jac": jac(t1, t2),
        "name_core_jac": jac(core1, core2),
        "name_tri_jac": jac(tri(n1), tri(n2)),
        "name_len_ratio": min(len(n1), len(n2)) / max(len(n1), len(n2), 1),
        "name_first_tok_eq": float(bool(t1) and bool(t2) and sorted(t1)[0] == sorted(t2)[0]),
        "suffix_both": float(bool(suffix_set(t1)) and bool(suffix_set(t2))),
        "suffix_eq": float(suffix_set(t1) == suffix_set(t2)) if (suffix_set(t1) or suffix_set(t2)) else np.nan,
        "script_mismatch": float(has_indic_script(n1) != has_indic_script(n2)),
        "addr1_missing": float(a1 == ""),
        "addr2_missing": float(a2 == ""),
    }
    if a1 and a2:
        h1, h2 = house_no(a1), house_no(a2)
        p1, p2 = postal(a1), postal(a2)
        at1, at2 = toks(a1), toks(a2)
        f.update({
            "addr_ratio": fuzz.ratio(a1.lower(), a2.lower()) / 100,
            "addr_token_set": fuzz.token_set_ratio(a1.lower(), a2.lower()) / 100,
            "addr_tok_jac": jac(at1, at2),
            "house_eq": float(h1 == h2) if (h1 and h2) else np.nan,
            "house_both": float(bool(h1 and h2)),
            "postal_eq": float(p1 == p2) if (p1 and p2) else np.nan,
            "postal_both": float(bool(p1 and p2)),
            "addr_num_jac": jac(set(_NUM.findall(a1)), set(_NUM.findall(a2))),
        })
    else:
        f.update({k: np.nan for k in ("addr_ratio", "addr_token_set", "addr_tok_jac", "house_eq",
                                      "house_both", "postal_eq", "postal_both", "addr_num_jac")})
    return f


def cosine_cols(pairs, rec_index, work, split):
    out = {}
    a = rec_index.loc[pairs.s1_id].to_numpy()
    b = rec_index.loc[pairs.cand_id].to_numpy()
    for view in ("full_text", "name_text"):
        emb = np.load(work / f"{split}_{view}.npy", mmap_mode="r")
        sims = np.empty(len(pairs), dtype=np.float32)
        for s in range(0, len(pairs), 500_000):
            ea = emb[a[s:s + 500_000]].astype(np.float32)
            eb = emb[b[s:s + 500_000]].astype(np.float32)
            sims[s:s + 500_000] = (ea * eb).sum(1)
        out[f"cos_{view}"] = sims
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    args = ap.parse_args()
    work = Path(args.work_dir)

    pairs = pd.read_parquet(args.pairs)
    rec = pd.read_parquet(work / f"{args.split}_records.parquet",
                          columns=["entity_id", "name_clean", "addr_clean"]).set_index("entity_id")
    rec_index = pd.Series(np.arange(len(rec)), index=rec.index)
    print(f"[features] {len(pairs):,} pairs", flush=True)

    n1 = rec.name_clean.loc[pairs.s1_id].to_numpy(); a1 = rec.addr_clean.loc[pairs.s1_id].to_numpy()
    n2 = rec.name_clean.loc[pairs.cand_id].to_numpy(); a2 = rec.addr_clean.loc[pairs.cand_id].to_numpy()
    parts = []                                   # chunked: 56M pairs would not fit as one list of dicts
    with Pool(args.workers) as pool:
        for s in range(0, len(pairs), args.chunk):
            e = min(s + args.chunk, len(pairs))
            feats = pool.map(pair_features, zip(n1[s:e], a1[s:e], n2[s:e], a2[s:e]), chunksize=20_000)
            parts.append(pd.DataFrame(feats).astype(np.float32)); del feats
            print(f"[features] {e:,}/{len(pairs):,}", flush=True)
    del n1, a1, n2, a2
    df = pd.concat(parts, ignore_index=True); del parts
    for k, v in cosine_cols(pairs, rec_index, work, args.split).items():
        df[k] = v
    df["cos_gap_to_best"] = df.groupby(pairs.s1_id.to_numpy()).cos_full_text.transform("max") - df.cos_full_text
    df["n_cands"] = pairs.groupby("s1_id").cand_id.transform("size").to_numpy()
    out = pd.concat([pairs.reset_index(drop=True), df], axis=1)
    out.to_parquet(args.out, index=False)
    print(f"[features] {out.shape[1] - pairs.shape[1]} features -> {args.out}")


if __name__ == "__main__":
    main()
