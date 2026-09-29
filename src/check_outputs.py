"""Check output/matching_results.tsv and output/candidate_pairs.tsv against the test records.

One row per test S1 in source order, only S2-/S3- ids, no duplicates inside a list, every id exists
in the test set, every match is among the candidates. Streams both files, so memory stays low.

    python check_outputs.py
"""
import hashlib
import sys

import numpy as np
import polars as pl

from config import DATA_DIR, OUT_DIR

OUT, D = str(OUT_DIR), str(DATA_DIR / "test")
def code(ids):  # "S2-123" -> 2*10^10 + 123
    return np.fromiter(((2 if x[1] == "2" else 3) * 10**10 + int(x[3:]) for x in ids), dtype=np.int64, count=len(ids))
valid = np.sort(np.concatenate([code(pl.read_csv(f"{D}/test_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False, columns=["entity_id"])["entity_id"].to_list()) for s in (2, 3)]))
s1 = pl.read_csv(f"{D}/test_source1.tsv", separator="\t", quote_char=None, infer_schema=False, columns=["entity_id"])["entity_id"].to_list()
print(f"test S1 {len(s1):,} | valid S2/S3 ids {len(valid):,}", flush=True)
err = {k: 0 for k in ("order", "dup", "prefix", "unknown", "not_in_candidates", "malformed")}
n = m_links = c_links = m_empty = c_empty = 0
buf = []
def flush():
    global buf
    if buf:
        a = code(buf); pos = np.searchsorted(valid, a); pos[pos == len(valid)] = 0
        err["unknown"] += int((valid[pos] != a).sum()); buf = []
with open(f"{OUT}/matching_results.tsv", encoding="utf-8") as fm, open(f"{OUT}/candidate_pairs.tsv", encoding="utf-8") as fc:
    hm, hc = fm.readline().rstrip("\n"), fc.readline().rstrip("\n")
    assert hm == "source1_entity_id\tmatched_entity_ids" and hc == "source1_entity_id\tcandidate_entity_ids", (hm, hc)
    for i, (lm, lc) in enumerate(zip(fm, fc)):
        a, t1, rm = lm.rstrip("\n").partition("\t"); b, t2, rc = lc.rstrip("\n").partition("\t")
        if not t1 or not t2: err["malformed"] += 1
        if a != s1[i] or b != s1[i]: err["order"] += 1
        M = rm.split(",") if rm else []; C = rc.split(",") if rc else []
        sm, scs = set(M), set(C)
        if len(sm) != len(M) or len(scs) != len(C): err["dup"] += 1
        if any(x[:3] not in ("S2-", "S3-") for x in C) or any(x[:3] not in ("S2-", "S3-") for x in M): err["prefix"] += 1; continue
        if sm - scs: err["not_in_candidates"] += 1
        m_links += len(M); c_links += len(C); m_empty += not M; c_empty += not C; n += 1
        buf.extend(C)                       # matched ids are a subset of candidates, so checking candidates covers both
        if len(buf) > 2_000_000: flush()
    flush()
    assert fm.readline() == "" and fc.readline() == "", "files have different lengths"
print(f"rows {n:,} (expected {len(s1):,}) | matches {m_links:,}, empty rows {m_empty:,} | candidates {c_links:,} ({c_links/n:.1f} per S1), empty rows {c_empty:,}")
print("problems:", err)
print("md5 matching_results.tsv:", hashlib.md5(open(f"{OUT}/matching_results.tsv", "rb").read()).hexdigest())
ok = n == len(s1) and not any(err.values())
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
