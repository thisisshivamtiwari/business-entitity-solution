"""Step 2 - candidate generation ("blocking").

Comparing every S1 record with every S2/S3 record is impossible (2M x 10M), so we first
find a short list of plausible candidates per S1 record.

How it works
------------
Every record emits *compound keys*: pairs of tokens that are individually common but rare
together. Two records that share a rare key are probably the same business.

    type  meaning                          example
    nn    two name tokens                  "maure|colombier"
    na    name token + address token       "payne|fremont"
    nr    name token + region              "twisted|@il"
    da    address number + address token   "3315|fremont"
    aa    two adjacent address tokens      "lakshmi|ground"
    c     whole name without spaces        "maurewilliamscolombier" (matches websites)

Single tokens are useless on their own ("Global", "Center", "Street" appear in 100k+
records), which is why every key combines two pieces of evidence.

A key is only used when it is rare on both sides (at most CAP S1 records and CAP23 S2/S3
records in that country); otherwise it would link thousands of unrelated records. Each
candidate pair is scored by summing 1/n1 over the keys it shares (rarer key = stronger
evidence), with separate sums per key type so rerank.py can weigh them differently.

Keys are only compared within the same country (no true match crosses countries).

Output: work/cand_raw/{split}.parquet with the TOPK best raw candidates per S1:
    r1, r2        rids of the S1 record and the S2/S3 candidate
    bscore        sum of 1/n1 over shared keys
    nkeys, minn1  number of shared keys, rarity of the rarest one
    s_<type>      bscore restricted to one key type
    brank         0-based rank of the candidate within its S1 by bscore
"""
import os
import sys
import time

import polars as pl

from config import WORK_DIR, norm_path
from memguard import start_memory_guard

# Caps can be overridden for experiments: BER_CAP=20 python block.py ...
CAP = int(os.environ.get("BER_CAP", 10))       # a key may be carried by at most this many S1 records ...
CAP23 = int(os.environ.get("BER_CAP23", 100))  # ... and at most this many S2/S3 records (per country)
TOPK = 100     # raw candidates kept per S1; rerank.py narrows these down
S1_JOIN_CHUNK = 200_000   # S1 records per join chunk: bounds memory whatever the caps
KEY_TYPES = ["nn", "na", "nr", "da", "aa", "c"]


def _tokens(col, alpha_only=False):
    """Split a space-separated column into a list of non-empty tokens."""
    e = pl.element()
    keep = e != ""
    if alpha_only:
        keep = keep & ~e.str.contains(r"^\d+$")
    return pl.col(col).str.split(" ").list.eval(e.filter(keep))


def _key_expr(key_type, a, b):
    """Hash "type:a|b" to a u64 key; the type prefix keeps key families apart."""
    return pl.concat_str(pl.lit(f"{key_type}:"), a, pl.lit("|"), b).hash()


def _cross_keys(d, left, right, key_type, unordered=False):
    """Keys for every (left token, right token) combination of each record."""
    x = (
        d.select("rid", "src", a=left, b=right)
        .explode("a", empty_as_null=True)
        .explode("b", empty_as_null=True)
        .drop_nulls()
    )
    if unordered:  # token pairs from the same list: keep each pair once, in sorted order
        x = x.filter(pl.col("a") < pl.col("b"))
    return x.select("rid", "src", k=_key_expr(key_type, "a", "b"),
                    typ=pl.lit(KEY_TYPES.index(key_type), pl.UInt8))


def build_keys(d):
    """All blocking keys of the records in `d` -> (rid, src, k, typ), one row per key."""
    d = d.with_columns(
        name_toks=_tokens("core"),
        addr_words=_tokens("addr", alpha_only=True),
        addr_toks=_tokens("addr"),
        num_toks=_tokens("nums"),
        region_toks=_tokens("region"),
    )
    adjacent = (
        d.select(
            "rid", "src",
            a=pl.col("addr_toks").list.slice(0, pl.col("addr_toks").list.len() - 1),
            b=pl.col("addr_toks").list.slice(1),
        )
        .explode("a", "b", empty_as_null=True).drop_nulls()
        .select("rid", "src", k=_key_expr("aa", "a", "b"),
                typ=pl.lit(KEY_TYPES.index("aa"), pl.UInt8))
    )
    whole_name = d.filter(pl.col("concat") != "").select(
        "rid", "src", k=("c:" + pl.col("concat")).hash(),
        typ=pl.lit(KEY_TYPES.index("c"), pl.UInt8),
    )
    parts = [
        _cross_keys(d, pl.col("name_toks"), pl.col("name_toks"), "nn", unordered=True),
        _cross_keys(d, pl.col("name_toks"), pl.col("addr_words"), "na"),
        _cross_keys(d, pl.col("name_toks"), pl.col("region_toks"), "nr"),
        _cross_keys(d, pl.col("num_toks"), pl.col("addr_words"), "da"),
        adjacent,
        whole_name,
    ]
    return pl.concat(parts).unique(["rid", "k"])


KEY_CHUNK = 1_000_000  # records per key-building chunk (keys depend on one record only)


def block_country(d, topk=TOPK):
    """Candidate pairs for the records of one country."""
    # Building keys chunk by chunk gives the same keys but keeps only one chunk's
    # intermediate token strings in memory (India: 31.8 GB -> ~10 GB peak).
    keys = pl.concat([build_keys(d.slice(i, KEY_CHUNK)) for i in range(0, d.height, KEY_CHUNK)])

    # Keep only keys that are rare among S1 records ...
    k1 = keys.filter(pl.col("src") == 1).drop("src")
    n1 = k1.group_by("k").len("n1").filter(pl.col("n1") <= CAP)
    # ... and rare among S2/S3 records.
    k23 = keys.filter(pl.col("src") > 1).drop("src").join(n1.select("k"), on="k", how="semi")
    n23 = k23.group_by("k").len("n23").filter(pl.col("n23") <= CAP23)
    n1 = n1.join(n23.select("k"), on="k", how="semi")
    k1 = k1.join(n1, on="k")
    k23 = k23.join(n1.select("k"), on="k", how="semi")

    del keys
    k23 = k23.rename({"rid": "r2"})

    # Join S1 keys with S2/S3 keys and aggregate the evidence per pair, one chunk of S1 records
    # at a time. A chunk holds complete S1 records, so the result is identical to one big join.
    weight = 1.0 / pl.col("n1")
    r1s = k1["rid"].unique().sort()
    out = []
    for i in range(0, len(r1s), S1_JOIN_CHUNK):
        chunk = k1.join(pl.DataFrame({"rid": r1s[i:i + S1_JOIN_CHUNK]}), on="rid", how="semi")
        pairs = (
            chunk.rename({"rid": "r1"})
            .join(k23, on=["k", "typ"])
            .group_by("r1", "r2")
            .agg(
                bscore=weight.sum().cast(pl.Float32),
                nkeys=pl.len().cast(pl.UInt16),
                minn1=pl.col("n1").min().cast(pl.UInt16),
                **{f"s_{t}": weight.filter(pl.col("typ") == i).sum().cast(pl.Float32)
                   for i, t in enumerate(KEY_TYPES)},
            )
        )
        # Keep the TOPK best per S1 (ties broken by r2 so reruns are identical).
        out.append(
            pairs.sort(["bscore", "r2"], descending=[True, False])
            .with_columns(brank=pl.int_range(pl.len(), dtype=pl.UInt32).over("r1"))
            .filter(pl.col("brank") < topk)
        )
    return pl.concat(out)


def main(split):
    t0 = time.time()
    print(f"  caps: S1 <= {CAP}, S2/S3 <= {CAP23}", flush=True)
    norm = pl.read_parquet(
        norm_path(split), columns=["rid", "src", "country", "core", "concat", "addr", "nums", "region"])

    per_country = []
    for country in norm["country"].unique().sort():  # every country label, incl. unseen ones
        cand = block_country(norm.filter(pl.col("country") == country))
        print(f"  {country}: {cand.height:,} pairs for {cand['r1'].n_unique():,} S1 records "
              f"({time.time() - t0:.0f}s)", flush=True)
        per_country.append(cand)

    cand = pl.concat(per_country)
    path = WORK_DIR / "cand_raw" / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    cand.write_parquet(path)
    print(f"{split}: {cand.height:,} raw candidate pairs -> {path}")


if __name__ == "__main__":
    start_memory_guard("block")
    for split in sys.argv[1:] or ["train", "test"]:
        main(split)
