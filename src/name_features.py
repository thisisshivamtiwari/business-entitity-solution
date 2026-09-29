"""Iteration 9b features: typo tolerance, templated names, and same-name branches.

Why (eda/Errors_iter2.ipynb, eda/NoAddress.ipynb, the France diagnosis):
  * char_tri_*: character 3-gram overlap tolerates typos *inside* words ("HEDOTEL" vs "HOTEL"),
    which word-level features miss. This is the guidelines' "TF-IDF / Jaccard" idea at character level.
  * name_nocity_tset: French (and Indian) names often contain the city ("Bordeaux Federation" vs
    "Bordeaux Amicale"); the shared city inflates name similarity, so compare names without the words
    that also appear in the record's own address.
  * max_idf_only1/2: the rarest word that only ONE side has, the "type word" that separates
    near-duplicate decoys (club vs école, clinique vs pharmacie).
  * twins_legal_*: when several S1 records share a name (branches), the record's legal form picks the
    right branch 86.5% of the time when exactly one branch matches it (above F0.5's ~75% break-even).
"""
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz

TRI_COLS = ["char_tri_jaccard", "char_tri_cover2", "concat_tri_jaccard", "name_nocity_tset"]


def _trigrams(s):
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _block(args):
    core1, core2, cc1, cc2, addr1, addr2 = args
    out = np.zeros((len(core1), len(TRI_COLS)), dtype=np.float32)
    for i in range(len(core1)):
        a, b = _trigrams(core1[i]), _trigrams(core2[i])
        inter = len(a & b)
        out[i, 0] = inter / max(len(a | b), 1)
        out[i, 1] = inter / max(len(b), 1)                       # share of the candidate's trigrams found in S1
        ca, cb = _trigrams(cc1[i]), _trigrams(cc2[i])
        out[i, 2] = len(ca & cb) / max(len(ca | cb), 1)
        # drop words that also occur in the record's own address (usually the city), then compare
        w1 = [t for t in core1[i].split() if t not in set(addr1[i].split())]
        w2 = [t for t in core2[i].split() if t not in set(addr2[i].split())]
        out[i, 3] = fuzz.token_set_ratio(" ".join(w1), " ".join(w2)) if w1 and w2 else -1
    return out


def trigram_features(df, procs=12, block=100_000):
    cols = ["core_1", "core_2", "concat_1", "concat_2", "addr_1", "addr_2"]
    data = [df[c].to_list() for c in cols]
    jobs = [[col[i:i + block] for col in data] for i in range(0, df.height, block)]
    with Pool(procs) as pool:
        blocks = pool.map(_block, jobs, chunksize=1)
    m = np.vstack(blocks) if blocks else np.zeros((0, len(TRI_COLS)), np.float32)
    return pl.DataFrame(m, schema=TRI_COLS)


def unshared_word_features(df, idf):
    """Rarest word present on only one side (IDF), per pair."""
    def words(side):
        return (df.select("pid", country="country_1", t=pl.col(f"core_{side}").str.split(" "))
                .explode("t", empty_as_null=True).drop_nulls().filter(pl.col("t") != "").unique(["pid", "t"]))
    a, b = words(1), words(2)
    only1 = a.join(b, on=["pid", "t"], how="anti").join(idf, on=["country", "t"], how="left")
    only2 = b.join(a, on=["pid", "t"], how="anti").join(idf, on=["country", "t"], how="left")
    fill = idf["idf"].max()
    m1 = only1.group_by("pid").agg(max_idf_only1=pl.col("idf").fill_null(fill).max())
    m2 = only2.group_by("pid").agg(max_idf_only2=pl.col("idf").fill_null(fill).max())
    return (df.select("pid").join(m1, on="pid", how="left").join(m2, on="pid", how="left").sort("pid")
            .select(pl.col("max_idf_only1", "max_idf_only2").fill_null(0).cast(pl.Float32)))


_TWIN_CACHE = {}


def twin_legal_features(df, norm):
    """Among S1 records sharing this S1's exact name: how many have the record's legal form?"""
    key = id(norm)
    if key not in _TWIN_CACHE:
        _TWIN_CACHE.clear()
        _TWIN_CACHE[key] = (norm.filter((pl.col("src") == 1) & (pl.col("core") != ""))
                            .group_by("country", "core", "legal").len("n_twins_same_legal"))
    counts = _TWIN_CACHE[key]
    x = (df.select("pid", country="country_1", core="core_1", legal="legal_2", legal_1="legal_1")
         .join(counts, on=["country", "core", "legal"], how="left").sort("pid")
         .with_columns(pl.col("n_twins_same_legal").fill_null(0)))
    return x.select(
        twins_same_legal=pl.col("n_twins_same_legal"),
        # this S1 is the ONLY same-name S1 whose legal form equals the record's
        only_twin_with_legal=((pl.col("legal_1") == pl.col("legal")) & (pl.col("n_twins_same_legal") == 1)),
    ).cast(pl.Float32)


def all_features(df, norm, idf_core):
    return pl.concat([trigram_features(df), unshared_word_features(df, idf_core),
                      twin_legal_features(df, norm)], how="horizontal")
