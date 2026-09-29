"""Iteration 9c: name-only candidate search for S2/S3 records with NO address.

Why (eda/NoAddress.ipynb): 97.7% of no-address records belong to some S1, but key-based blocking
needs an address or an exact name word pair, so a no-address record with a typo or a dropped/added
word ("METROPOLITAN HEDOTEL", "Woods Inc Services") never becomes a candidate. That's 1,725 unique-name
true pairs per 150k validation S1.

How: for every no-address S2/S3 record, find the TOP_K S1 records in the same country whose names are
most similar by character-trigram TF-IDF cosine (sparse_dot_topn, multi-threaded, memory-bounded).
Trigrams that occur in more than MAX_DF of the country's S1 names are ignored (too common to help).
Pairs go to work/cand_name/{split}.parquet (r1, r2, name_sim); train.load_candidates adds them to
the ranker's top-12, where the matcher decides.

Usage:  python namesearch.py train test [trainlike]
"""
import sys
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from config import WORK_DIR, norm_path
from memguard import start_memory_guard

TOP_K = 5
MIN_SIM = 0.5          # cosine below this is not a plausible name match
MAX_DF = 0.002         # ignore trigrams present in > 0.2% of a country's S1 names
QUERY_CHUNK = 50_000


def search_country(d):
    s1 = d.filter((pl.col("src") == 1) & (pl.col("core") != ""))
    q = d.filter((pl.col("src") > 1) & (pl.col("f_addr_empty") == 1) & (pl.col("core") != ""))
    if s1.height == 0 or q.height == 0:
        return pl.DataFrame(schema={"r1": pl.UInt32, "r2": pl.UInt32, "name_sim": pl.Float32})
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_df=MAX_DF, min_df=2,
                          dtype=np.float32, sublinear_tf=True)
    A = vec.fit_transform(s1["core"].to_list())             # S1 names (fit IDF on the S1 side)
    At = A.T.tocsr()
    r1_ids = s1["rid"].to_numpy()
    out = []
    for i in range(0, q.height, QUERY_CHUNK):
        part = q.slice(i, QUERY_CHUNK)
        B = vec.transform(part["core"].to_list())
        C = sp_matmul_topn(B, At, top_n=TOP_K, threshold=MIN_SIM, n_threads=10).tocoo()
        out.append(pl.DataFrame({"r1": r1_ids[C.col].astype(np.uint32),
                                 "r2": part["rid"].to_numpy()[C.row].astype(np.uint32),
                                 "name_sim": C.data.astype(np.float32)}))
    return pl.concat(out)


def main(split):
    t0 = time.time()
    norm = pl.read_parquet(norm_path(split), columns=["rid", "src", "country", "core", "f_addr_empty"])
    found = []
    for country in norm["country"].unique().sort():
        pairs = search_country(norm.filter(pl.col("country") == country))
        print(f"  {country}: {pairs.height:,} name-only pairs for {pairs['r2'].n_unique():,} no-address records "
              f"({time.time() - t0:.0f}s)", flush=True)
        found.append(pairs)
    out = pl.concat(found)
    path = WORK_DIR / "cand_name" / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    print(f"{split}: {out.height:,} name-only candidate pairs -> {path}")


if __name__ == "__main__":
    start_memory_guard("namesearch")
    for s in sys.argv[1:] or ["train", "test"]:
        main(s)
