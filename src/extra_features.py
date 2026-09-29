"""Name-uniqueness and decoy-detection features (item 3d / 3e of the improvement plan).

Why (from eda/Errors_iter2.ipynb and eda/Funnel.ipynb):
  * Recall falls from 0.969 for S1 names that are unique to 0.865 for names shared by > 50 S1
    records, and empty-address pairs are only resolvable when the name is (nearly) unique. The model
    had no direct signal for "how many other businesses have this exact name".
  * False matches are mostly near-duplicate decoys: similar name, house number off by a few, a
    different legal form, extra words, and longer names (decoy names average 28.6 vs 24.0 chars).

Counts are computed over the records of the world being scored (its own S1 and S2/S3), so the
same code works for train, the test-like world and test.
"""
import polars as pl

_CACHE = {}


def record_counts(norm):
    """For every record (by rid): how many S1 / S2-S3 records in its country share its exact core
    name (0 for an empty name). Computed once per table and cached."""
    key = id(norm)
    if key not in _CACHE:
        named = norm.filter(pl.col("core") != "")
        s1 = named.filter(pl.col("src") == 1).group_by("country", "core").len("n_s1")
        s23 = named.filter(pl.col("src") > 1).group_by("country", "core").len("n_s23")
        per = (norm.select("rid", "country", "core")
               .join(s1, on=["country", "core"], how="left").join(s23, on=["country", "core"], how="left")
               .sort("rid").with_columns(pl.col("n_s1", "n_s23").fill_null(0)))
        _CACHE.clear()
        _CACHE[key] = (per["n_s1"].to_numpy(), per["n_s23"].to_numpy())
    return _CACHE[key]


def uniqueness_features(df, norm):
    """df: pairs with r1 / r2 (rid == row number of norm). Returns new columns, row-aligned."""
    n_s1, n_s23 = record_counts(norm)
    r1, r2 = df["r1"].to_numpy(), df["r2"].to_numpy()
    return pl.DataFrame({
        # how many S1 records share this S1's exact name (1 = unique)
        "s1_same_name": n_s1[r1],
        # how many S2/S3 records carry that name (popular / generic names)
        "s23_same_name": n_s23[r1],
        # how many S1 records the candidate's name matches exactly (0 = matches no S1 name exactly)
        "s1_share_cand_name": n_s1[r2],
    }).cast(pl.Float32)


def decoy_features(df, base):
    """df: attached pairs; base: already-computed feature frame (same row order)."""
    def words(c):
        return pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))

    w = df.select(w1=words("core_1"), w2=words("core_2"))
    extra = w.select(
        extra_words2=pl.col("w2").list.set_difference(pl.col("w1")).list.len(),
        missing_words1=pl.col("w1").list.set_difference(pl.col("w2")).list.len(),
    )
    legal_differs = base["legal_eq"] == 0
    house_differs = (base["hn_both"] == 1) & (base["hn_eq"] == 0)
    return extra.with_columns(
        name_len_diff=(df["core_2"].str.len_chars().cast(pl.Int32) - df["core_1"].str.len_chars().cast(pl.Int32)),
        legal_and_house_differ=(legal_differs & house_differs),
        legal_differs_house_missing=(legal_differs & (base["hn_both"] == 0)),
    ).cast(pl.Float32)
