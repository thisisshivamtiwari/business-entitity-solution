"""Step 4a - features describing one (S1 record, candidate) pair.

The matcher never sees raw text; it sees ~80 numbers per pair, grouped as:

  record flags     candidate has a website name / native-script name / empty address ...
  name strings     fuzzy similarities of the names (rapidfuzz: ratio, token-set, Jaro-Winkler)
  address strings  fuzzy similarities of the addresses
  house number     equal? truncated ("2904" vs "290")? off by a few ("321" vs "322")?
                   - true matches show truncation noise, near-duplicate decoys show small
                   differences, so the model needs to tell these apart
  token overlap    IDF-weighted overlap of name / address tokens (rare shared words count more)
  sets             shared numbers, legal forms (Pvt Ltd vs Ltd), region agreement
  context          blocking / ranking scores and competition with other candidates

No feature uses the country label itself, so the model applies unchanged to France.
"""
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from config import norm_path
from extra_features import decoy_features, uniqueness_features
from name_features import all_features as name_features

# Normalised fields needed from both sides of a pair.
FIELDS = ["src", "country", "name", "core", "concat", "alias", "legal", "addr", "nums",
          "region", "f_domain", "f_nonlatin", "f_addr_empty", "f_landmark"]


def load_norm(split):
    return pl.read_parquet(norm_path(split), columns=["rid"] + FIELDS)


def attach(pairs, norm):
    """Add both records' fields to each pair: columns <field>_1 (S1) and <field>_2 (S2/S3).
    `rid` equals the row number of `norm`, so this is a fast positional gather."""
    fields = norm.drop("rid")
    left = fields[pairs["r1"].to_numpy()].rename(lambda c: c + "_1")
    right = fields[pairs["r2"].to_numpy()].rename(lambda c: c + "_2")
    return pairs.hstack(left).hstack(right)


# ---------------------------------------------------------------------------------------
# Fuzzy string similarities (rapidfuzz, parallelised over a process pool)
# ---------------------------------------------------------------------------------------

STR_COLS = [
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw",   # core names
    "cc_ratio", "cc_partial",                               # spaceless names (websites)
    "nfull_ratio",                                          # full names incl. legal form
    "a_ratio", "a_tset", "a_partial", "a_tsort",            # addresses
    "alias_tset",                                           # S1 name vs candidate's alias
    "hn_lev", "hn_absdiff",                                 # house numbers
]


def _string_block(args):
    """Compute STR_COLS for a block of pairs; each argument is a list of strings."""
    core1, core2, cc1, cc2, name1, name2, addr1, addr2, alias2, house1, house2 = args
    out = np.zeros((len(core1), len(STR_COLS)), dtype=np.float32)
    for i in range(len(core1)):
        a, b = core1[i], core2[i]
        out[i, 0] = fuzz.ratio(a, b)
        out[i, 1] = fuzz.token_sort_ratio(a, b)
        out[i, 2] = fuzz.token_set_ratio(a, b)
        out[i, 3] = fuzz.partial_ratio(a, b)
        out[i, 4] = JaroWinkler.normalized_similarity(a, b) * 100
        out[i, 5] = fuzz.ratio(cc1[i], cc2[i])
        out[i, 6] = fuzz.partial_ratio(cc1[i], cc2[i])
        out[i, 7] = fuzz.ratio(name1[i], name2[i])
        u, v = addr1[i], addr2[i]
        out[i, 8] = fuzz.ratio(u, v)
        out[i, 9] = fuzz.token_set_ratio(u, v)
        out[i, 10] = fuzz.partial_ratio(u, v)
        out[i, 11] = fuzz.token_sort_ratio(u, v)
        out[i, 12] = fuzz.token_set_ratio(a, alias2[i]) if alias2[i] else -1
        if house1[i] and house2[i]:
            out[i, 13] = Levenshtein.distance(house1[i], house2[i])
            out[i, 14] = min(abs(int(house1[i][:9]) - int(house2[i][:9])), 10**6)
        else:
            out[i, 13] = out[i, 14] = -1  # -1 = "not available" (trees handle it)
    return out


def house_number(side):
    """House number = first digit run of the address, in written order."""
    return pl.col(f"addr_{side}").str.extract(r"(\d+)")


def string_features(df, procs=12, block=100_000):
    cols = ["core_1", "core_2", "concat_1", "concat_2", "name_1", "name_2",
            "addr_1", "addr_2", "alias_2"]
    data = [df[c].to_list() for c in cols]
    data += [df.select(house_number(s).fill_null(""))[:, 0].to_list() for s in ("1", "2")]
    jobs = [[col[i:i + block] for col in data] for i in range(0, df.height, block)]
    with Pool(procs) as pool:
        blocks = pool.map(_string_block, jobs, chunksize=1)
    matrix = np.vstack(blocks) if blocks else np.zeros((0, len(STR_COLS)), np.float32)
    return pl.DataFrame(matrix, schema=STR_COLS)


# ---------------------------------------------------------------------------------------
# IDF-weighted token overlap
# ---------------------------------------------------------------------------------------

def idf_table(norm, col):
    """IDF of every token of `col`, per country: log(N records / records with the token).
    Frequent words ("services", "delhi") get low weight, rare ones high weight."""
    n_records = norm.group_by("country").len("N")
    return (
        norm.select("rid", "country", t=pl.col(col).str.split(" "))
        .explode("t", empty_as_null=True).drop_nulls().filter(pl.col("t") != "")
        .unique(["rid", "t"])
        .group_by("country", "t").len("df")
        .join(n_records, on="country")
        .select("country", "t", idf=(pl.col("N") / pl.col("df")).log().cast(pl.Float32))
    )


def overlap_features(df, idf, col, tag):
    """Shared vs unshared IDF mass between the token sets of both sides."""
    def tokens(side):
        return (
            df.select("pid", country="country_1", t=pl.col(f"{col}_{side}").str.split(" "))
            .explode("t", empty_as_null=True).drop_nulls().filter(pl.col("t") != "")
            .unique(["pid", "t"])
            .join(idf, on=["country", "t"], how="left")
            .with_columns(pl.col("idf").fill_null(pl.col("idf").max()))  # unseen = rarest
            .select("pid", "t", "idf")
        )

    a, b = tokens(1), tokens(2)
    mass_a = a.group_by("pid").agg(wa=pl.col("idf").sum(), maxa=pl.col("idf").max())
    mass_b = b.group_by("pid").agg(wb=pl.col("idf").sum())
    shared = a.join(b.select("pid", "t"), on=["pid", "t"]).group_by("pid").agg(
        wi=pl.col("idf").sum(), ni=pl.len(), maxi=pl.col("idf").max())
    x = (
        df.select("pid").join(mass_a, on="pid", how="left").join(mass_b, on="pid", how="left")
        .join(shared, on="pid", how="left").fill_null(0)
    )
    eps = 1e-6
    return x.select(
        pl.col("wi").alias(f"{tag}_wi"),                                          # shared mass
        (2 * pl.col("wi") / (pl.col("wa") + pl.col("wb") + eps)).alias(f"{tag}_dice"),
        (pl.col("wi") / (pl.col("wa") + eps)).alias(f"{tag}_cov1"),               # share of S1 covered
        (pl.col("wi") / (pl.col("wb") + eps)).alias(f"{tag}_cov2"),               # share of cand covered
        (pl.col("wa") - pl.col("wi")).alias(f"{tag}_miss1"),                      # S1-only mass
        (pl.col("wb") - pl.col("wi")).alias(f"{tag}_miss2"),                      # cand-only mass
        pl.col("ni").alias(f"{tag}_ni"),                                          # shared count
        (pl.col("maxi") / (pl.col("maxa") + eps)).alias(f"{tag}_maxrat"),         # rarest word shared?
    ).cast(pl.Float32)


# ---------------------------------------------------------------------------------------
# Set comparisons: numbers, legal forms, region, house number
# ---------------------------------------------------------------------------------------

def set_features(df):
    def as_list(c):
        return pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))

    x = df.select(
        n1=as_list("nums_1"), n2=as_list("nums_2"),
        l1=as_list("legal_1"), l2=as_list("legal_2"),
        g1=as_list("region_1"), g2=as_list("region_2"),
    )

    def common(a, b):
        return pl.col(a).list.set_intersection(pl.col(b)).list.len()

    def only(a, b):
        return pl.col(a).list.set_difference(pl.col(b)).list.len()

    def size(a):
        return pl.col(a).list.len()

    return x.select(
        num_inter=common("n1", "n2"),
        num_n1=size("n1"), num_n2=size("n2"),
        num_first_eq=(pl.col("n1").list.first() == pl.col("n2").list.first()).fill_null(False),
        num_conflict=(size("n1") > 0) & (size("n2") > 0) & (common("n1", "n2") == 0),
        num_only1=only("n1", "n2"), num_only2=only("n2", "n1"),
        legal_inter=common("l1", "l2"),
        legal_conflict=(size("l1") > 0) & (size("l2") > 0) & (common("l1", "l2") == 0),
        legal_n2=size("l2"),
        legal_eq=pl.col("l1").list.sort() == pl.col("l2").list.sort(),
        legal_only1=only("l1", "l2"), legal_only2=only("l2", "l1"),
        reg_eq=common("g1", "g2") > 0,
        reg_conflict=(size("g1") > 0) & (size("g2") > 0) & (common("g1", "g2") == 0),
        reg_missing=size("g2") == 0,
    ).cast(pl.Float32).hstack(house_features(df))


def house_features(df):
    h1, h2 = house_number("1"), house_number("2")
    both = h1.is_not_null() & h2.is_not_null()
    differ = both & (h1 != h2)
    return df.select(
        hn_both=both,
        hn_eq=both & (h1 == h2),
        hn_prefix=differ & (h1.str.starts_with(h2) | h2.str.starts_with(h1)),  # "2904"/"290"
        hn_suffix=differ & (h1.str.ends_with(h2) | h2.str.ends_with(h1)),      # "2880"/"880"
        hn_lendiff=(h1.str.len_chars().cast(pl.Int32)
                    - h2.str.len_chars().cast(pl.Int32)).fill_null(-9),
        # S1 house number appears anywhere among the candidate's numbers
        hn_in_nums2=pl.col("nums_2").str.contains(
            pl.concat_str(pl.lit(r"(^| )"), h1, pl.lit(r"( |$)"))).fill_null(False),
    ).cast(pl.Float32)


# ---------------------------------------------------------------------------------------
# Context from blocking / ranking
# ---------------------------------------------------------------------------------------

def add_block_context(cand):
    """Competition features. Must be computed on the FULL candidate table (every S1),
    before any sampling, so training samples see the same values as the test run."""
    return cand.with_columns(
        rrank_r=pl.col("rscore").rank("ordinal", descending=True).over("r2").cast(pl.Float32),
        rs_gap2=pl.col("rscore") - pl.col("rscore").max().over("r2"),
        b_gap=pl.col("bscore") / pl.col("bscore").max().over("r1"),
        n_cand=pl.len().over("r1").cast(pl.Float32),
        rrank=pl.col("bscore").rank("ordinal", descending=True).over("r2").cast(pl.Float32),
        n_rev=pl.len().over("r2").cast(pl.Float32),
        rev_gap=pl.col("bscore") / pl.col("bscore").max().over("r2"),
    )


CTX_COLS = ["bscore", "nkeys", "brank", "b_gap", "n_cand", "rrank", "n_rev", "rev_gap",
            "rscore", "rrank_r", "rs_gap2", "minn1", "s_nn", "s_na", "s_nr", "s_da", "s_aa", "s_c",
            "name_sim", "from_namesearch"]
# iteration 10 (BER_EMB=1): embedding similarity, neighbour ranks and within-S1 embedding gaps
EMB_CTX = ["cos", "erank", "rev_rank", "rev_gap_cos", "in_emb", "in_raw", "cos_gap1", "cos_rank1"]


# ---------------------------------------------------------------------------------------
# All features
# ---------------------------------------------------------------------------------------

def build(pairs, norm, idf_core, idf_addr):
    """Candidate pairs (with context columns) -> feature frame, same row order."""
    df = attach(pairs.with_row_index("pid"), norm)
    flags = df.select(
        src2=pl.col("src_2"),
        dom2=pl.col("f_domain_2"),
        nonlat2=pl.col("f_nonlatin_2"),
        aempty2=pl.col("f_addr_empty_2"),
        aempty1=pl.col("f_addr_empty_1"),
        land2=pl.col("f_landmark_2"),
        alias2=pl.col("alias_2") != "",
        len_c1=pl.col("core_1").str.len_chars(),
        len_c2=pl.col("core_2").str.len_chars(),
    ).cast(pl.Float32)
    base = pl.concat([
        pairs.select("r1", "r2"),
        flags,
        string_features(df),
        overlap_features(df, idf_core, "core", "nm"),
        overlap_features(df, idf_addr, "addr", "ad"),
        set_features(df),
        df.select(pl.col(CTX_COLS + [c for c in EMB_CTX if c in df.columns]).cast(pl.Float32)),
    ], how="horizontal")
    # name uniqueness (how many S1 share this name) and near-duplicate decoy signals (iteration 6);
    # models trained earlier select their own columns by name, so they are unaffected
    return pl.concat([base, uniqueness_features(df, norm), decoy_features(df, base),
                      name_features(df, norm, idf_core)], how="horizontal")      # iteration 9b
