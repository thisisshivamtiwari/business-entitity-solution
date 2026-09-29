"""Step 1 - normalise business names and addresses into comparable token strings.

Raw records are noisy in ways that defeat exact comparison:
    "Payne Enterprises"  vs  "PAYNE-ENRTPRMISES"  vs  "paynenterprises.com"
    "Plot No. ##80, ..., महाराष्ट्र"  vs  "80, ..., Maharashtra"  vs  "..., MH"
This step maps every record to canonical lowercase ASCII tokens so that later steps can
compare them. It is deliberately country-agnostic: one shared dictionary covers legal
forms and street words for all countries, and anything more specific is left to IDF
weighting downstream (so France, absent from training, gets the same treatment).

Output: work/norm/{split}.parquet, one row per record of all three sources:
    rid           row number (a compact integer id used by every later step)
    entity_id, src (1/2/3), country, name_raw, addr_raw
    name          normalised full name, legal forms in canonical spelling
    core          name without legal forms, stop words or repeated words
    concat        core with spaces removed (matches domain-style names)
    alias         text before a "dba" / "t/a" / "formerly" marker, else ""
    legal         canonical legal-form tokens, sorted ("ltd pvt")
    addr          normalised address tokens (region removed, see below)
    nums          unique numbers found in the address, sorted
    region        canonical state / region token(s), e.g. "@oh", "@mh", "@hdf"
    f_domain, f_nonlatin, f_addr_empty, f_landmark   0/1 flags
"""
import json
import re
import sys
import unicodedata
from multiprocessing import Pool

import polars as pl
from unidecode import unidecode

from config import WORK_DIR, gt_pq_path, norm_path, pq_path
from geo import CITY_SYN, REGION
from translit import translate, variant

# ---------------------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------------------

# Legal forms -> canonical spelling. They are kept apart from the "core" name because
# sources add, drop and reorder them freely ("Pvt Ltd" / "Private Limited" / "(Pvt)").
LEGAL = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt", "pvtltd": "pvt ltd", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc",
    "pc": "pc", "public": "public", "opc": "opc",
    "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scop": "scop", "selarl": "selarl", "ets": "ets", "etablissements": "ets",
    # iter9: more French forms (seen in the test set, none in training)
    "association": "assoc", "asso": "assoc", "ei": "ei", "eirl": "eirl", "cie": "co", "gie": "gie",
    "scp": "scp", "sccv": "sccv", "selas": "selas",
}
# Words that carry no identity ("Jyoti & Sons" == "Jyoti Sons").
STOP = {"the", "and", "of", "et", "de", "du", "des", "la", "le", "les", "m/s", "ms"}

# "Orbiriza t/a Bombay Projects LLP": the real name follows the marker.
ALIAS_RE = re.compile(r"\s(?:dba|d/b/a|t/a|trading as|formerly|fka|f/k/a|aka|a/k/a)\s")
# "maurewilliamscolombier.com": a website used as the name.
DOMAIN_RE = re.compile(r"^(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|co|biz|info|us|io)$")
NONWORD_RE = re.compile(r"[^a-z0-9/ ]+")
WS_RE = re.compile(r"\s+")
DIGITS_RE = re.compile(r"\d+")
ZERO_WIDTH_RE = re.compile("[​-‏⁠﻿]")

# Placeholders that mean "no value" ("<NULL>", "N/A", ...), dropped from addresses.
ADDR_NULL = {"", "null", "<null>", "n/a", "na", "none", "nil", "-", "--", "unknown"}
# Street words / abbreviations -> one short form ("Street" == "St" == "ST.").
# Note "saint" -> "st": sources corrupt "St" into "Saint", and French place names
# ("Saint-Nazaire" / "St-Nazaire") need the same mapping anyway.
ADDR_CANON = {
    "street": "st", "str": "st", "saint": "st", "st": "st",
    "avenue": "ave", "av": "ave", "ave": "ave", "avn": "ave",
    "road": "rd", "rd": "rd", "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd",
    "court": "ct", "ct": "ct", "place": "pl", "pl": "pl", "circle": "cir", "cir": "cir",
    "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy", "terrace": "ter",
    "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq", "suite": "ste", "ste": "ste",
    "apartment": "apt", "apt": "apt", "building": "bldg", "bldg": "bldg", "floor": "fl",
    "fl": "fl", "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "nagar": "ngr", "ngr": "ngr", "colony": "col", "sector": "sec", "sec": "sec",
    "marg": "marg", "chowk": "chowk", "township": "twp", "twp": "twp",
    "rue": "rue", "r": "rue", "chemin": "ch", "ch": "ch", "allee": "all", "impasse": "imp",
    "imp": "imp", "route": "rte", "rte": "rte", "quai": "quai",
    "cours": "crs", "crs": "crs", "appt": "apt", "appartement": "apt",
}
# iter9 (French address formats): "N°10" -> "10", "5 bis" -> "5", "l'Hermitte" -> "hermitte".
# A letter glued to a number ("40B", "43A") is kept: in Indian addresses it distinguishes units
# (stripping it lowered validation recall in experiment iter9a).
NUMBER_MARK_RE = re.compile(r"\bn(?:deg|o)\s*(?=\d)")                 # N° / No written before a number
NUMBER_SUFFIX_RE = re.compile(r"\b(\d+)\s*(?:bis|ter|quater)\b")
ELISION_RE = re.compile(r"\b[ld]'")
# Words that only introduce a number ("Plot No. 80", "Door No 38") or a landmark.
ADDR_DROP = {"no", "number", "door", "plot", "house", "h", "hno", "kh", "khasra", "flat",
             "unit", "near", "opp", "opposite", "behind", "c/o", "po", "box"}
LANDMARK = {"near", "opp", "opposite", "behind", "beside"}


# ---------------------------------------------------------------------------------------
# Record-level normalisation (pure functions, run in a process pool)
# ---------------------------------------------------------------------------------------

def is_nonlatin(text):
    """True if the text contains letters outside Latin scripts (Devanagari, Tamil, ...)."""
    return any(ord(ch) > 0x24F and ch.isalpha() for ch in text)


def to_ascii_lower(text):
    """Strip zero-width characters, transliterate every script to ASCII, lowercase."""
    text = ZERO_WIDTH_RE.sub("", text)
    return unidecode(unicodedata.normalize("NFKC", text)).lower()


def norm_name(raw):
    """Normalise one business name.

    Returns (name, core, concat, alias, legal, f_domain, f_nonlatin).
    """
    nonlatin = is_nonlatin(ZERO_WIDTH_RE.sub("", raw))
    # Indian-script words -> English words via the learned dictionary (translit.py); anything
    # left unmapped is transliterated phonetically by to_ascii_lower as before.
    s = WS_RE.sub(" ", to_ascii_lower(translate(raw))).strip()

    # "Kelovantage dba Shree Foods" -> alias "kelovantage", name "shree foods"
    alias = ""
    parts = ALIAS_RE.split(f" {s} ")
    if len(parts) > 1:
        alias, s = parts[0].strip(), parts[-1].strip()

    # A website as the name: keep only its stem, which equals the spaceless name.
    domain = DOMAIN_RE.match(s)
    if domain:
        stem = domain.group(1).replace("-", "")
        return stem, stem, stem, alias, "", 1, int(nonlatin)

    # "S.A.S" -> "sas", "&"/"+" -> "and", then drop remaining punctuation.
    s = s.replace(".", "").replace("&", " and ").replace("+", " and ")
    s = s.replace("m/s", " ").replace("/", " ")
    s = NONWORD_RE.sub(" ", s)

    name_tokens, core, legal = [], [], []
    for tok in s.split():
        if tok in LEGAL:
            canon = LEGAL[tok]
            name_tokens.append(canon)
            legal.extend(canon.split())
        else:
            tok = variant(tok)          # learned spelling variants: lakshmi -> laxmi, sree -> shree
            name_tokens.append(tok)
            # Skip stop words and immediate repeats ("Jyoti Jyoti & Sons").
            if tok not in STOP and (not core or core[-1] != tok):
                core.append(tok)

    return (" ".join(name_tokens), " ".join(core), "".join(core), alias,
            " ".join(sorted(set(legal))), 0, int(nonlatin))


def region_of_component(component):
    """Return a region token if this comma-separated address part is a state / region."""
    if is_nonlatin(component) and not any(ch.isdigit() for ch in component):
        # A whole component in a native script is a state name in this data
        # ("महाराष्ट्र"). Keep its transliteration; learn_region_map() maps it later.
        translit = NONWORD_RE.sub(" ", unidecode(component).lower()).split()
        return "@x_" + "_".join(translit)
    key = to_ascii_lower(component).strip()
    key = WS_RE.sub(" ", key.replace(".", "").replace("-", " ").replace("'", " ")).strip()
    if key in REGION:
        return "@" + REGION[key]
    return None


def norm_addr(raw):
    """Normalise one address.

    Returns (addr, nums, region, f_addr_empty, f_landmark). Region components (a known
    state name / code or a native-script component) are moved out of `addr` into `region`,
    because sources disagree on how to write them ("OH" / "Ohio", "UP" / "Uttar Pradesh").
    """
    components, region = [], []
    for comp in ZERO_WIDTH_RE.sub("", raw).split(","):
        comp = comp.strip()
        reg = region_of_component(comp)
        if reg is not None:
            region.append(reg)
            continue
        comp = to_ascii_lower(comp).strip()
        if comp not in ADDR_NULL:
            components.append(comp)
    region = " ".join(sorted(set(region)))
    if not components:
        return "", "", region, 1, 0

    s = " ".join(components).replace("c/o", " ").replace("#", " ").replace("&", " and ")
    s = NUMBER_MARK_RE.sub(" ", s)
    s = NUMBER_SUFFIX_RE.sub(r"\1", s)
    s = ELISION_RE.sub("", s)
    s = NONWORD_RE.sub(" ", s.replace("-", " "))

    tokens, landmark = [], 0
    for tok in s.replace("/", " / ").split():
        if tok == "/":
            continue
        if tok in LANDMARK:
            landmark = 1
        if tok in ADDR_DROP or tok in ADDR_NULL:
            continue
        if tok.isdigit():
            tok = tok.lstrip("0") or "0"  # "02880" == "2880"
        tok = ADDR_CANON.get(tok, tok)
        tokens.append(CITY_SYN.get(tok, tok))  # "bombay" -> "mumbai", ...

    nums = sorted({d.lstrip("0") or "0" for d in DIGITS_RE.findall(s)})
    return " ".join(tokens), " ".join(nums), region, int(not tokens), landmark


def _normalize_chunk(args):
    names, addrs = args
    return [norm_name(n) for n in names], [norm_addr(a) for a in addrs]


# ---------------------------------------------------------------------------------------
# Learned native-script region names
# ---------------------------------------------------------------------------------------

def learn_region_map(norm, min_support=20, min_purity=0.8):
    """Learn "@x_mhaaraassttr" -> "@mh" style mappings from the training pairs.

    For each true pair whose S1 side has exactly one canonical region and whose S2/S3 side
    has a native-script region token, count how often the two co-occur, and keep mappings
    seen at least `min_support` times with at least `min_purity` agreement.
    """
    pairs = pl.read_parquet(gt_pq_path())
    reg = norm.select("entity_id", "region")
    counts = (
        pairs.join(reg.rename({"entity_id": "s1", "region": "r1"}), on="s1")
        .join(reg.rename({"entity_id": "cand", "region": "r2"}), on="cand")
        .filter(~pl.col("r1").str.contains(" ") & pl.col("r1").str.starts_with("@")
                & ~pl.col("r1").str.starts_with("@x_"))
        .with_columns(pl.col("r2").str.split(" "))
        .explode("r2", empty_as_null=True)
        .filter(pl.col("r2").str.starts_with("@x_"))
        .group_by("r2", "r1").len()
        .with_columns(total=pl.col("len").sum().over("r2"))
    )
    best = (
        counts.sort("len", descending=True).unique("r2", keep="first")
        .filter((pl.col("total") >= min_support) & (pl.col("len") / pl.col("total") >= min_purity))
    )
    return dict(zip(best["r2"], best["r1"]))


def apply_region_map(norm, mapping):
    if not mapping:
        return norm
    return norm.with_columns(
        region=pl.col("region").str.split(" ")
        .list.eval(pl.element().replace(mapping)).list.unique().list.sort().list.join(" ")
    )


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------

def normalize_split(split, procs=12, chunk=50_000):
    """Normalise all three sources of one split; `rid` = row number across sources."""
    frames, offset = [], 0
    for source in (1, 2, 3):
        df = pl.read_parquet(pq_path(split, source)).with_columns(
            rid=pl.int_range(offset, offset + pl.len(), dtype=pl.UInt32),
            src=pl.lit(source, pl.UInt8),
        )
        offset += df.height
        frames.append(df)
    df = pl.concat(frames)

    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    name_rows, addr_rows = [], []
    with Pool(procs) as pool:
        for n, a in pool.imap(_normalize_chunk, jobs):  # imap keeps the input order
            name_rows.extend(n)
            addr_rows.extend(a)

    name_df = pl.DataFrame(name_rows, orient="row", schema=[
        "name", "core", "concat", "alias", "legal", "f_domain", "f_nonlatin"])
    addr_df = pl.DataFrame(addr_rows, orient="row", schema=[
        "addr", "nums", "region", "f_addr_empty", "f_landmark"])
    return (
        df.rename({"business_name": "name_raw", "business_address": "addr_raw"})
        .hstack(name_df).hstack(addr_df)
        .with_columns(pl.col("^f_.*$").cast(pl.UInt8))
    )


def main(splits):
    normalized = {split: normalize_split(split) for split in splits}

    # The native-script region map is learned on train only and reused for test.
    map_path = WORK_DIR / "norm" / "region_map.json"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    if "train" in normalized:
        mapping = learn_region_map(normalized["train"])
        map_path.write_text(json.dumps(mapping, indent=1, ensure_ascii=False))
        print(f"learned {len(mapping)} native-script region mappings -> {map_path}")
    mapping = json.loads(map_path.read_text())

    for split, df in normalized.items():
        df = apply_region_map(df, mapping)
        df.write_parquet(norm_path(split))
        print(f"{split}: {df.height:,} records -> {norm_path(split)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["train", "test"])
