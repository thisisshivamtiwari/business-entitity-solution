"""Indian-script → English word dictionary, learned from the data (no external resources).

Why: 7.2% of true pairs have an S2/S3 name written in one of nine Indian scripts. unidecode only
transliterates them phonetically ("गोल्ड ट्रेडिंग" → "goldd ttreddin"), so their names never share
a word with the English S1 name. 20% of these pairs were missed.

What the EDA showed makes this easy:
  * a native name is a word-for-word rendering of the English name (same word count in ~100% of
    pairs once zero-width joiners are removed), and
  * the vocabulary is tiny: ~170 distinct words per script, 96% of test words already appear in
    the training pairs.

Steps
  learn       Align words by position in training pairs (folds 1-4 only, so the validation fold
              stays unseen). Keep a mapping if it occurs >= MIN_SUPPORT times and wins
              >= MIN_PURITY of its native word's alignments. One dictionary serves all nine
              scripts: each script has its own Unicode block, so words can't collide.
  learn-test  (experimental, NOT used by the pipeline) infer words that never occur in the training
              pairs from test inputs only. Finding: those ~10 words (motors, stores, sweets, …)
              are decoy vocabulary. They are rare in S1 names (9-700) but common in S2/S3 (3-9k), in
              train and test alike, so recovering them buys little recall, and the inference was
              wrong on test (stores -> "systems"). Unknown words therefore stay transliterated.
  check       Honesty check for learn-test: hide known words, recover them from training inputs
              (best setting recovers 54% of hidden words, 86% correctly).

Usage:  python translit.py learn | learn-test | check
Output: work/norm/indic_dict.json ({native word: english word})
"""
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz
from unidecode import unidecode

from config import SEED, WORK_DIR, gt_pq_path, pq_path
from evaluate import fold_expr

DICT_PATH = WORK_DIR / "norm" / "indic_dict.json"
VARIANTS_PATH = WORK_DIR / "norm" / "english_variants.json"
MIN_SUPPORT, MIN_PURITY = 5, 0.8
VARIANT_MIN_SHARE = 0.3            # each spelling must carry >= 30% of the native word's alignments
# tuned with `check` (hide 80 known words, recover them): 54% recovered, 86% of those correct
TEST_MIN_VOTES, TEST_MIN_SHARE, TEST_MIN_SOUND = 5, 0.5, 75

ZERO_WIDTH = re.compile("[​-‏⁠﻿]")
SPLIT = re.compile(r"[^\wऀ-෿]+")          # keep letters, digits and Indic vowel signs
INDIC = re.compile(r"[ऀ-෿]")               # Devanagari … Malayalam blocks
# English spellings that mean the same legal form, merged so their votes add up
CANON = {"private": "pvt", "limited": "ltd", "incorporated": "inc", "corporation": "corp",
         "company": "co", "pvt": "pvt", "ltd": "ltd"}


def tokens(text):
    """Split a name into words: zero-width joiners removed, punctuation dropped, lowercased."""
    text = unicodedata.normalize("NFC", ZERO_WIDTH.sub("", text or "")).lower()
    return [t for t in SPLIT.split(text) if t and t != "_"]


def english(tok):
    return CANON.get(tok, tok)


def is_indic(tok):
    return bool(INDIC.search(tok))


def skeleton(word):
    """Sound-alike key: transliterate, fold look-alike letters, drop vowels and repeats."""
    s = unidecode(word).lower()
    for a, b in (("ph", "f"), ("sh", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"), ("gh", "g"),
                 ("ch", "c"), ("w", "v"), ("z", "j"), ("q", "k"), ("x", "ks"), ("y", "i")):
        s = s.replace(a, b)
    s = re.sub(r"[^a-z]", "", s)
    s = s[:1] + re.sub(r"[aeiou]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", s)


# ---------------------------------------------------------------------------------------
# Learning from aligned training pairs
# ---------------------------------------------------------------------------------------

def load_named_pairs(folds_exclude_zero=None):
    """Training pairs whose S2/S3 name is in an Indian script. Experiments learn from folds 1-4 only
    (the validation fold stays unseen); the final run (BER_FINAL=1) learns from every fold."""
    if folds_exclude_zero is None:
        folds_exclude_zero = os.environ.get("BER_FINAL") != "1"
    names = pl.concat([pl.read_parquet(pq_path("train", n), columns=["entity_id", "business_name"])
                       for n in (1, 2, 3)])
    pairs = (pl.read_parquet(gt_pq_path())
             .join(names.rename({"entity_id": "s1", "business_name": "n1"}), on="s1")
             .join(names.rename({"entity_id": "cand", "business_name": "n2"}), on="cand")
             .filter(pl.col("n2").str.contains(r"[ऀ-෿]")))
    if folds_exclude_zero:
        pairs = pairs.filter(fold_expr("s1") != 0)
    return pairs


def align_counts(pairs, hidden=frozenset()):
    counts = defaultdict(Counter)
    for n1, n2 in zip(pairs["n1"].to_list(), pairs["n2"].to_list()):
        t1, t2 = tokens(n1), tokens(n2)
        if len(t1) != len(t2):
            continue
        for e, nat in zip(t1, t2):
            if is_indic(nat) and nat not in hidden:
                counts[nat][english(e)] += 1
    return counts


def confident(counts):
    out = {}
    for nat, c in counts.items():
        (eng, k), total = c.most_common(1)[0], sum(c.values())
        if k >= MIN_SUPPORT and k / total >= MIN_PURITY:
            out[nat] = eng
    return out


def spelling_variants(counts):
    """English spelling variants: a native word split between two English spellings.

    e.g. लक्ष्मी -> laxmi 2,809 / lakshmi 2,796, श्री -> shree / sree, जय -> jay / jai. Both spellings
    then mean the same word, so the minority spelling is mapped to the majority one everywhere.
    """
    pair_votes = Counter()
    for c in counts.values():
        top = c.most_common(2)
        total = sum(c.values())
        if len(top) == 2 and min(len(top[0][0]), len(top[1][0])) >= 3 and \
                min(top[0][1], top[1][1]) / total >= VARIANT_MIN_SHARE and \
                (top[0][1] + top[1][1]) / total >= 0.9 and top[0][1] + top[1][1] >= MIN_SUPPORT:
            pair_votes[tuple(sorted((top[0][0], top[1][0])))] += total
    english_freq = Counter()
    for c in counts.values():
        english_freq.update(c)
    variants = {}
    for (a, b), _ in pair_votes.most_common():
        major, minor = (a, b) if english_freq[a] >= english_freq[b] else (b, a)
        if minor not in variants and major not in variants:
            variants[minor] = major
    return variants


def learn():
    pairs = load_named_pairs()
    counts = align_counts(pairs)
    variants = spelling_variants(counts)
    # merge variant spellings before choosing each native word's English word
    merged = defaultdict(Counter)
    for nat, c in counts.items():
        for eng, k in c.items():
            merged[nat][variants.get(eng, eng)] += k
    mapping = confident(merged)
    VARIANTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VARIANTS_PATH.write_text(json.dumps(variants, indent=1, sort_keys=True))
    print(f"English spelling variants ({len(variants)}): {variants}")
    DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DICT_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=0, sort_keys=True))
    folds = "all folds" if os.environ.get("BER_FINAL") == "1" else "folds 1-4"
    print(f"learned {len(mapping):,} native→English mappings from {pairs.height:,} training pairs "
          f"({folds}) -> {DICT_PATH}")
    by_script = Counter(unicodedata.name(w[0]).split()[0] for w in mapping)
    print("  per script:", dict(by_script.most_common()))
    print("  examples:", dict(list(mapping.items())[:8]))


# ---------------------------------------------------------------------------------------
# Unseen words: vote from same-shaped English names (inputs only, no labels)
# ---------------------------------------------------------------------------------------

def infer_unknown(native_names, english_names, mapping, min_sound=TEST_MIN_SOUND,
                  min_votes=TEST_MIN_VOTES, min_share=TEST_MIN_SHARE):
    """For native words missing from `mapping`, vote an English word from same-shaped names.

    native_names / english_names: lists of raw name strings from the same country.
    Returns {native word: english word} for confident inferences only.
    """
    index = defaultdict(Counter)                         # (length, position, context) -> english word counts
    for name in english_names:
        t = [english(x) for x in tokens(name)]
        for i, w in enumerate(t):
            index[(len(t), i, tuple(t[:i] + t[i + 1:]))][w] += 1
    votes = defaultdict(Counter)
    for name in native_names:
        t = tokens(name)
        unknown = [i for i, w in enumerate(t) if is_indic(w) and w not in mapping]
        if len(unknown) != 1:
            continue
        i = unknown[0]
        ctx = [mapping.get(w, w) if is_indic(w) else english(w) for w in t]
        for w, k in index.get((len(t), i, tuple(ctx[:i] + ctx[i + 1:])), {}).items():
            votes[t[i]][w] += k
    inferred = {}
    for nat, c in votes.items():
        # score = votes x (sound-alike similarity)^2, among words that sound close enough
        scored = {}
        for w, k in c.items():
            sim = fuzz.ratio(skeleton(nat), skeleton(w))
            if sim >= min_sound:
                scored[w] = (k * (sim / 100) ** 2, k)
        if not scored:
            continue
        best = max(scored, key=lambda w: scored[w][0])
        share = scored[best][0] / sum(v[0] for v in scored.values())
        if scored[best][1] >= min_votes and share >= min_share:
            inferred[nat] = best
    return inferred


def _names(split, sources, country="India"):
    df = pl.concat([pl.read_parquet(pq_path(split, n), columns=["business_name", "country"]) for n in sources])
    return df.filter(pl.col("country") == country)["business_name"].to_list()


def learn_test():
    mapping = json.loads(DICT_PATH.read_text())
    native = [n for n in _names("test", (2, 3)) if INDIC.search(n)]
    inferred = infer_unknown(native, _names("test", (1,)), mapping)
    mapping.update(inferred)
    DICT_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=0, sort_keys=True))
    print(f"inferred {len(inferred)} unseen test words from test inputs (no labels): {inferred}")


def check(n_hidden=40, **params):
    """Hide known words, recover them from TRAINING inputs with infer_unknown, measure accuracy."""
    import random
    pairs = load_named_pairs()
    full = confident(align_counts(pairs))
    freq = Counter(w for n in pairs["n2"].to_list() for w in tokens(n) if w in full)
    candidates = [w for w, k in freq.most_common() if full[w] not in CANON.values()]
    random.Random(SEED).shuffle(candidates)
    hidden = set(candidates[:n_hidden])
    reduced = {k: v for k, v in full.items() if k not in hidden}
    native = [n for n in _names("train", (2, 3)) if INDIC.search(n)]
    got = infer_unknown(native, _names("train", (1,)), reduced, **params)
    got = {k: v for k, v in got.items() if k in hidden}
    correct = sum(full[k] == v for k, v in got.items())
    print(f"hidden {len(hidden)} known words; recovered {len(got)} ({len(got) / len(hidden):.0%}), "
          f"correct {correct}/{len(got)} ({correct / max(len(got), 1):.0%})")
    for k, v in list(got.items())[:10]:
        print(f"   {k} -> {v}  (truth: {full[k]})")


# ---------------------------------------------------------------------------------------
# Applying the dictionary (used by normalize.py)
# ---------------------------------------------------------------------------------------

_MAPPING = None
_VARIANTS = None


def variant(word):
    """Canonical spelling of an English word (learned variant groups), e.g. lakshmi -> laxmi."""
    global _VARIANTS
    if _VARIANTS is None:
        _VARIANTS = json.loads(VARIANTS_PATH.read_text()) if VARIANTS_PATH.exists() else {}
    return _VARIANTS.get(word, word)


def translate(name):
    """Replace every known native-script word in a name by its English word; keep the rest."""
    global _MAPPING
    if _MAPPING is None:
        _MAPPING = json.loads(DICT_PATH.read_text()) if DICT_PATH.exists() else {}
    if not _MAPPING or not INDIC.search(name or ""):
        return name
    return " ".join(_MAPPING.get(w, w) for w in tokens(name))


if __name__ == "__main__":
    {"learn": learn, "learn-test": learn_test, "check": check}[sys.argv[1]]()
