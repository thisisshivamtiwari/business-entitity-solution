"""Shared helpers: TSV loading, text cleaning, record serialisation, device pick.

Cleaning follows the project's preprocessing spec exactly: keep case, script and
accents (the multilingual encoder wants them); only strip invisible characters,
leading junk punctuation, null-like address parts and stray '#'.
"""
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

SOURCES = (1, 2, 3)
PREFIX = "query: "                       # e5 models: same prefix for both sides (symmetric task)

_INVISIBLE = re.compile("[​⁠﻿]")          # keep U+200C/U+200D: Indic scripts need them
_LEADING_JUNK = re.compile(r"^[^\w]+", re.UNICODE)       # \w == letters, digits, underscore
_WS = re.compile(r"\s+")
_HASHES = re.compile(r"#+")
_NULLISH = {"<null>", "null", "n/a", "na", "none", "nil", "unknown", "-", "--"}
_NONLATIN = re.compile(r"[ऀ-෿]")                # Devanagari … Sinhala block range (Indic scripts)
_KEY_PUNCT = re.compile(r"[^\w\s\u0900-\u0dff]", re.UNICODE)   # keep Indic vowel signs (category Mn/Mc)


def read_tsv(path):
    """Read a source TSV with quoting off and every column as str."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3, encoding="utf-8")


def clean_name(s: str) -> str:
    s = _INVISIBLE.sub("", s or "")
    s = _LEADING_JUNK.sub("", s)
    return _WS.sub(" ", s).strip()


def clean_address(s: str) -> str:
    s = _INVISIBLE.sub("", s or "")
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p and p.lower() not in _NULLISH]
    s = _HASHES.sub(" ", ", ".join(parts))
    return _WS.sub(" ", s).strip()


def name_key(s: str) -> str:
    """Lowercased, punctuation-free key used only for the exact-name block and hard negatives."""
    s = unicodedata.normalize("NFKC", s or "").lower().replace("&", " and ")
    return _WS.sub(" ", _KEY_PUNCT.sub(" ", s)).strip()


def full_text(name: str, address: str) -> str:
    return f"{PREFIX}{name} | {address}" if address else f"{PREFIX}{name}"


def name_text(name: str) -> str:
    return f"{PREFIX}{name}"


def has_indic_script(s: str) -> bool:
    return bool(_NONLATIN.search(s or ""))


def load_records(data_dir, split):
    """Load the three source files of a split into one cleaned DataFrame."""
    frames = []
    for s in SOURCES:
        df = read_tsv(Path(data_dir) / split / f"{split}_source{s}.tsv")
        df["source"] = s
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["name_clean"] = df["business_name"].map(clean_name)
    df["addr_clean"] = df["business_address"].map(clean_address)
    df["name_key"] = df["name_clean"].map(name_key)
    df["addr_missing"] = df["addr_clean"].eq("")
    df["indic_name"] = df["name_clean"].map(has_indic_script)
    return df


def pick_device():
    """CUDA > MPS > CPU, or force one with the DEVICE environment variable."""
    import torch

    forced = os.environ.get("DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
