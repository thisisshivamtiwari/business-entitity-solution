"""Encoder v1 text recipe: NFKC, lowercase, '&' -> 'and', punctuation stripped; the encoder input is
'query: name: {name} | address: {address} | country: {country}' (the address tag stays when the address is empty)."""
import re
import unicodedata
from pathlib import Path

import pandas as pd

SOURCES = (1, 2, 3)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3, encoding="utf-8")


def normalise(text: str) -> str:
    """NFKC, lowercase, '&' -> 'and', drop punctuation, collapse whitespace."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).lower().replace("&", " and ")
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def serialise(name: str, address: str, country: str, prefix: str = "query: ") -> str:
    return f"{prefix}name: {name} | address: {address} | country: {country.lower()}"


def load_records(data_dir, split):
    frames = []
    for s in SOURCES:
        df = read_tsv(Path(data_dir) / split / f"{split}_source{s}.tsv")
        df["source"] = s
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["name_norm"] = df["business_name"].map(normalise)
    df["addr_norm"] = df["business_address"].map(normalise)
    df["addr_missing"] = df["addr_norm"].eq("")
    return df
