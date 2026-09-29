"""Export ready-to-embed text for an external embedding model (e.g. a multilingual sentence encoder).

Why not hand over the pipeline's normalised text? normalize.py transliterates every script to ASCII
(good for string matching, bad for a multilingual model, which understands the original script).
So this export keeps the ORIGINAL text, only lightly cleaned, and adds our normalised English view
as an optional second input.

Writes <project>/share/embedding_input/:
    records_train.parquet, records_test.parquet   one row per record (all three sources)
    train_pairs.parquet                           ground truth: (s1, cand) entity-id pairs
Usage:  ../.venv/bin/python export_embedding_input.py
"""
import re

import polars as pl

from config import PROJECT_DIR, gt_pq_path, norm_path
from evaluate import fold_expr
from memguard import start_memory_guard

OUT = PROJECT_DIR / "share" / "embedding_input"

# placeholder address parts ("<NULL>", "N/A", ...) - same list as normalize.ADDR_NULL
PLACEHOLDER = r"(?i)^\s*(<null>|null|n/a|na|none|nil|unknown|-|--)\s*$"
# zero-width space / BOM only; ZWNJ/ZWJ (U+200C/D) are kept because Indic scripts need them
INVISIBLE = "[​⁠﻿]"


def clean_name(col):
    """Original script and accents kept; junk punctuation prefixes and extra spaces removed."""
    c = pl.col(col).str.replace_all(INVISIBLE, "")
    c = c.str.replace(r"^[^\p{L}\p{N}]+", "")                  # "-- Holloway", "<< Team", "#oncology"
    return c.str.replace_all(r"\s+", " ").str.strip_chars()


def clean_addr(col):
    """Placeholder parts dropped, '#'-runs removed, empty parts removed; script kept."""
    parts = pl.col(col).str.replace_all(INVISIBLE, "").str.split(",")
    parts = parts.list.eval(pl.element().str.strip_chars()
                            .filter(~pl.element().str.contains(PLACEHOLDER) & (pl.element() != "")))
    return parts.list.join(", ").str.replace_all(r"#+", " ").str.replace_all(r"\s+", " ").str.strip_chars()


def export(split):
    d = pl.read_parquet(norm_path(split), columns=[
        "rid", "entity_id", "src", "country", "name_raw", "addr_raw", "name", "addr", "region",
        "f_nonlatin", "f_addr_empty"])
    out = d.select(
        "entity_id",
        pl.col("rid").alias("row"),                                   # 0..n-1, stable order
        source=pl.format("S{}", "src"),
        country="country",
        name_text=clean_name("name_raw"),                             # <- embed this
        address_text=clean_addr("addr_raw"),                          # <- and this
        name_en=pl.col("name"),                                       # optional: our English/ASCII view
        address_en=pl.concat_str([pl.col("addr"), pl.col("region").str.replace_all("@", "")],
                                 separator=" ").str.strip_chars(),
        native_script_name=pl.col("f_nonlatin").cast(pl.Boolean),
        has_address=(pl.col("f_addr_empty") == 0),
        # fold of the S1 record (train only): fold 0 is OUR validation set - do not train on it
        fold=pl.when(pl.col("src") == 1).then(fold_expr()).otherwise(None) if split == "train"
        else pl.lit(None, pl.UInt8),
        split=pl.lit(split),
    ).with_columns(
        # one string for "name + address" embeddings; name alone when there is no address
        full_text=pl.when(pl.col("address_text") != "")
        .then(pl.concat_str(["name_text", "address_text"], separator=" | "))
        .otherwise(pl.col("name_text")),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT / f"records_{split}.parquet", compression="zstd")
    print(f"{split}: {out.height:,} records -> {OUT / f'records_{split}.parquet'}")
    return out


if __name__ == "__main__":
    start_memory_guard("export")
    for split in ("train", "test"):
        export(split)
    pl.read_parquet(gt_pq_path()).write_parquet(OUT / "train_pairs.parquet")
    print(f"ground truth pairs -> {OUT / 'train_pairs.parquet'}")
