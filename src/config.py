"""Shared paths and settings for every pipeline step.

Default layout (all paths can be overridden with environment variables):

    <project>/data/                       raw records: train/, test/    (BER_DATA)
    <project>/work/                       intermediate files, created   (BER_WORK)
    <project>/output/                     result files, created         (BER_OUT)
    <project>/src/                        this code

<project> is the repository root.
"""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]          # repository root
PROJECT_DIR = Path(os.environ.get("BER_PROJECT", PACKAGE_DIR))

DATA_DIR = Path(os.environ.get("BER_DATA", PROJECT_DIR / "data"))
WORK_DIR = Path(os.environ.get("BER_WORK", PROJECT_DIR / "work"))
OUT_DIR = Path(os.environ.get("BER_OUT", PROJECT_DIR / "output"))

# One seed for every sampling / model step, so reruns give the same result.
SEED = 42

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def raw_path(split, source):
    """Original source file, e.g. data/train/train_source2.tsv."""
    return DATA_DIR / split / f"{split}_source{source}.tsv"


def pq_path(split, source):
    """Parquet copy of a source file (written by convert.py)."""
    return WORK_DIR / "parquet" / f"{split}_source{source}.parquet"


def gt_raw_path():
    return DATA_DIR / "train" / "train_ground_truth.tsv"


def gt_pq_path():
    """Ground truth as one row per (S1 id, matched S2/S3 id) pair."""
    return WORK_DIR / "parquet" / "train_pairs.parquet"


def norm_path(split):
    """Normalised records of all three sources (written by normalize.py)."""
    return WORK_DIR / "norm" / f"{split}.parquet"
