"""Step 0 - convert the raw TSV files to Parquet.

Parquet loads in seconds instead of minutes, which matters because every later step
re-reads the data. All columns are kept as plain strings; nothing is cleaned here.

The ground truth file lists, per S1 entity, a comma-separated string of matched ids. We
"explode" it into one row per (s1, cand) pair, which is the shape every later join needs.
S1 entities without matches simply have no rows (they are recovered from the S1 table).
"""
import polars as pl

from config import SOURCES, SPLITS, gt_pq_path, gt_raw_path, pq_path, raw_path

EXPECTED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    """Read a source TSV exactly as-is: tab separated, no quote handling, all strings."""
    return pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema=False, truncate_ragged_lines=True,
    )


def convert_sources():
    for split in SPLITS:
        for source in SOURCES:
            df = read_tsv(raw_path(split, source))
            assert df.columns == EXPECTED_COLUMNS, (split, source, df.columns)
            assert df["entity_id"].is_unique().all(), "entity ids must be unique"

            # Missing name/address/country become "" so later string code never sees None.
            df = df.with_columns(pl.col(c).fill_null("") for c in EXPECTED_COLUMNS[1:])
            df.write_parquet(pq_path(split, source))

            countries = df["country"].value_counts().sort("country").rows()
            print(f"{split} source{source}: {df.height:,} rows {countries}")


def convert_ground_truth():
    gt = read_tsv(gt_raw_path())
    pairs = (
        gt.filter(pl.col("matched_entity_ids").fill_null("") != "")
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids", empty_as_null=True)
        .rename({"source1_entity_id": "s1", "matched_entity_ids": "cand"})
    )
    pairs.write_parquet(gt_pq_path())
    print(f"ground truth: {gt.height:,} S1 entities -> {pairs.height:,} matched pairs")


def main():
    pq_path("train", 1).parent.mkdir(parents=True, exist_ok=True)
    convert_sources()
    convert_ground_truth()


if __name__ == "__main__":
    main()
