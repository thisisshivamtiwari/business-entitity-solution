"""Step 5 - score the test candidates and write the two result files.

  output/candidate_pairs.tsv   every candidate the model scored (the blocking output)
  output/matching_results.tsv  the final matches - the final matches

Both files have exactly one row per test S1 entity (an empty list when there is nothing),
tab-separated, with comma-separated S2/S3 ids - the format the validator checks.
"""
import pickle
import time

import polars as pl

import features as F
from config import OUT_DIR, WORK_DIR, norm_path
from memguard import start_memory_guard
from train import featurize, load_candidates, predict_proba, select

S1_PER_CHUNK = 250_000


def write_id_lists(pairs, s1_ids, ids, column, path):
    """(r1, r2) pairs -> TSV with one row per S1 and a sorted comma-joined id list."""
    lists = (
        pairs.join(ids.rename({"rid": "r2", "entity_id": "eid"}), on="r2")
        .sort("r1", "eid")
        .group_by("r1").agg(pl.col("eid").str.join(","))
    )
    table = s1_ids.join(lists, on="r1", how="left").select(
        source1_entity_id="entity_id", **{column: pl.col("eid").fill_null("")})
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(path, separator="\t", quote_style="never")
    return table


def reselect(threshold=None, world="trainlike"):
    """Rewrite matching_results.tsv from the saved test probabilities at another threshold.

    By default the threshold is the best one found in the test-like validation world
    (work/results/<tag>__trainlike.json, most recent), because test has test-like decoy density.
    """
    import json
    if threshold is None:
        latest = max((WORK_DIR / "results").glob(f"*__{world}.json"), key=lambda p: p.stat().st_mtime)
        threshold = json.loads(latest.read_text())["best_threshold"]
        print(f"threshold {threshold:.2f} = best in the {world} world ({latest.name})")
    pred = pl.read_parquet(WORK_DIR / "test_pred.parquet")
    ids = pl.read_parquet(norm_path("test"), columns=["rid", "entity_id", "src"])
    s1_ids = ids.filter(pl.col("src") == 1).select(r1="rid", entity_id="entity_id")
    matches = select(pred, threshold)
    result = write_id_lists(matches, s1_ids, ids.select("rid", "entity_id"), "matched_entity_ids",
                            OUT_DIR / "matching_results.tsv")
    n_empty = (result["matched_entity_ids"] == "").sum()
    print(f"re-selected at threshold {threshold:.2f}: {matches.height:,} matches, {n_empty:,} S1 with no match "
          f"-> {OUT_DIR / 'matching_results.tsv'}")


def main():
    t0 = time.time()
    with open(WORK_DIR / "model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, feat_cols, threshold = bundle["model"], bundle["feat_cols"], bundle["thr"]

    norm = F.load_norm("test")
    ids = pl.read_parquet(norm_path("test"), columns=["rid", "entity_id", "src"])
    s1_ids = ids.filter(pl.col("src") == 1).select(r1="rid", entity_id="entity_id")  # file order
    ids = ids.select("rid", "entity_id")

    cand = load_candidates("test")
    print(f"test: {cand.height:,} candidates for {cand['r1'].n_unique():,} S1", flush=True)
    idf_core, idf_addr = F.idf_table(norm, "core"), F.idf_table(norm, "addr")

    # Score in chunks of S1 to keep memory bounded.
    scored = []
    r1s = cand["r1"].unique().sort()
    for i in range(0, len(r1s), S1_PER_CHUNK):
        chunk = cand.join(pl.DataFrame({"r1": r1s[i:i + S1_PER_CHUNK]}), on="r1", how="semi")
        x = featurize(chunk, norm, idf_core, idf_addr)
        p = predict_proba(bundle, x, norm, "test")
        scored.append(x.select("r1", "r2").with_columns(p=p))
        print(f"  scored {min(i + S1_PER_CHUNK, len(r1s)):,}/{len(r1s):,} S1 "
              f"({time.time() - t0:.0f}s)", flush=True)
    pred = pl.concat(scored)
    pred.write_parquet(WORK_DIR / "test_pred.parquet")

    matches = select(pred, threshold)
    write_id_lists(cand.select("r1", "r2"), s1_ids, ids, "candidate_entity_ids",
                   OUT_DIR / "candidate_pairs.tsv")
    result = write_id_lists(matches, s1_ids, ids, "matched_entity_ids",
                            OUT_DIR / "matching_results.tsv")

    n_empty = (result["matched_entity_ids"] == "").sum()
    print(f"wrote {result.height:,} S1 rows: {matches.height:,} matches, {n_empty:,} S1 with "
          f"no match, {cand.height / result.height:.1f} candidates per S1, threshold "
          f"{threshold:.2f} ({time.time() - t0:.0f}s)")
    print(f"  -> {OUT_DIR / 'matching_results.tsv'}   (upload this file)")
    print(f"  -> {OUT_DIR / 'candidate_pairs.tsv'}")


if __name__ == "__main__":
    import sys
    start_memory_guard("predict")
    if len(sys.argv) > 1 and sys.argv[1] == "reselect":      # python predict.py reselect [threshold]
        reselect(float(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        main()
