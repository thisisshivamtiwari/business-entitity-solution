"""Decision rule on the stacked test scores -> output/matching_results.tsv and output/candidate_pairs.tsv.

1. Assignment: an S2/S3 record may only go to the S1 that gives it the highest stacker score.
2. Keep a candidate if score >= bar (t_abs; t_noaddr when the candidate has no address) and
   score >= best score of that S1 - delta.
Bars used for the reported result: t_abs = t_noaddr = 0.70, delta = 0.2. candidate_pairs.tsv lists the union of both
models' candidates, best stacker score first."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "model_a" / "matching"))
from config import DATA_DIR, OUT_DIR, WORK_DIR  # noqa: E402
from decide import Scorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacked", default=str(WORK_DIR / "ensemble" / "test_stacked.parquet"))
    ap.add_argument("--test-s1", default=str(DATA_DIR / "test" / "test_source1.tsv"))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--t-abs", type=float, default=0.70)
    ap.add_argument("--t-noaddr", type=float, default=0.70)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--country-mix", default=None, help="COUNTRY:w  -> for S1 entities of that country the score becomes "
                    "(1-w)*stack + w*combiner_prob on pairs our model scored (pairs it never proposed keep the stack)")
    ap.add_argument("--records", default=str(WORK_DIR / "model_a" / "test_records.parquet"))
    ap.add_argument("--baseline", default=None, help="a matching_results.tsv to compare against")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    t = pd.read_parquet(args.stacked)
    if args.country_mix:
        ctry, w = args.country_mix.split(":"); w = float(w)
        c = pd.read_parquet(args.records, columns=["entity_id", "country"]).set_index("entity_id").country
        m = (c.loc[t.s1_id].to_numpy() == ctry) & t.combiner_prob.notna().to_numpy()
        t.loc[m, "stack"] = ((1 - w) * t.loc[m, "stack"] + w * t.loc[m, "combiner_prob"]).astype("float32")
        print(f"[apply] {ctry}: mixed {m.sum():,} pairs with weight {w} on our combiner")
    d = pd.DataFrame({"s1_id": t.s1_id.to_numpy(), "cand_id": t.cand_id.to_numpy(), "rank": np.arange(len(t)),
                      "score": t["stack"].to_numpy(), "addr2_missing": t.cand_addr_missing.to_numpy()})
    sc = Scorer(d, None, score_col="score", assign=True)
    lists = sc.lists(sc.keep_mask(args.t_abs, args.t_noaddr, args.delta))
    s1 = pd.read_csv(args.test_s1, sep="\t", dtype=str, keep_default_na=False, quoting=3, usecols=["entity_id", "country"])
    with open(out / "matching_results.tsv", "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for e in s1.entity_id:
            fh.write(f"{e}\t{','.join(lists.get(e, []))}\n")
    t = t.sort_values(["s1_id", "stack"], ascending=[True, False])
    cl = t.groupby("s1_id", sort=False).cand_id.agg(",".join).to_dict()
    with open(out / "candidate_pairs.tsv", "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for e in s1.entity_id:
            fh.write(f"{e}\t{cl.get(e, '')}\n")
    sizes = np.array([len(lists.get(e, [])) for e in s1.entity_id]); s1["n"] = sizes
    print(f"[apply] bars t_abs={args.t_abs} t_noaddr={args.t_noaddr} delta={args.delta} | rows {len(s1):,} | with matches {(sizes>0).mean():.1%} "
          f"| singletons {(sizes==0).mean():.1%} | mean matches per matched S1 {sizes[sizes>0].mean():.2f} | total links {sizes.sum():,}")
    print(s1.groupby("country").n.agg(entities="size", singleton_share=lambda x: (x == 0).mean(), mean_matches=lambda x: x[x > 0].mean()).round(3).to_string())
    if args.baseline:
        b = pd.read_csv(args.baseline, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        bl = {a: frozenset(x for x in v.split(",") if x) for a, v in zip(b.source1_entity_id, b.matched_entity_ids)}
        s1["changed"] = [frozenset(lists.get(e, [])) != bl[e] for e in s1.entity_id]
        s1["added"] = [len(set(lists.get(e, [])) - bl[e]) for e in s1.entity_id]
        s1["removed"] = [len(bl[e] - set(lists.get(e, []))) for e in s1.entity_id]
        print(f"[apply] vs baseline: rows changed {s1.changed.mean():.3%} ({s1.changed.sum():,}) | links added {s1.added.sum():,} | links removed {s1.removed.sum():,}")
        print(s1.groupby("country").agg(rows_changed=("changed", "mean"), added=("added", "sum"), removed=("removed", "sum")).round(4).to_string())
    print("APPLY_DONE")


if __name__ == "__main__":
    main()
