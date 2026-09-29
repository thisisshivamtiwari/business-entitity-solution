"""Apply the stacked ensemble to the test set.

1. Re-fit the stacker on fold 0 with a POOL-SAFE feature set: only features that do not
   depend on how many S1 entities compete (fold 0 has 442k S1, the test set 1.73M, so
   claim counts per candidate would shift). Cross-fitted 5-fold by S1 entity for the
   honest score, a threshold-sensitivity curve, and the bars.
2. Join the two test pair tables, score with the stacker fitted on all of fold 0 and
   write work/ensemble/test_stacked.parquet. It also writes result files at the
   fold-0-tuned bars into --out-dir (work/ensemble/fold0_bars/, for reference only).
   The final files come from ensemble_apply.py with bars 0.70 / 0.70 / delta 0.2.

Inputs: work/ensemble/model_a/{fold0,test}_pair_probs.parquet (run_model_a.py) and
        work/ensemble/model_b/{fold0,test}_pair_probs.parquet (export_pair_probs.py).
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))                                   # config.py (paths)
sys.path.insert(0, str(SRC / "model_a" / "matching"))          # decide.py (assignment + bars + metric)
sys.path.insert(0, str(SRC / "model_a" / "blocking"))          # common.py (TSV reader)
from config import DATA_DIR, WORK_DIR  # noqa: E402
from common import read_tsv  # noqa: E402
from decide import Scorer, grid  # noqa: E402
from ensemble_fold0 import add_features, load_union, scorer_for  # noqa: E402

SAFE = ["ber_prob", "judge_prob", "combiner_prob", "emb_cosine", "blocking_cosine", "rank_b", "rank_a",
        "in_a", "in_b", "cand_addr_missing", "from_namesearch",
        "mean_prob", "min_prob", "max_prob", "abs_diff",
        "s1_max_ber", "s1_max_comb", "s1_n_cands", "s1_n_ber_hi", "s1_n_comb_hi", "gap_to_s1_best_ber", "gap_to_s1_best_comb"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, verbose=-1, num_threads=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(WORK_DIR / "ensemble"))
    ap.add_argument("--ground-truth", default=str(DATA_DIR / "train" / "train_ground_truth.tsv"))
    ap.add_argument("--test-s1", default=str(DATA_DIR / "test" / "test_source1.tsv"))
    ap.add_argument("--out-dir", default=str(WORK_DIR / "ensemble" / "fold0_bars"))
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--bar-shift", type=float, default=0.0, help="added to the tuned bars for the test set (more decoys there)")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--judge-fill-fold0", default=None); ap.add_argument("--judge-fill-test", default=None)
    args = ap.parse_args()
    W = Path(args.work); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ fold 0: pool-safe stacker
    u = add_features(load_union(W / "model_a/fold0_pair_probs.parquet", W / "model_b/fold0_pair_probs.parquet", args.judge_fill_fold0))
    gt = read_tsv(args.ground_truth)
    ents = set(u.s1_id.unique())
    truth = {e: set(x for x in l.split(",") if x) for e, l in zip(gt.source1_entity_id, gt.matched_entity_ids) if e in ents}
    fold = np.array([zlib.crc32(f"stack:{e}".encode()) % 5 for e in u.s1_id.to_numpy()])
    X = u[SAFE]; y = u.label.to_numpy(); oof = np.zeros(len(u), dtype=np.float32)
    for k in range(5):
        tr = fold != k
        oof[~tr] = lgb.train(PARAMS, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=args.rounds).predict(X[~tr])
    u["stack"] = oof
    sc = scorer_for(u, "stack", truth)
    half = np.array([zlib.crc32(f"half:{e}".encode()) % 2 for e in sc.ent]); A, B = half == 0, half == 1
    sA, pA = grid(sc, A); sB, pB = grid(sc, B)
    honest = (sc.macro(sc.keep_mask(**pA), B) + sc.macro(sc.keep_mask(**pB), A)) / 2
    sAll, bars = grid(sc, None)
    keep = sc.keep_mask(**bars); f, n_pred, tp = sc.f05_per_entity(keep); single = sc.n_true == 0
    print(f"[fold0] pool-safe stacker: honest macro F0.5 {honest:.4f} | in-sample {sAll:.4f} | bars {bars}")
    print(f"[fold0] singletons left empty {(n_pred[single]==0).mean():.4f} | false merges {(n_pred-tp).sum():,} | missed {(sc.n_true-tp).sum():,}")
    print("[fold0] sensitivity to the absolute bar (no-address bar moves with it, delta fixed):")
    gap = bars["t_noaddr"] - bars["t_abs"]
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        k2 = sc.keep_mask(t, min(0.995, t + gap), bars["delta"]); f2, n2, t2 = sc.f05_per_entity(k2)
        print(f"   t_abs {t:.2f}: macro F0.5 {f2.mean():.4f} | false merges {(n2-t2).sum():,} | missed {(sc.n_true-t2).sum():,} | singletons empty {(n2[single]==0).mean():.4f}")
    final = lgb.train(PARAMS, lgb.Dataset(X, label=y), num_boost_round=args.rounds)
    final.save_model(str(W / "stacker_safe.txt"))
    imp = sorted(zip(SAFE, final.feature_importance("gain")), key=lambda t: -t[1])
    print("[fold0] stacker features by gain:", ", ".join(f"{c}={g:.0f}" for c, g in imp[:8]))
    test_bars = dict(t_abs=round(min(0.99, bars["t_abs"] + args.bar_shift), 3),
                     t_noaddr=round(min(0.995, bars["t_noaddr"] + args.bar_shift), 3), delta=bars["delta"])
    (W / "ensemble_params.json").write_text(json.dumps(dict(fold0_bars=bars, test_bars=test_bars, honest_f05=honest,
                                                            bar_shift=args.bar_shift, features=SAFE), indent=2))
    del u, X, sc
    if args.skip_test:
        print("ENSEMBLE_FOLD0_SAFE_DONE"); return

    # ------------------------------------------------------------------ test
    t = add_features(load_union(W / "model_a/test_pair_probs.parquet", W / "model_b/test_pair_probs.parquet", args.judge_fill_test))
    print(f"[test] union pairs {len(t):,} | only Model A {((t.in_a==1)&(t.in_b==0)).sum():,} | only Model B {((t.in_a==0)&(t.in_b==1)).sum():,} | both {((t.in_a==1)&(t.in_b==1)).sum():,}")
    t["stack"] = final.predict(t[SAFE]).astype(np.float32)
    t[["s1_id", "cand_id", "stack", "ber_prob", "combiner_prob", "judge_prob", "in_a", "in_b", "cand_addr_missing", "selected_b"]].to_parquet(W / "test_stacked.parquet", index=False)
    d = pd.DataFrame({"s1_id": t.s1_id.to_numpy(), "cand_id": t.cand_id.to_numpy(), "rank": np.arange(len(t)),
                      "score": t["stack"].to_numpy(), "addr2_missing": t.cand_addr_missing.to_numpy()})
    sc = Scorer(d, None, score_col="score", assign=True)
    keep = sc.keep_mask(**test_bars)
    lists = sc.lists(keep)
    s1 = pd.read_csv(args.test_s1, sep="\t", dtype=str, keep_default_na=False, quoting=3, usecols=["entity_id", "country"])
    n_non = 0
    with open(out / "matching_results.tsv", "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for e in s1.entity_id:
            l = lists.get(e, []); n_non += bool(l)
            fh.write(f"{e}\t{','.join(l)}\n")
    # union candidate file, best stacked score first
    t = t.sort_values(["s1_id", "stack"], ascending=[True, False])
    cl = t.groupby("s1_id", sort=False).cand_id.agg(",".join).to_dict()
    with open(out / "candidate_pairs.tsv", "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for e in s1.entity_id:
            fh.write(f"{e}\t{cl.get(e, '')}\n")
    sizes = np.array([len(lists.get(e, [])) for e in s1.entity_id])
    print(f"[test] bars {test_bars} | rows {len(s1):,} | with matches {n_non/len(s1):.1%} | singletons {1-n_non/len(s1):.1%} | mean matches per matched S1 {sizes[sizes>0].mean():.2f}")
    s1["n"] = sizes
    print(s1.groupby("country").n.agg(entities="size", singleton_share=lambda x: (x == 0).mean(), mean_matches=lambda x: x[x > 0].mean()).round(3).to_string())
    print("ENSEMBLE_TEST_DONE")


if __name__ == "__main__":
    main()
