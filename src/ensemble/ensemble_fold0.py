"""Combine two matchers on fold 0 (validation only).

Model A (cross-encoder judge + LightGBM combiner):                    work/ensemble/model_a/fold0_pair_probs.parquet
Model B (keyword + embedding blocking, ranker, 113-feature LightGBM): work/ensemble/model_b/fold0_pair_probs.parquet

Analysis script: baselines, error overlap, simple averages and the stacker with every feature.
The final stacker (pool-safe features) is fitted by ensemble_test.py, which imports
load_union / add_features / scorer_for from this file.

Nothing here touches the base models. The stacker is a second-level model fitted with
5-fold cross-fitting BY S1 ENTITY inside fold 0: each entity is scored by a stacker that
never saw it, so every number reported is out of fold. Bars are tuned on half A and scored
on half B (and swapped), as in decide.py.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))                                   # config.py (paths)
sys.path.insert(0, str(SRC / "model_a" / "matching"))          # decide.py (assignment + bars + metric)
sys.path.insert(0, str(SRC / "model_a" / "blocking"))          # common.py (TSV reader)
from config import DATA_DIR, WORK_DIR  # noqa: E402
from common import read_tsv  # noqa: E402
from decide import Scorer, grid  # noqa: E402

FEATS = ["ber_prob", "judge_prob", "combiner_prob", "emb_cosine", "blocking_cosine", "rank_b", "rank_a",
         "in_a", "in_b", "cand_addr_missing", "from_namesearch", "selected_b",
         "mean_prob", "min_prob", "max_prob", "abs_diff",
         "s1_max_ber", "s1_max_comb", "s1_n_cands", "s1_n_ber_hi", "s1_n_comb_hi", "gap_to_s1_best_ber", "gap_to_s1_best_comb",
         "cand_n_claims_ber", "cand_n_claims_comb", "cand_best_ber", "cand_best_comb", "is_top_claim_ber", "is_top_claim_comb"]


def load_union(a_path, b_path, judge_fill=None):
    a = pd.read_parquet(a_path).rename(columns={"rank": "rank_a"})
    b = pd.read_parquet(b_path).rename(columns={"rank": "rank_b", "selected": "selected_b", "cand_addr_missing": "cam_b"})
    a["in_a"] = 1; b["in_b"] = 1
    lab = "label" in a.columns
    if lab:
        b = b.rename(columns={"label": "label_b"})
    u = a.merge(b, on=["s1_id", "cand_id"], how="outer")
    u["in_a"] = u.in_a.fillna(0).astype("int8"); u["in_b"] = u.in_b.fillna(0).astype("int8")
    u["cand_addr_missing"] = u.cand_addr_missing.fillna(u.cam_b).fillna(0).astype("int8")
    u["from_namesearch"] = u.from_namesearch.fillna(0).astype("int8")
    u["selected_b"] = u.selected_b.fillna(0).astype("int8")
    if lab:
        u["label"] = u.label.fillna(u.label_b).astype("int8")
        u = u.drop(columns=["label_b"])
    if judge_fill:        # our judge's score for the pairs only the other pipeline proposed
        j = pd.read_parquet(judge_fill, columns=["s1_id", "cand_id", "ce_score"])
        u = u.merge(j, on=["s1_id", "cand_id"], how="left")
        n0 = int(u.judge_prob.isna().sum())
        u["judge_prob"] = u.judge_prob.fillna(u.ce_score)
        u["combiner_prob"] = u.combiner_prob.fillna(u.judge_prob)       # our best available score for the pair
        print(f"[union] judge score filled for {n0 - int(u.judge_prob.isna().sum()):,} pairs; still without one: {int(u.judge_prob.isna().sum()):,}")
        u = u.drop(columns=["ce_score"])
    return u.drop(columns=["cam_b"])


def add_features(u):
    """Pair, S1-level and candidate-level context. A missing probability means that model
    never proposed the pair; it is kept as NaN for the stacker and treated as 0 in aggregates."""
    ber0 = u.ber_prob.fillna(0.0); comb0 = u.combiner_prob.fillna(0.0)
    u["mean_prob"] = (ber0 + comb0) / 2
    u["min_prob"] = np.minimum(ber0, comb0); u["max_prob"] = np.maximum(ber0, comb0)
    u["abs_diff"] = (ber0 - comb0).abs()
    g = u.groupby("s1_id", sort=False)
    u["s1_max_ber"] = g.ber_prob.transform("max").fillna(0); u["s1_max_comb"] = g.combiner_prob.transform("max").fillna(0)
    u["s1_n_cands"] = g.cand_id.transform("size")
    u["s1_n_ber_hi"] = (ber0 >= 0.5).groupby(u.s1_id, sort=False).transform("sum")
    u["s1_n_comb_hi"] = (comb0 >= 0.5).groupby(u.s1_id, sort=False).transform("sum")
    u["gap_to_s1_best_ber"] = u.s1_max_ber - ber0; u["gap_to_s1_best_comb"] = u.s1_max_comb - comb0
    c = u.groupby("cand_id", sort=False)
    u["cand_n_claims_ber"] = (ber0 >= 0.5).groupby(u.cand_id, sort=False).transform("sum")
    u["cand_n_claims_comb"] = (comb0 >= 0.5).groupby(u.cand_id, sort=False).transform("sum")
    u["cand_best_ber"] = c.ber_prob.transform("max").fillna(0); u["cand_best_comb"] = c.combiner_prob.transform("max").fillna(0)
    u["is_top_claim_ber"] = (ber0 >= u.cand_best_ber - 1e-6).astype("int8")
    u["is_top_claim_comb"] = (comb0 >= u.cand_best_comb - 1e-6).astype("int8")
    return u


def scorer_for(u, col, truth, assign=True):
    d = pd.DataFrame({"s1_id": u.s1_id.to_numpy(), "cand_id": u.cand_id.to_numpy(),
                      "rank": np.arange(len(u)), "score": u[col].fillna(0).to_numpy(dtype=np.float32),
                      "addr2_missing": u.cand_addr_missing.to_numpy()})
    return Scorer(d, truth, score_col="score", assign=assign)


def honest(sc, label):
    half = np.array([zlib.crc32(f"half:{e}".encode()) % 2 for e in sc.ent])
    A, B = half == 0, half == 1
    sA, pA = grid(sc, A); sB, pB = grid(sc, B)
    hB = sc.macro(sc.keep_mask(**pA), B); hA = sc.macro(sc.keep_mask(**pB), A)
    sAll, pAll = grid(sc, None)
    keep = sc.keep_mask(**pAll)
    f, n_pred, tp = sc.f05_per_entity(keep)
    single = sc.n_true == 0
    print(f"{label:<46} honest {(hA+hB)/2:.4f} | in-sample {sAll:.4f} | bars {pAll} | singletons empty {(n_pred[single]==0).mean():.4f} "
          f"| false merges {(n_pred-tp).sum():,} | missed {(sc.n_true-tp).sum():,}", flush=True)
    return (hA + hB) / 2, pAll, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=str(WORK_DIR / "ensemble" / "model_a" / "fold0_pair_probs.parquet"))
    ap.add_argument("--theirs", default=str(WORK_DIR / "ensemble" / "model_b" / "fold0_pair_probs.parquet"))
    ap.add_argument("--ground-truth", default=str(DATA_DIR / "train" / "train_ground_truth.tsv"))
    ap.add_argument("--out-dir", default=str(WORK_DIR / "ensemble"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=400)
    args = ap.parse_args()
    out = Path(args.out_dir)

    u = load_union(args.ours, args.theirs)
    gt = read_tsv(args.ground_truth)
    a_ent = set(pd.read_parquet(args.ours, columns=["s1_id"]).s1_id.unique())
    truth = {e: set(x for x in l.split(",") if x) for e, l in zip(gt.source1_entity_id, gt.matched_entity_ids) if e in a_ent}
    # entities with no candidate in either model still count: add them through the truth dict
    n_links = sum(len(v) for v in truth.values())
    print(f"union pairs {len(u):,} | only ours {((u.in_a==1)&(u.in_b==0)).sum():,} | only theirs {((u.in_a==0)&(u.in_b==1)).sum():,} | both {((u.in_a==1)&(u.in_b==1)).sum():,}")
    print(f"true links {n_links:,} | candidate recall: ours {u[(u.in_a==1)].label.sum()/n_links:.4f}, theirs {u[(u.in_b==1)].label.sum()/n_links:.4f}, union {u.label.sum()/n_links:.4f}")
    both = u[(u.in_a == 1) & (u.in_b == 1)]
    print(f"on shared pairs: corr(ber, combiner) {both.ber_prob.corr(both.combiner_prob):.4f} | disagree (one >=0.5, other <0.5) {((both.ber_prob>=.5)!=(both.combiner_prob>=.5)).mean():.4%}")
    dis = both[(both.ber_prob >= .5) != (both.combiner_prob >= .5)]
    print(f"  in disagreements, theirs is right {((dis.ber_prob>=.5)==(dis.label==1)).mean():.3f}, ours is right {((dis.combiner_prob>=.5)==(dis.label==1)).mean():.3f}  (n={len(dis):,})")
    u = add_features(u)

    # entities present in neither candidate set get no rows; Scorer only knows entities with rows,
    # so report their cost explicitly
    ent_rows = set(u.s1_id.unique())
    lost = [e for e in truth if e not in ent_rows]
    lost_with_truth = sum(1 for e in lost if truth[e])
    print(f"fold-0 entities without any candidate row: {len(lost)} ({lost_with_truth} have true matches; singletons among them score 1.0)")
    N_ALL = len(truth)
    def full_macro(f_rows):      # macro over all 442,001 entities
        return (f_rows.sum() + (len(lost) - lost_with_truth)) / N_ALL

    print("\n=== baselines (one evaluator, all fold-0 entities) ===")
    scb = scorer_for(u, "ber_prob", truth)
    # their shipped rule: the `selected` flag, put in the same (s1_id, row) order the Scorer uses
    d = pd.DataFrame({"s1_id": u.s1_id.to_numpy(), "rank": np.arange(len(u)), "sel": u.selected_b.to_numpy()}).sort_values(["s1_id", "rank"])
    fsel, npred, tp = scb.f05_per_entity(d.sel.to_numpy() == 1)
    print(f"{'theirs, shipped rule (selected flag)':<46} macro F0.5 {full_macro(fsel):.4f} | false merges {(npred-tp).sum():,} | missed {(scb.n_true-tp).sum():,}")
    res = {}
    res["theirs + our decision rule"] = honest(scb, "theirs (ber_prob) + assignment + bars")
    sca = scorer_for(u, "combiner_prob", truth)
    res["ours"] = honest(sca, "ours (combiner) + assignment + bars")
    f_ours = res["ours"][2]; f_theirs = fsel

    print("\n=== error overlap per entity (ours at its bars, theirs shipped rule) ===")
    po, pt = f_ours >= 0.9999, f_theirs >= 0.9999
    print(f"both perfect {np.mean(po&pt):.4f} | only theirs perfect {np.mean(~po&pt):.4f} | only ours perfect {np.mean(po&~pt):.4f} | neither {np.mean(~po&~pt):.4f}")
    print(f"oracle (best of the two per entity): {np.maximum(f_ours, f_theirs).mean():.4f}")

    print("\n=== simple combinations ===")
    for w in (0.5, 0.7, 0.85):
        u["avg"] = w * u.ber_prob.fillna(0) + (1 - w) * u.combiner_prob.fillna(0)
        res[f"avg w={w}"] = honest(scorer_for(u, "avg", truth), f"weighted average, theirs weight {w}")
    u["geo"] = np.sqrt(u.ber_prob.fillna(0) * u.combiner_prob.fillna(0))
    res["geo"] = honest(scorer_for(u, "geo", truth), "geometric mean (both must agree)")

    print("\n=== learned stacker, 5-fold cross-fitted by S1 entity ===")
    fold = np.array([zlib.crc32(f"stack:{e}".encode()) % args.folds for e in u.s1_id.to_numpy()])
    oof = np.zeros(len(u), dtype=np.float32)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, verbose=-1, num_threads=0)
    X = u[FEATS]; y = u.label.to_numpy()
    for k in range(args.folds):
        tr, te = fold != k, fold == k
        m = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=args.rounds)
        oof[te] = m.predict(X[te])
        print(f"  fold {k}: trained on {tr.sum():,} pairs, scored {te.sum():,}", flush=True)
    u["stack"] = oof
    res["stack"] = honest(scorer_for(u, "stack", truth), "stacker (out of fold) + assignment + bars")
    res["stack no assign"] = honest(scorer_for(u, "stack", truth, assign=False), "stacker (out of fold), no assignment")

    final = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=args.rounds)
    final.save_model(str(out / "stacker.txt"))
    imp = sorted(zip(FEATS, final.feature_importance("gain")), key=lambda t: -t[1])
    print("\nstacker top features by gain:", ", ".join(f"{c}={g:.0f}" for c, g in imp[:10]))
    summary = {k: {"honest_f05": float(v[0]), "bars": v[1]} for k, v in res.items()}
    summary["theirs shipped"] = {"f05": float(full_macro(fsel))}
    (out / "ensemble_fold0_summary.json").write_text(json.dumps(summary, indent=2))
    u[["s1_id", "cand_id", "label", "ber_prob", "combiner_prob", "judge_prob", "stack", "in_a", "in_b", "cand_addr_missing", "selected_b"]].to_parquet(out / "fold0_stacked.parquet", index=False)
    print("ENSEMBLE_FOLD0_DONE")


if __name__ == "__main__":
    main()
