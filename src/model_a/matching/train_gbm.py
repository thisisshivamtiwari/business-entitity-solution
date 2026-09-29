"""Matching stage 4: LightGBM over the cross-encoder score plus pairwise features.

Trains on the 'gbm' group of training pairs (scored by a cross-encoder that never saw
them) and writes a final `score` column for any scored+featured pairs file.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

KEYS = {"s1_id", "cand_id", "label", "group"}


def feature_cols(df):
    return [c for c in df.columns if c not in KEYS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train pairs parquet with features + ce_score")
    ap.add_argument("--predict", nargs="+", required=True, help="pairs parquets to score")
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tr = pd.read_parquet(args.train)
    if "group" in tr.columns:
        tr = tr[tr.group == "gbm"]
    cols = feature_cols(tr)
    print(f"[gbm] training on {len(tr):,} pairs, {len(cols)} features, positives {tr.label.mean():.3%}")
    ds = lgb.Dataset(tr[cols], label=tr.label)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  seed=args.seed, verbose=-1, num_threads=0)
    model = lgb.train(params, ds, num_boost_round=args.rounds)
    model.save_model(args.model_out)
    imp = sorted(zip(cols, model.feature_importance("gain")), key=lambda x: -x[1])
    Path(args.model_out).with_suffix(".importance.json").write_text(json.dumps(
        [(c, float(g)) for c, g in imp], indent=1))
    print("[gbm] top features by gain:", ", ".join(f"{c}={g:.0f}" for c, g in imp[:8]))
    for p in args.predict:
        df = pd.read_parquet(p)
        df["score"] = model.predict(df[cols]).astype(np.float32)
        out = Path(p).with_name(Path(p).stem.replace("_feat", "") + "_scored.parquet")
        df[["s1_id", "cand_id", "rank", "score"] + (["label"] if "label" in df else []) +
           ["addr2_missing", "ce_score", "cos_full_text"]].to_parquet(out, index=False)
        print(f"[gbm] scored {len(df):,} pairs -> {out}")


if __name__ == "__main__":
    main()
