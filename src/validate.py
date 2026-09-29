"""Score the trained matcher on the validation fold in a given "world" and record the result.

    python validate.py <tag> [world]        world = train (ordinary validation) or trainlike

The validation S1 sample is the same as in train.py (fold 0, 150k, same seed). For world=train
this reproduces train.py's validation score; for world=trainlike it measures the same models
under test-like decoy density. Results go to work/results/<tag>__<world>.json so iterations can be
compared side by side (see compare_results()).
"""
import json
import pickle
import sys
import time

import numpy as np
import polars as pl

import features as F
from config import SEED, WORK_DIR
from evaluate import f05, per_s1_f05, truth_rids
from memguard import start_memory_guard
from train import featurize, load_candidates, predict_proba, select

N_VAL = 150_000   # must match train.py
RESULTS = WORK_DIR / "results"


def main(tag, world="train"):
    t0 = time.time()
    with open(WORK_DIR / "model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, feat_cols, thr = bundle["model"], bundle["feat_cols"], bundle["thr"]
    truth, s1 = truth_rids()
    val_s1 = s1.filter(pl.col("fold") == 0).sample(N_VAL, seed=SEED)
    vtruth = truth.join(val_s1, on="r1", how="semi")

    cand = load_candidates(world).join(val_s1.select("r1"), on="r1", how="semi")
    norm = F.load_norm(world)
    x = featurize(cand, norm, F.idf_table(norm, "core"), F.idf_table(norm, "addr"))
    pred = x.select("r1", "r2").with_columns(p=predict_proba(bundle, x, norm, world))
    del x, norm

    curve = {float(round(t, 2)): f05(select(pred, t), vtruth, val_s1) for t in np.arange(0.30, 0.96, 0.05)}
    best_thr = max(curve, key=curve.get)
    chosen = select(pred, thr)
    per = per_s1_f05(chosen, vtruth, val_s1)
    tp = int(per["tp"].sum())
    result = {
        "tag": tag, "world": world, "threshold_used": round(thr, 2),
        "f05": per["f"].mean(), "precision": tp / max(int(per["np"].sum()), 1), "recall": tp / vtruth.height,
        "false_matches": int(per["np"].sum()) - tp,
        "candidate_recall": pred.join(vtruth, on=["r1", "r2"], how="semi").height / vtruth.height,
        "candidates_per_S1": pred.height / val_s1.height,
        "best_threshold": best_thr, "f05_at_best_threshold": curve[best_thr], "curve": curve,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}__{world}.json").write_text(json.dumps(result, indent=1))
    print(f"[{tag} | {world}] F0.5 {result['f05']:.4f} at thr {thr:.2f} (best {curve[best_thr]:.4f} at "
          f"{best_thr:.2f}) | precision {result['precision']:.4f} recall {result['recall']:.4f} | "
          f"false matches {result['false_matches']:,} | candidate recall {result['candidate_recall']:.4f} "
          f"({time.time() - t0:.0f}s)")


def compare_results():
    """Table of every recorded (iteration, world) result."""
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*__*.json"))]
    return pl.DataFrame([{k: r[k] for k in ("tag", "world", "f05", "precision", "recall", "false_matches",
                                          "candidate_recall", "threshold_used", "best_threshold",
                                          "f05_at_best_threshold")} for r in rows])


if __name__ == "__main__":
    start_memory_guard("validate")
    main(*sys.argv[1:])
