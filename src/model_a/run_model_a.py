"""Model A end to end: blocking encoder -> exact top-k candidates -> cross-encoder judge -> LightGBM combiner.

    python model_a/run_model_a.py            # all stages; finished stages are skipped on a re-run

Writes, for the stacker:
    work/ensemble/model_a/fold0_pair_probs.parquet   s1_id, cand_id, rank, judge_prob, combiner_prob, blocking_cosine, cand_addr_missing, label
    work/ensemble/model_a/test_pair_probs.parquet    same, without label

Every model is trained on folds 1-4 only. Fold 0 (S1 with polars hash(entity_id, seed=7) % 5 == 0,
442,001 S1) and the test set are only scored. Intermediate files go to work/model_a/.

Needs a CUDA GPU. NGPU (default: all visible GPUs) spreads encoding, candidate search, judge training
and judge scoring over the GPUs. The reported result was produced with 2 x RTX 4090 (24 GB each).
The settings below are the ones behind the reported result; the environment variables exist for small test runs.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from config import DATA_DIR, WORK_DIR  # noqa: E402

PY = sys.executable
B, M = HERE / "blocking", HERE / "matching"
W = WORK_DIR / "model_a"
OUT = WORK_DIR / "ensemble" / "model_a"
GT = DATA_DIR / "train" / "train_ground_truth.tsv"
E = os.environ.get


def gpu_count():
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return len([l for l in r.stdout.splitlines() if l.strip()]) if r.returncode == 0 else 0
    except FileNotFoundError:
        return 0


NGPU = int(E("NGPU") or gpu_count())
# ---- blocking (encoder v2: team text recipe, two views)
ENC_BASE = E("ENC_BASE", "intfloat/multilingual-e5-small")
ENC_STEPS, ENC_CLUSTERS, ENC_HARD_FRAC = E("ENC_STEPS", "4000"), E("ENC_CLUSTERS", "128"), "0.4"
K, NAME_K, EXACT_NAME_CAP = E("K", "25"), E("NAME_K", "10"), "30"
# ---- judge
CE_MODEL = E("CE_MODEL", "FacebookAI/xlm-roberta-base")
CE_EPOCHS, CE_TOTAL_BATCH, CE_LR, CE_MAX_LEN = E("CE_EPOCHS", "2"), int(E("CE_TOTAL_BATCH", "192")), "2e-5", E("CE_MAX_LEN", "96")
CE_NEG_PER_POS, CE_MAX_PAIRS, CE_AUGMENT_P, CE_ADDR_DROP_POS = 4, int(E("CE_MAX_PAIRS", "2400000")), "0.3", "0.30"
SCORE_BATCH = E("SCORE_BATCH", "512")
# ---- combiner
GBM_SHARE, GBM_ROUNDS = "0.05", E("GBM_ROUNDS", "600")
WORKERS = E("WORKERS", str(os.cpu_count() or 8))


def log(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, env=None):
    cmd = [str(c) for c in cmd]
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})})


def run_shards(make_cmd):
    """One process per GPU; make_cmd(i) -> command for shard i."""
    procs = [subprocess.Popen([str(c) for c in make_cmd(i)],
                              env={**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(i)}) for i in range(NGPU)]
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f"a shard failed: exit codes {codes}")


def encode(split):
    if (W / f"{split}_name_text.npy").exists() and _encoded(split):
        return print(f"  {split} embeddings exist, skipping")
    base = [PY, B / "encode.py", "--work-dir", W, "--split", split, "--model-dir", W / "sc_encoder"]
    if NGPU >= 2:
        cfg = json.loads((W / "sc_encoder" / "1_Pooling" / "config.json").read_text())
        run(base + ["--init-only", cfg.get("embedding_dimension") or cfg["word_embedding_dimension"]])
        run_shards(lambda i: base + ["--shard", f"{i}/{NGPU}"])
    else:
        run(base)


def _encoded(split):
    import pandas as pd
    n = len(pd.read_parquet(W / f"{split}_ids.parquet"))
    chunks = -(-n // 200_000)
    return all((W / f"{split}_{v}.done").exists() and len((W / f"{split}_{v}.done").read_text().split()) >= chunks
               for v in ("full_text", "name_text"))


def retrieve(out, split, s1_ids=None):
    if out.exists():
        return print(f"  {out.name} exists, skipping")
    args = ["--work-dir", W, "--split", split, "--k", K, "--name-k", NAME_K, "--exact-name-cap", EXACT_NAME_CAP]
    if s1_ids:
        args += ["--s1-ids", s1_ids]
    if NGPU >= 2:
        run_shards(lambda i: [PY, B / "retrieve.py", "--out", f"{out}.part{i}", "--shard", f"{i}/{NGPU}"] + args)
        with open(out, "w", encoding="utf-8") as f:
            for i in range(NGPU):
                with open(f"{out}.part{i}", encoding="utf-8") as p:
                    head = p.readline()
                    if i == 0:
                        f.write(head)
                    for line in p:
                        f.write(line)
                os.remove(f"{out}.part{i}")
                Path(f"{out}.part{i}").with_suffix(".stats.json").unlink(missing_ok=True)
    else:
        run([PY, B / "retrieve.py", "--out", out] + args)


def score(pairs, out, split, only_group=None):
    if out.exists():
        return print(f"  {out.name} exists, skipping")
    args = ["score", "--work-dir", W, "--split", split, "--pairs", pairs, "--model-dir", W / "cross_encoder",
            "--batch-size", SCORE_BATCH, "--max-len", CE_MAX_LEN] + (["--only-group", only_group] if only_group else [])
    if NGPU >= 2:
        import pandas as pd
        parts = [Path(f"{str(out)[:-8]}.part{i}.parquet") for i in range(NGPU)]
        run_shards(lambda i: [PY, M / "cross_encoder.py"] + args + ["--out", parts[i], "--shard", f"{i}/{NGPU}"])
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(out, index=False)
        for p in parts:
            p.unlink()
    else:
        run([PY, M / "cross_encoder.py"] + args + ["--out", out])


def features(pairs, out, split):
    if not out.exists():
        run([PY, M / "features.py", "--work-dir", W, "--split", split, "--pairs", pairs, "--out", out, "--workers", WORKERS])


def export(scored, out):
    import pandas as pd
    d = pd.read_parquet(scored).rename(columns={"ce_score": "judge_prob", "score": "combiner_prob",
                                                "addr2_missing": "cand_addr_missing", "cos_full_text": "blocking_cosine"})
    d["cand_addr_missing"] = d.cand_addr_missing.fillna(0).astype("int8")
    cols = ["s1_id", "cand_id", "rank", "judge_prob", "combiner_prob", "blocking_cosine", "cand_addr_missing"]
    d[cols + (["label"] if "label" in d else [])].to_parquet(out, index=False)
    print(f"  {len(d):,} pairs, {d.s1_id.nunique():,} S1 -> {out}")


def main():
    import pandas as pd
    W.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    print(f"Model A | GPUs {NGPU} | data {DATA_DIR} | work {W}")
    t0 = time.time()

    log("1/12 clean the raw files, ground-truth links, fold 0")
    run([PY, B / "prepare.py", "--data-dir", DATA_DIR, "--work-dir", W, "--splits", "train", "test"])

    log("2/12 blocking encoder: supervised contrastive fine-tuning on folds 1-4")
    if not (W / "sc_encoder" / "config.json").exists():
        run([PY, B / "train_sc.py", "--work-dir", W, "--base-model", ENC_BASE, "--out-dir", W / "sc_encoder",
             "--steps", ENC_STEPS, "--clusters-per-batch", ENC_CLUSTERS, "--hard-frac", ENC_HARD_FRAC])

    log("3/12 embeddings of the training records (two views)")
    encode("train")

    log("4/12 candidates: fold 0, then every training S1 of folds 1-4 that has a match")
    retrieve(W / "val_candidates.tsv", "train", W / "val_s1_ids.txt")
    run([PY, B / "evaluate_blocking.py", "--candidates", W / "val_candidates.tsv", "--ground-truth", GT,
         "--records", W / "train_records.parquet", "--show-misses", "0"])
    if not (W / "train_s1_ids.txt").exists():
        p = pd.read_parquet(W / "train_pairs.parquet", columns=["s1_id", "fold"])
        ids = p[p.fold != 0].s1_id.drop_duplicates()
        lim = int(E("TRAIN_S1", "0"))
        (ids.sample(n=lim, random_state=42) if lim and lim < len(ids) else ids).to_csv(W / "train_s1_ids.txt", index=False, header=False)
    retrieve(W / "train_candidates.tsv", "train", W / "train_s1_ids.txt")

    log("5/12 labelled pairs; 5% of the training S1 are kept away from the judge for the combiner")
    if not (W / "train_pairs_m.parquet").exists():
        run([PY, M / "build_pairs.py", "--candidates", W / "train_candidates.tsv", "--ground-truth", GT,
             "--out", W / "train_pairs_m.parquet", "--group-split", GBM_SHARE])
    if not (W / "val_pairs_m.parquet").exists():
        run([PY, M / "build_pairs.py", "--candidates", W / "val_candidates.tsv", "--ground-truth", GT, "--out", W / "val_pairs_m.parquet"])

    log("6/12 judge: fine-tune the cross-encoder")
    if not (W / "cross_encoder" / "config.json").exists():
        p = pd.read_parquet(W / "train_pairs_m.parquet"); p = p[p.group == "ce"]
        pos, neg = p[p.label == 1], p[p.label == 0]
        if len(neg) > CE_NEG_PER_POS * len(pos):
            neg = neg.sample(n=int(CE_NEG_PER_POS * len(pos)), random_state=42)
        p = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True).iloc[:CE_MAX_PAIRS or None]
        p.to_parquet(W / "train_pairs_judge.parquet", index=False)
        print(f"  {len(p):,} training pairs from {p.s1_id.nunique():,} S1, positives {p.label.mean():.1%}")
        del p, pos, neg
        launch = [PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={NGPU}"] if NGPU >= 2 else [PY]
        run(launch + [M / "cross_encoder.py", "train", "--work-dir", W, "--pairs", W / "train_pairs_judge.parquet",
                      "--out-dir", W / "cross_encoder", "--base-model", CE_MODEL, "--epochs", CE_EPOCHS,
                      "--batch-size", CE_TOTAL_BATCH // max(1, NGPU), "--lr", CE_LR, "--max-len", CE_MAX_LEN,
                      "--neg-per-pos", CE_NEG_PER_POS, "--augment-p", CE_AUGMENT_P, "--addr-drop-pos", CE_ADDR_DROP_POS])

    log("7/12 judge scores: combiner group and fold 0")
    score(W / "train_pairs_m.parquet", W / "train_pairs_ce.parquet", "train", only_group="gbm")
    score(W / "val_pairs_m.parquet", W / "val_pairs_ce.parquet", "train")

    log("8/12 pair features")
    features(W / "train_pairs_ce.parquet", W / "train_pairs_feat.parquet", "train")
    features(W / "val_pairs_ce.parquet", W / "val_pairs_feat.parquet", "train")

    log("9/12 combiner (LightGBM) and Model A's own fold-0 score")
    if not (W / "val_pairs_scored.parquet").exists():
        run([PY, M / "train_gbm.py", "--train", W / "train_pairs_feat.parquet", "--predict", W / "val_pairs_feat.parquet",
             "--model-out", W / "gbm.txt", "--rounds", GBM_ROUNDS])
    run([PY, M / "decide.py", "tune", "--search", "grid", "--scored", W / "val_pairs_scored.parquet", "--score-col", "score",
         "--ground-truth", GT, "--records", W / "train_records.parquet", "--params-out", W / "decision_params.json"])
    export(W / "val_pairs_scored.parquet", OUT / "fold0_pair_probs.parquet")

    log("10/12 test set: embeddings and candidates")
    for f in ("train_full_text.npy", "train_name_text.npy", "train_full_text.done", "train_name_text.done"):
        if E("KEEP_TRAIN_EMB") != "1":
            (W / f).unlink(missing_ok=True)          # 19 GB, no longer needed
    encode("test")
    retrieve(W / "test_candidates.tsv", "test")
    if not (W / "test_pairs_m.parquet").exists():
        run([PY, M / "build_pairs.py", "--candidates", W / "test_candidates.tsv", "--out", W / "test_pairs_m.parquet"])

    log("11/12 test set: judge scores and features")
    score(W / "test_pairs_m.parquet", W / "test_pairs_ce.parquet", "test")
    features(W / "test_pairs_ce.parquet", W / "test_pairs_feat.parquet", "test")

    log("12/12 test set: combiner scores, export")
    import lightgbm as lgb
    import numpy as np
    m = lgb.Booster(model_file=str(W / "gbm.txt"))
    d = pd.read_parquet(W / "test_pairs_feat.parquet")
    d["score"] = m.predict(d[m.feature_name()]).astype(np.float32)
    d[["s1_id", "cand_id", "rank", "score", "addr2_missing", "ce_score", "cos_full_text"]].to_parquet(W / "test_pairs_scored.parquet", index=False)
    del d
    export(W / "test_pairs_scored.parquet", OUT / "test_pair_probs.parquet")
    print(f"\nMODEL_A_DONE in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
