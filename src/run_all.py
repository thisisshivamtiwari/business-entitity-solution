"""One command for the whole pipeline: Model B -> Model A -> stacker -> output/*.tsv (test macro F0.5 0.9866).

    python run_all.py                 # B1 ... V, in order (the embeddings must exist: README, "Embeddings")
    python run_all.py --from A1       # resume at a stage (earlier outputs must exist)
    python run_all.py --list          # print the stages

Stages (each is a separate process; the run stops at the first failure):
    G1  encoder v1 and training embeddings for Model B (GPU) -> Embeddings/   [not in the default run: --from G1]
    B1  Model B experiment on folds 1-4, scored on fold 0 in both validation worlds -> threshold (work/results/)
    B2  Model B matcher scores ALL fold-0 S1 (trained on folds 1-4 only, so the scores are out of sample)
    B3  export Model B fold-0 pair probabilities           -> work/ensemble/model_b/fold0_pair_probs.parquet
    B4  Model B final run: retrain on all training data, score the full test set
    B5  export Model B test pair probabilities             -> work/ensemble/model_b/test_pair_probs.parquet
    A1  Model A (cross-encoder judge + combiner)           -> work/ensemble/model_a/{fold0,test}_pair_probs.parquet
    E1  stacker: cross-fit on fold 0, fit on all of fold 0, score the test union -> work/ensemble/test_stacked.parquet
    E2  decision rule (bars 0.70 / 0.70, delta 0.2)        -> output/matching_results.tsv, output/candidate_pairs.tsv
    V   format and consistency check of the two output files (check_outputs.py)

B2 must run before B4: the final run replaces the fold 1-4 ranker's candidates with the all-data ranker's.
A1 needs a CUDA GPU (src/model_a/README.md). After G1, build the test embeddings and neighbour lists with src/gpu/.
"""
import argparse
import os
import subprocess
import sys
import time

from config import DATA_DIR, OUT_DIR, WORK_DIR

SRC = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TEST_S1 = str(DATA_DIR / "test" / "test_source1.tsv")

# Model B's final configuration (iter10a_emb): key caps 50/200, cluster second pass, name-only search, rescue 0.5, embeddings.
MODEL_B_ENV = {"BER_CAP": "50", "BER_CAP23": "200", "BER_CLUSTER": "1", "BER_NAMESEARCH": "1", "BER_RESCUE": "0.5",
               "BER_EMB": "1", "BER_MEM_LIMIT_GB": "34", "POLARS_MAX_THREADS": "10"}
MODEL_B_OUT = str(WORK_DIR / "model_b_output")      # Model B's standalone result files (not the final output)

STAGES = [
    ("G1", "encoder v1 + training embeddings (GPU)", ["encoder_v1/run_encoder_v1.py"], {}),
    ("B1", "Model B experiment (folds 1-4, fold 0 scored)", ["run_pipeline.py", "--experiment", "--tag", "iter10a_emb"], MODEL_B_ENV),
    ("B2", "Model B scores all fold-0 S1", ["train.py", "400000", "442001"], MODEL_B_ENV),
    ("B3", "export Model B fold-0 pair probabilities", ["export_pair_probs.py", "fold0"], MODEL_B_ENV),
    ("B4", "Model B final run (all training data -> test)", ["run_pipeline.py", "--final", "iter10a_emb"],
     {**MODEL_B_ENV, "BER_OUT": MODEL_B_OUT}),
    ("B5", "export Model B test pair probabilities", ["export_pair_probs.py", "test"], MODEL_B_ENV),
    ("A1", "Model A: judge + combiner, fold 0 and test", ["model_a/run_model_a.py"], {}),
    ("E1", "stacker: cross-fit on fold 0, score the test union", ["ensemble/ensemble_test.py"], {}),
    ("E2", "decision rule -> output/", ["ensemble/ensemble_apply.py", "--stacked", str(WORK_DIR / "ensemble" / "test_stacked.parquet"),
                                        "--test-s1", TEST_S1, "--out-dir", str(OUT_DIR),
                                        "--t-abs", "0.70", "--t-noaddr", "0.70", "--delta", "0.2"], {}),
    ("V", "check the two output files", ["check_outputs.py"], {}),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default="B1", choices=[s[0] for s in STAGES])
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for code, name, cmd, _ in STAGES:
            print(f"{code:<3} {name}: {' '.join(cmd)}")
        return
    todo = STAGES[[s[0] for s in STAGES].index(args.start):]
    missing = [c[0] for _, _, c, _ in todo if c[0].endswith(".py") and not os.path.isabs(c[0]) and not os.path.exists(os.path.join(SRC, c[0]))]
    if missing:
        sys.exit(f"missing scripts: {', '.join(missing)}")
    started = time.time()
    for code, name, cmd, env in todo:
        print(f"\n=== [{code}] {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        rc = subprocess.run([PY, *cmd], cwd=SRC, env={**os.environ, **env}).returncode
        print(f"=== [{code}] {'OK' if rc == 0 else f'FAILED (exit code {rc})'} in {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            sys.exit(rc)
    print(f"\nAll stages done in {(time.time() - started) / 60:.1f} min. Result: {OUT_DIR / 'matching_results.tsv'}")


if __name__ == "__main__":
    main()
