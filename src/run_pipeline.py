"""Run the whole pipeline with one command: raw TSVs -> validated result files.

    cd code/business_entity_resolution/src
    ../.venv/bin/python run_pipeline.py --tag iter3            # everything (~35 min)
    ../.venv/bin/python run_pipeline.py --from train          # resume from a step
    ../.venv/bin/python run_pipeline.py --only predict validate
    ../.venv/bin/python run_pipeline.py --experiment --tag X  # train + validation only, no test set
    ../.venv/bin/python run_pipeline.py --final iter8_cap50    # FINAL: all training data -> test set
        (use the same BER_* settings as the chosen experiment; see EXPERIMENTS.md)

Steps (each runs in its own process, so memory is freed in between; the run stops at the
first failure):

    convert       TSV -> Parquet
    translit      learn the Indian-script -> English word dictionary (+ spelling variants)
    normalize     clean names / addresses, learn native-script region names
    block         compound-key candidate search, top-100 raw candidates per S1
    rerank_fit    learn a better candidate ranking from the training labels
    rerank_apply  keep the best 12 candidates per S1
    train         pair features + gradient-boosted matcher + threshold on validation fold
    predict       score test candidates, write output/*.tsv
    validate      format check (check_outputs.py)
    score         validation F0.5 in the ordinary world and in the test-like world (as many
                  decoys per S1 as test; see testlike.py), saved to work/results/<tag>__*.json
    finalize      re-select the test matches at the test-like world's best threshold, re-validate

Result: <project>/output/matching_results.tsv   <- the final matches
        <project>/output/candidate_pairs.tsv    (the blocking candidate set)
Peak memory is ~36 GB (blocking); tested on a 48 GB Apple M4 Pro.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from config import DATA_DIR, OUT_DIR

SRC_DIR = Path(__file__).resolve().parent
PY = sys.executable
VALIDATOR = Path(__file__).resolve().parent / "check_outputs.py"

STEPS = [
    ("convert", ["convert.py"]),
    ("translit", ["translit.py", "learn"]),
    ("normalize", ["normalize.py", "train", "test"]),
    ("block", ["block.py", "train", "test"]),
    ("rerank_fit", ["rerank.py", "fit"]),
    ("rerank_apply", ["rerank.py", "apply", "train", "test"]),
    ("namesearch", ["namesearch.py", "train", "test"]),
    ("train", ["train.py", "400000", "150000"]),
    ("predict", ["predict.py"]),
    ("validate", [str(VALIDATOR)]),
    ("score", None),   # expanded in main(): validate.py in both worlds
    ("finalize", None),  # expanded in main(): threshold from the test-like world + format check
]
STEP_NAMES = [name for name, _ in STEPS]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=STEP_NAMES, default=STEP_NAMES[0],
                        help="start at this step (earlier outputs must already exist)")
    parser.add_argument("--only", nargs="+", choices=STEP_NAMES, help="run just these steps")
    parser.add_argument("--tag", default="latest", help="name under which validation scores are saved")
    parser.add_argument("--final", metavar="TAG",
                        help="final run: retrain dictionary, ranker and matcher on ALL training "
                             "data, threshold from experiment TAG's test-like score, then predict the test set")
    parser.add_argument("--experiment", action="store_true",
                        help="train and score on validation only: skip every test-set step (faster); "
                             "run without it once the final configuration is chosen")
    args = parser.parse_args()
    selected = args.only or STEP_NAMES[STEP_NAMES.index(args.start):]
    if args.final:
        return run_final(args.final)

    started = time.time()
    test_steps = {"predict", "validate", "finalize"}
    for name, cmd in STEPS:
        if name not in selected or (args.experiment and name in test_steps):
            continue
        if args.experiment and cmd and cmd[0] in ("block.py", "normalize.py"):
            cmd = [c for c in cmd if c != "test"]                  # train split only
        if args.experiment and cmd and cmd[0] == "namesearch.py":
            cmd = [c for c in cmd if c != "test"]
        if args.experiment and cmd and cmd[:2] == ["rerank.py", "apply"]:
            cmd = [c for c in cmd if c != "test"]
        if name == "score":   # ordinary world, then the test-like world
            cmds = [["validate.py", args.tag, "train"], ["testlike.py"], ["block.py", "trainlike"],
                    ["rerank.py", "apply", "trainlike"], ["namesearch.py", "trainlike"],
                    ["validate.py", args.tag, "trainlike"]]
        elif name == "finalize":
            cmds = [["predict.py", "reselect"], dict(STEPS)["validate"]]
        else:
            cmds = [cmd]
        print(f"\n=== [{name}] {' | '.join(' '.join(c) for c in cmds)}", flush=True)
        t0 = time.time()
        code = 0
        for c in cmds:
            code = subprocess.run([PY, *c], cwd=SRC_DIR).returncode
            if code != 0:
                break
        status = "OK" if code == 0 else f"FAILED (exit code {code})"
        print(f"=== [{name}] {status} in {time.time() - t0:.0f}s", flush=True)
        if code != 0:
            sys.exit(code)

    print(f"\nPipeline finished in {(time.time() - started) / 60:.1f} min.")
    print(f"Matches: {OUT_DIR / 'matching_results.tsv'}")


def run_final(tag):
    """All training data -> models -> full test set. Every step is a separate, memory-guarded process."""
    env = {**os.environ, "BER_FINAL": "1"}
    steps = [
        ("translit (all folds)", ["translit.py", "learn"]),
        ("normalize", ["normalize.py", "train", "test"]),
        ("block", ["block.py", "train", "test"]),
        ("ranker (all folds)", ["final.py", "ranker"]),
        ("rerank_apply", ["rerank.py", "apply", "train", "test"]),
        ("namesearch", ["namesearch.py", "train", "test"]),
        ("matcher (all training S1)", ["final.py", "matcher", tag]),
        ("predict (full test set)", ["predict.py"]),
        ("validate", dict(STEPS)["validate"]),
    ]
    if os.environ.get("BER_TRAIN_WORLD", "train") == "trainlike":
        # matcher trained in the test-like world: rebuild that world and its candidates with the final ranker
        names = [n for n, _ in steps]
        steps.insert(names.index("block") + 1, ("test-like world", ["testlike.py"]))
        steps.insert(names.index("block") + 2, ("block trainlike", ["block.py", "trainlike"]))
        steps = [(n, c + ["trainlike"]) if n in ("rerank_apply", "namesearch") else (n, c) for n, c in steps]
    started = time.time()
    for name, cmd in steps:
        print(f"\n=== [final: {name}] {' '.join(cmd)}", flush=True)
        t0 = time.time()
        code = subprocess.run([PY, *cmd], cwd=SRC_DIR, env=env).returncode
        print(f"=== [final: {name}] {'OK' if code == 0 else f'FAILED (exit code {code})'} in {time.time() - t0:.0f}s",
              flush=True)
        if code != 0:
            sys.exit(code)
    print(f"\nFinal run finished in {(time.time() - started) / 60:.1f} min. "
          f"Matches: {OUT_DIR / 'matching_results.tsv'}")


if __name__ == "__main__":
    main()
