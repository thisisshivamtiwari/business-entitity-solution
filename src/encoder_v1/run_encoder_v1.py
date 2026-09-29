"""Build the v1 encoder and the training embeddings that Model B reads.

    python encoder_v1/run_encoder_v1.py      # needs a CUDA GPU; about 12 min training + 40 min encoding on one NVIDIA L4

Writes <project>/Embeddings/: sc_encoder/, train_emb.npy, train_records.parquet, test_records.parquet.
The test embeddings and the neighbour lists are then built with src/gpu/ (see src/gpu/README.md).
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from config import DATA_DIR, PROJECT_DIR  # noqa: E402

EMB = PROJECT_DIR / "Embeddings"


def run(*cmd):
    subprocess.run([sys.executable, *map(str, cmd)], check=True, cwd=HERE)


def main():
    EMB.mkdir(parents=True, exist_ok=True)
    run("prepare_v1.py", "--data-dir", DATA_DIR, "--work-dir", EMB)
    if not (EMB / "sc_encoder" / "config.json").exists():
        run("train_sc_v1.py", "--work-dir", EMB, "--out-dir", EMB / "sc_encoder")
    run("encode_v1.py", "--work-dir", EMB, "--split", "train", "--model-dir", EMB / "sc_encoder")
    print("ENCODER_V1_DONE")


if __name__ == "__main__":
    main()
