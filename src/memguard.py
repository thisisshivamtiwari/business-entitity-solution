"""Memory watchdog for heavy pipeline steps.

A background thread checks this process's resident memory and the system's swap usage every
second. If either passes its limit, the step exits with code 3 *before* the machine runs out of
memory (an earlier unguarded script shut the whole computer down). Finished steps keep their
outputs, so the pipeline can be resumed with `run_pipeline.py --from <step>`.

Usage (first line of a step's main):
    from memguard import start_memory_guard
    start_memory_guard("block")
"""
import os
import subprocess
import threading
import time

MEM_LIMIT_GB = float(os.environ.get("BER_MEM_LIMIT_GB", 30))   # the dev machine has 48 GB
SWAP_LIMIT_GB = float(os.environ.get("BER_SWAP_LIMIT_GB", 2))


def _rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip() or 0) / 1048576


def _swap_gb():
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        return float(out.split("used = ")[1].split("M")[0]) / 1024
    except (IndexError, ValueError, FileNotFoundError):   # non-macOS: no swap check
        return 0.0


def start_memory_guard(step):
    swap_at_start = _swap_gb()
    peak = {"gb": 0.0}

    def watch():
        while True:
            rss, swap = _rss_gb(), _swap_gb() - swap_at_start
            peak["gb"] = max(peak["gb"], rss)
            if rss > MEM_LIMIT_GB or swap > SWAP_LIMIT_GB:
                print(f"!! MEMORY GUARD: '{step}' stopped at {rss:.1f} GB (new swap {swap:.1f} GB). "
                      f"Limits: {MEM_LIMIT_GB} GB / {SWAP_LIMIT_GB} GB swap.", flush=True)
                os._exit(3)
            time.sleep(1)

    threading.Thread(target=watch, daemon=True).start()
    return peak
