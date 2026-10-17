"""Portable reproduction entry point; uses fresh output if --output-dir is supplied."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--htm-workers", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    output = (args.output_dir or base / "work/runs_deg_per_second").resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FARMER_UNIT_RUNS=str(output),
               FARMER_BENCHMARK_ROOT=str(base / "work"),
               FARMER_BENCHMARK_RUNS=str(output), OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="2", OMP_NUM_THREADS="2")
    logs = output / "execution_logs"
    logs.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(base / "document_unit_evidence.py")], env=env, check=True)
    subprocess.run([sys.executable, str(base / "prepare_unit_sensitivity.py")], env=env, check=True)
    subprocess.run([sys.executable, str(base / "verify_reference.py")], env=env, check=True)
    for model in ["classical", "CNN", "LSTM", "Transformer"]:
        with (logs / f"baselines_{model}.log").open("w") as log:
            subprocess.run([sys.executable, str(base / "work/empirical_experiment_tools/run_baselines.py"),
                            "--model", model], env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        print(f"Baseline stage completed: {model}", flush=True)
    subprocess.run([sys.executable, str(base / "run_htm_unit_sensitivity.py"), "--encode-only"], env=env, check=True)
    seeds = [7, 42, 2026]
    for start in range(0, len(seeds), args.htm_workers):
        children = []
        for seed in seeds[start:start + args.htm_workers]:
            log = (logs / f"htm_seed_{seed}.log").open("w")
            p = subprocess.Popen([sys.executable, str(base / "run_htm_unit_sensitivity.py"),
                                  "--seed", str(seed)], env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append((seed, p, log))
        for seed, child, log in children:
            code = child.wait(); log.close()
            if code:
                raise RuntimeError(f"HTM seed {seed} failed; inspect execution_logs")
            print(f"HTM stage completed: seed {seed}", flush=True)
    subprocess.run([sys.executable, str(base / "summarize_unit_sensitivity.py")], env=env, check=True)


if __name__ == "__main__":
    main()
