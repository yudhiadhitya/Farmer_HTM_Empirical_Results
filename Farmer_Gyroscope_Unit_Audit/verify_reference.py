"""Verify archived neural weights against archived probabilities using original quaternion inputs."""
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.environ["FARMER_BENCHMARK_ROOT"] = str(BASE / "original")
os.environ["FARMER_BENCHMARK_RUNS"] = str(BASE / "original/empirical_runs")
sys.path.insert(0, str(BASE / "original/empirical_experiment_tools"))
import numpy as np
import torch
import run_baselines as models


def main():
    test = models.select(models.load_records(), "test")
    rows = []
    for kind in ["CNN", "LSTM", "Transformer"]:
        for seed in [7, 42, 2026]:
            folder = BASE / f"original/empirical_runs/baselines/{kind}/seed_{seed}"
            state = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
            model = getattr(models, kind)()
            model.load_state_dict(state["state_dict"])
            prob = models.probabilities(model, test, state["mean"], state["std"])
            with np.load(folder / "test_predictions.npz") as z:
                saved = z["probabilities"]
                diff = float(np.max(np.abs(prob - saved)))
            assert np.array_equal(prob.argmax(1), saved.argmax(1))
            assert diff < 1e-6
            rows.append({"model": kind, "seed": seed, "maximum_probability_difference": diff,
                         "all_predicted_labels_identical": True})
            print(rows[-1], flush=True)
    (BASE / "work").mkdir(exist_ok=True)
    (BASE / "work/original_checkpoint_verification.json").write_text(
        json.dumps({"original_neural_checkpoint_verification": rows}, indent=2))


if __name__ == "__main__":
    main()
