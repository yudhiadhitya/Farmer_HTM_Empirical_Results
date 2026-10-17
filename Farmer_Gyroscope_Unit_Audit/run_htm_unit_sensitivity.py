"""Matched full-model unit sensitivity; retain the archived validation-selection rule."""
import argparse
import gc
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "work/empirical_experiment_tools"))
from run_htm import (Backbone, PROTOCOL, RUNS, encoding_cache, evaluate,
                     fit_readout, infer, load_records, save_model, select, write_json)


def run_seed(seed, records, encodings):
    root = RUNS / "htm" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "unit_sensitivity_completed.json").exists():
        return
    backbone = Backbone(seed)
    train_seconds = backbone.fit(records, encodings)
    features, extract_seconds = backbone.extract(records, encodings)
    validation = select(records, "validation")
    options = []
    for alpha in PROTOCOL["auxiliary_weights"]:
        fit = fit_readout(records, features, seed, alpha, 65536)
        score = evaluate(validation, infer(validation, features, fit[0], fit[1]))["macro_f1"]
        options.append((score, -alpha, fit))
        print(f"HTM sensitivity seed {seed}, alpha {alpha}: validation macro-F1 {score:.8f}", flush=True)
    chosen = max(options, key=lambda a: (a[0], a[1]))
    selected_alpha = -chosen[1]
    extra = {"backbone_training_seconds": train_seconds,
             "frozen_feature_extraction_seconds": extract_seconds,
             "validation_auxiliary_selection": [{"alpha": -x[1], "validation_macro_f1": x[0]} for x in options],
             "native_backbone_hz": 50, "readout_and_evaluation_hz": 10,
             "unit_interpretation": "CSV-header hypothesis: source gyro values in deg/s, multiplied by pi/180",
             "selection_uses_test_labels": False}
    for _, negative_alpha, fit in options:
        alpha = -negative_alpha
        save_model(root / f"aux_{alpha:g}", f"HTM auxiliary weight {alpha:g}", seed, alpha,
                   records, features, fit, 65536, extra=extra)
    save_model(root / "full", "Full HTM (validation-selected weight)", seed, selected_alpha,
               records, features, chosen[2], 65536, extra=extra)
    write_json(root / "unit_sensitivity_completed.json",
               {"seed": seed, "selected_auxiliary_weight": selected_alpha,
                "all_planned_unit_sensitivity_runs_completed": True})
    del backbone, features, options, chosen
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=PROTOCOL["model_seeds"])
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()
    records = load_records()
    encodings = encoding_cache(records)
    if not args.encode_only:
        for seed in ([args.seed] if args.seed is not None else PROTOCOL["model_seeds"]):
            print(f"Starting full-HTM unit sensitivity seed {seed}", flush=True)
            run_seed(seed, records, encodings)
