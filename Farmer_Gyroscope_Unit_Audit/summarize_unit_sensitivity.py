"""Recompute recorded outcomes, verify pairing, and write manuscript-ready unit sensitivity."""
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "original/empirical_runs"
ALTERNATIVE = Path(os.environ.get("FARMER_UNIT_RUNS", BASE / "work/runs_deg_per_second"))
OUT = BASE / "report"
sys.path.insert(0, str(BASE / "work/empirical_experiment_tools"))
from common import evaluate, load_records, select

MODELS = [("Full HTM", "htm/seed_{seed}/full"),
          ("Logistic regression", "baselines/LogisticRegression/seed_{seed}"),
          ("CNN", "baselines/CNN/seed_{seed}"),
          ("LSTM", "baselines/LSTM/seed_{seed}"),
          ("Transformer", "baselines/Transformer/seed_{seed}"),
          ("Elapsed time only", "baselines/TimeOnly/seed_{seed}")]
SEEDS = [7, 42, 2026]
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "probability_mae",
           "probability_mse", "probability_rmse", "transition_error", "false_switches_per_minute"]


def mean_sd(values):
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}


def cell(value, percentage=True, latex=False):
    factor = 100 if percentage else 1
    digits = 2 if percentage else 4
    sign = r"\pm" if latex else "±"
    return f"{value['mean']*factor:.{digits}f} {sign} {value['sample_sd']*factor:.{digits}f}"


def main():
    OUT.mkdir(exist_ok=True)
    records = load_records()
    test = select(records, "test")
    feature = json.loads((BASE / "work/feature_sensitivity.json").read_text())
    checkpoint = json.loads((BASE / "work/original_checkpoint_verification.json").read_text())
    assert feature["original_readme_feature_reproduction"]["all_float32_arrays_bit_exact"]
    assert len(checkpoint["original_neural_checkpoint_verification"]) == 9
    assert all(r["maximum_probability_difference"] == 0 for r in checkpoint["original_neural_checkpoint_verification"])
    with np.load(REFERENCE / "recordings.npz") as a, np.load(ALTERNATIVE / "recordings.npz") as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            if not key.startswith("q_"):
                np.testing.assert_array_equal(a[key], b[key])
    assert json.loads((REFERENCE / "inventory.json").read_text()) == json.loads((ALTERNATIVE / "inventory.json").read_text())
    outcomes = []; aggregate = {}; verified = []; time_controls_exact = []
    for model, relative in MODELS:
        aggregate[model] = {}
        values = {}
        for name, root in [("README_rad_s", REFERENCE), ("CSV_deg_s", ALTERNATIVE)]:
            values[name] = []
            for seed in SEEDS:
                folder = root / relative.format(seed=seed)
                result = json.loads((folder / "result.json").read_text())
                with np.load(folder / "test_predictions.npz") as z:
                    p = z["probabilities"].copy()
                    y = z["labels"].copy()
                    saved_ids = z["clip_id"].copy()
                    saved_frames = z["native_frame"].copy()
                assert len(y) == 4636 and set(np.unique(y)) == set(range(6))
                np.testing.assert_array_equal(y, np.concatenate([r["labels"][r["endpoints"]] for r in test]))
                np.testing.assert_array_equal(saved_ids, np.concatenate([np.repeat(r["clip_id"], len(r["endpoints"])) for r in test]))
                np.testing.assert_array_equal(saved_frames, np.concatenate([r["endpoints"] for r in test]))
                recalculated = evaluate(test, p)
                for metric in METRICS:
                    np.testing.assert_allclose(recalculated[metric], result["test"][metric], rtol=1e-6, atol=1e-7)
                values[name].append(result["test"])
                row = {"model": model, "interpretation": name, "seed": seed,
                       **{m: result["test"][m] for m in METRICS},
                       "selected_auxiliary_weight": result.get("auxiliary_weight"),
                       "selected_logistic_C": result.get("selected_C")}
                outcomes.append(row)
                verified.append({"model": model, "interpretation": name, "seed": seed,
                                 "test_endpoints": len(y), "metrics_recomputed_from_saved_predictions": True})
                if model == "Elapsed time only" and name == "CSV_deg_s":
                    with np.load(REFERENCE / relative.format(seed=seed) / "test_predictions.npz") as old:
                        assert np.array_equal(p, old["probabilities"])
                    time_controls_exact.append(seed)
            aggregate[model][name] = {m: mean_sd([x[m] for x in values[name]]) for m in METRICS}
        aggregate[model]["paired_CSV_minus_README"] = {
            m: mean_sd([values["CSV_deg_s"][i][m] - values["README_rad_s"][i][m] for i in range(3)])
            for m in METRICS}
    assert len(verified) == 36 and time_controls_exact == SEEDS
    auxiliary_rows = []
    for name, root in [("README_rad_s", REFERENCE), ("CSV_deg_s", ALTERNATIVE)]:
        for seed in SEEDS:
            candidates = []
            for alpha in [0.0, 0.1, 0.3]:
                folder = root / f"htm/seed_{seed}/aux_{alpha:g}"
                result = json.loads((folder / "result.json").read_text())
                for split in ["validation", "test"]:
                    with np.load(folder / f"{split}_predictions.npz") as z:
                        score = evaluate(select(records, split), z["probabilities"])
                    for metric in METRICS:
                        np.testing.assert_allclose(score[metric], result[split][metric], rtol=1e-6, atol=1e-7)
                candidates.append((result["validation"]["macro_f1"], -alpha, result))
                auxiliary_rows.append({"interpretation": name, "seed": seed, "auxiliary_weight": alpha,
                                       "validation_macro_f1": result["validation"]["macro_f1"],
                                       "test_macro_f1": result["test"]["macro_f1"]})
            selected = max(candidates, key=lambda x: (x[0], x[1]))
            full = json.loads((root / f"htm/seed_{seed}/full/result.json").read_text())
            assert full["auxiliary_weight"] == -selected[1]
            assert full["test"] == selected[2]["test"]
    with (OUT / "auxiliary_weight_selection.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(auxiliary_rows[0])); w.writeheader(); w.writerows(auxiliary_rows)
    outcome_fields = list(outcomes[0])
    with (OUT / "per_seed_unit_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, outcome_fields); w.writeheader(); w.writerows(outcomes)
    summary = {"study": "Matched gyroscope-unit interpretation sensitivity on the original fixed CERTH split",
               "reference_analysis": "Archived rad/s fits and predictions; quaternion features and all nine neural checkpoints independently reproduced exactly",
               "alternative_analysis": "New deg/s hypothesis fits; source values multiplied by pi/180 before Madgwick",
               "selection": "Source README determines the primary convention; test performance does not select the physical unit",
               "seeds": SEEDS, "test_participants": ["Q", "R", "S", "T"], "test_endpoints": 4636,
               "feature_sensitivity": feature,
               "aggregate": aggregate, "per_seed": outcomes,
               "verification": {"paired_labels_frames_and_recordings_identical": True,
                                "time_control_predictions_identical_for_seeds": time_controls_exact,
                                "all_18_auxiliary_candidates_verified": auxiliary_rows,
                                "all_6_full_model_selections_use_validation_macro_f1": True,
                                "all_36_outcomes_recomputed_from_saved_predictions": verified,
                                **checkpoint},
               "interpretation_limit": "No orientation ground truth or source-author clarification establishes which conflicting unit label is physically correct. Higher classification accuracy cannot establish units.",
               "statistical_scope": "Mean and sample SD over model seeds on one fixed participant split; no population confidence interval or significance test"}
    (OUT / "unit_sensitivity_results.json").write_text(json.dumps(summary, indent=2))
    rows = []
    for model, _ in MODELS:
        a = aggregate[model]["README_rad_s"]; b = aggregate[model]["CSV_deg_s"]
        rows.append([model, cell(a["accuracy"]), cell(b["accuracy"]), cell(a["macro_f1"]),
                     cell(b["macro_f1"]), f"{aggregate[model]['paired_CSV_minus_README']['macro_f1']['mean']*100:+.2f}"])
    header = "| Model | rad/s accuracy (%) | deg/s accuracy (%) | rad/s macro-F1 (%) | deg/s macro-F1 (%) | Δ macro-F1 (pp) |\n|---|---:|---:|---:|---:|---:|"
    table = header + "\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)
    fs = feature["orientation_difference_degrees"]["test_endpoints"]
    htm_a = aggregate["Full HTM"]["README_rad_s"]; htm_b = aggregate["Full HTM"]["CSV_deg_s"]
    htm_weights = {unit: [row["selected_auxiliary_weight"] for row in outcomes if row["model"] == "Full HTM" and row["interpretation"] == unit] for unit in ["README_rad_s", "CSV_deg_s"]}
    report = f'''# Gyroscope-unit audit and matched sensitivity analysis

The dataset release explicitly conflicts: its README states “Angular velocity is measured in rad/s.”, while all three CSV gyroscope columns carry `deg/s` suffixes. The original study's hardware range and Vicon's sensor specification give ±2000 degrees/s; a hardware range does not establish the units of a subsequently released numeric array. AHRS 0.4.0 Madgwick requires rad/s.

The primary convention remains the dataset README's rad/s interpretation. No independent author clarification or orientation ground truth was available. The alternate CSV-header interpretation is evaluated as a sensitivity hypothesis, not selected according to its test performance. Consequently, this audit resolves the implementation and quantifies the metadata uncertainty without certifying the true physical source unit.

## Verified provenance

- Original dataset archive SHA-256: `{feature['source_archive_sha256']}`.
- All 186 chest recordings and all 100,579 native frames are real released data.
- Fixed groups: train A–L (12), validation M–P (4), test Q–T (4).
- Native filter/TM rate 50 Hz; scored endpoints 10 Hz; 4,636 held-out endpoints.
- Recomputed README-convention float32 quaternion features match the archived arrays bit-for-bit: maximum absolute difference 0.
- All nine archived CNN/LSTM/Transformer checkpoints reproduce their saved test probabilities exactly under the pinned scientific package versions.
- Test labels, trial order, timestamps, endpoints, participant groups, and the retained training duplicate are identical between scenarios.
- Three time-only control prediction arrays are exactly unchanged.
- All 36 reported model/seed/scenario outcomes were verified from saved test probabilities.

## Feature effect

The alternative interprets the same source values as degrees/s and supplies `source_gyro * pi/180` to Madgwick. Filter gain 0.033, initialization, quaternion sign continuity, and sample selection remain unchanged. At held-out endpoints, the sign-invariant geodesic difference between the two estimated orientations has mean {fs['mean']:.2f} degrees, median {fs['median']:.2f} degrees, RMS {fs['rms']:.2f} degrees, and 95th percentile {fs['p95']:.2f} degrees. These are differences between estimates, not errors against ground truth.

## Model effect

The rad/s reference uses the archived fits. Alternative features are used to refit logistic regression, CNN, LSTM, Transformer, and the full HTM model with the original code, architectures, participant split, seeds, training rules, and validation-selection procedure. The full HTM backbone is trained afresh for each alternative seed. Three auxiliary weights are fit on each frozen backbone and validation macro-F1 selects the reported full model. The time-only forest is rerun as a negative control. Module-removal ablations are outside this unit-sensitivity comparison.

{table}

Values are mean ± sample SD over seeds 7, 42, 2026. Δ is deg/s minus rad/s in percentage points. Full-HTM validation-selected weights are {htm_weights['README_rad_s']} under rad/s and {htm_weights['CSV_deg_s']} under deg/s. Seed SD describes variation in model fitting on one fixed split; it is not a population confidence interval.

Probability MAE, MSE, RMSE, macro precision/recall, transition errors, and unintended switches are retained in the accompanying CSV/JSON files. Classifier performance changes demonstrate material sensitivity to the unit assumption. They do not calibrate the sensor or establish the true physical unit.

## Manuscript changes

Use `Gyroscope_Units_Methods.tex` for the Methods replacement and `Gyroscope_Unit_Sensitivity_Results.tex` for an added Results subsection. They use existing dataset/AHRS citation keys `ref12` and `ref13` and automatic table numbering. Update table references after inserting the new table. Retain the original main tables under the explicit README convention and report the alternative separately.

## Primary sources

1. Dataset institution: https://ibo.certh.gr/open-datasets/
2. Released archive: https://ibo.certh.gr/wp-content/uploads/2025/01/Human-activity-recognition-for-human-robot-interaction-in-agriculture.zip
3. Source study: https://doi.org/10.3390/app11052188
4. AHRS Madgwick API (rad/s): https://ahrs.readthedocs.io/en/latest/filters/madgwick.html
5. Vicon hardware range: https://vicon-help.atlassian.net/wiki/spaces/LMAC/pages/8880610/Capture.U+Education+-+Learn+More

The original README, source archive, license, frozen protocols, and numeric predictions are included for inspection and reproduction. See the package README for commands.
'''
    (OUT / "Gyroscope_Unit_Audit.md").write_text(report)
    latex_rows = []
    for model, _ in MODELS:
        a = aggregate[model]["README_rad_s"]["macro_f1"]
        b = aggregate[model]["CSV_deg_s"]["macro_f1"]
        delta = aggregate[model]["paired_CSV_minus_README"]["macro_f1"]["mean"]*100
        latex_rows.append(f"{model} & ${cell(a, latex=True)}$ & ${cell(b, latex=True)}$ & ${delta:+.2f}$ \\\\")
    methods = r'''The README distributed with the CERTH dataset \cite{ref12} specifies
angular velocity in radians per second, whereas the CSV gyroscope
column names contain degree-per-second suffixes. The primary analysis
followed the README and supplied the released numeric values directly
to the AHRS 0.4.0 Madgwick filter, which expects radians per second
\cite{ref13}. A sensitivity analysis evaluated the alternative CSV-header
interpretation by multiplying the same gyroscope values by $\pi/180$
before orientation estimation. Both scenarios used 50~Hz IMU updates,
filter gain $0.033$, and no magnetometer input. Initialization,
quaternion normalization and sign continuity, participant groups,
evaluation endpoints, model architectures, seeds, and training and
validation-selection rules were unchanged. Each alternative full-HTM
backbone was trained afresh. The primary convention was determined
from the dataset README before comparing performance.
'''
    results = r'''
\subsubsection{Sensitivity to gyroscope-unit interpretation}

Recomputing the README-convention quaternion features reproduced all
archived float32 arrays exactly. Under the alternative interpretation,
the same source values were treated as degrees per second and converted
to radians per second. Let $\mathbf q_t^{(R)}$ and $\mathbf q_t^{(D)}$
denote the unit quaternions estimated under the README and converted
CSV interpretations, respectively. The orientation difference in radians
was calculated as $2\arccos(\lvert(\mathbf q_t^{(R)})^\top\mathbf q_t^{(D)}\rvert)$,
with numerical clipping of the absolute inner product to $[0,1]$.
Differences were converted to degrees for reporting.
'''
    results += f'''At the 4,636 held-out endpoints, this difference had a median of
${fs['median']:.2f}^\\circ$ and a 95th percentile of ${fs['p95']:.2f}^\\circ$.
These values describe differences between two orientation estimates;
orientation ground truth was unavailable.

Table~\\ref{{tab:gyro_unit_sensitivity}} compares the archived primary
fits with models refitted under the alternative interpretation using
the same protocol and seeds. Validation selection was repeated within
each interpretation. The elapsed-time-only control, which does not use
sensor features, produced identical predictions under both scenarios.

'''
    results += r'''\begin{table}[htbp]
\caption{Sensitivity of test macro-F1 to the interpretation of released
gyroscope values. Entries are mean $\pm$ sample SD across three model
seeds (7, 42, and 2026) on one fixed participant split. The difference is the converted
deg/s scenario minus the README rad/s scenario, in percentage points.}
\label{tab:gyro_unit_sensitivity}
\centering
\small
\begin{tabular}{lrrr}
\hline
Model & README rad/s (\%) & CSV deg/s (\%) & Difference (pp) \\
\hline
'''
    results += "\n".join(latex_rows) + "\n"
    results += r'''\hline
\end{tabular}
\end{table}

The results demonstrate sensitivity to the unit interpretation.
Classification performance does not establish the true physical unit,
and the conflicting source metadata remain a limitation pending
clarification from the dataset creators. The primary results retain the
explicit README convention. This sensitivity comparison does not
validate orientation accuracy or generalization beyond the fixed
participant split.
'''
    (OUT / "Gyroscope_Units_Methods.tex").write_text(methods)
    (OUT / "Gyroscope_Unit_Sensitivity_Results.tex").write_text(results)
    print(table)
    print("VERIFIED_OUTCOMES", len(verified))


if __name__ == "__main__":
    main()
