# Farmer activity recognition: gyroscope-unit audit

This package contains a measured sensitivity analysis of the gyroscope-unit
ambiguity in the CERTH agricultural activity release. It accompanies the
archived empirical evaluation in `original/`. It uses released participant
recordings; no new cohort was recruited.

`original/` contains an unchanged subset of the earlier full supplement,
selected for this audit. The historical `original/README.md` and
`experiment_manifest.json` describe that full supplement. Its module-removal
outputs and original plots are not duplicated here. The audit's own included
files are listed and hashed by the top-level `manifest.json`.

## Main files

- `report/Gyroscope_Unit_Audit.md`: documentary findings, measured feature and
  model differences, interpretation, and primary-source links.
- `report/Gyroscope_Units_Methods.tex`: replacement Methods paragraph.
- `report/Gyroscope_Unit_Sensitivity_Results.tex`: Results subsection and table.
- `report/Gyroscope_Unit_Revision.pdf`: rendered Methods/Results excerpt for review.
- `report/per_seed_unit_sensitivity.csv`: accuracy, macro precision/recall/F1,
  probability MAE/MSE/RMSE, transition error, and false switches for each seed.
- `report/unit_sensitivity_results.json`: aggregates, pairing checks, and detailed
  provenance. Fractions in CSV/JSON are on a 0--1 scale, not percentages.
- `work/feature_sensitivity.json`: orientation differences and exact reproduction
  checks for the archived quaternion arrays.
- `work/source_unit_evidence.json`: verified README statement, CSV headers and
  the pinned AHRS API's input-unit requirement.
- `work/original_checkpoint_verification.json`: exact reproduction of all nine
  archived neural test-probability arrays.
- `manifest.json`: SHA-256 hashes for the packaged files.

The manuscript snippets use the existing dataset and AHRS bibliography keys
`ref12` and `ref13`. The PDF is a revision excerpt, not a replacement of the
complete manuscript. Insert the two snippets into the manuscript's LaTeX
source and recompile to update its table and section numbers.

## What the two scenarios mean

The original dataset README states radians per second. Its CSV column names
instead contain degree-per-second suffixes. AHRS 0.4.0 Madgwick expects rad/s.
The original evaluation followed the README and used the numeric gyro values
directly. That convention remains the primary analysis. The alternative treats
the same values as deg/s and multiplies them by pi/180 before the same filter.

The reference results are the unchanged archived fits. Their quaternion arrays
and nine neural checkpoints were independently reproduced exactly. The
alternative models were fitted anew under the same participant split, seeds,
architectures, training and validation-selection rules. Each alternative HTM
backbone was trained afresh. Three auxiliary weights were fitted per backbone;
the full-model result selects among them by validation macro-F1, with the
smallest weight breaking a tie. Test scores do not select the unit convention.

This resolves the implementation description and quantifies the ambiguity.
It does not establish the physical source unit. There is no orientation ground
truth or source-author clarification. The angular differences are separations
between estimates, not errors against ground truth. A sensor's specified
measurement range does not establish the units in a released numeric array.

## Data, split, and statistical scope

The unchanged original archive and its GNU GPL version 2 license are included
under `original/empirical_source_data/`. Retain the source attribution/license
when redistributing data. Cite Anagnostis et al. (2021), Applied Sciences 11,
2188, https://doi.org/10.3390/app11052188.

There are 186 chest-sensor recordings (sensor 1530), 100,579 native frames at
50 Hz, and six activity classes. Training uses participants A--L, validation
M--P, and test Q--T. Models process native frames; scoring uses every fifth
frame, giving 4,636 test endpoints at 10 Hz. D101_1530 and D102_1530 are identical
training files and are retained equally in both scenarios. There are no exact
recording duplicates across participant groups.

Tables use mean and sample SD over seeds 7, 42, and 2026 on one fixed split.
These SDs are neither population confidence intervals nor significance tests.
All models use orientation features except the elapsed-time-only control.
That control has identical predictions in both scenarios. Module-removal
ablations were not rerun for this unit comparison. Auxiliary-weight candidates
are retained to document full-model selection; the selected full model reuses
its winning candidate fit. Class sequence/timing confounds
and the lack of independent field-task validation remain limitations.

## Reproduce

Use Python 3.12.14, a C++17 build toolchain, CMake, and an isolated Python
environment. Start in this extracted package directory. Install the pinned
scientific dependencies and the exact htm.core source:

```bash
python -m pip install -r requirements.txt
python -m pip install cmake==3.31.6 pybind11==3.1.0 scikit-build-core==1.1.1 setuptools wheel
git clone https://github.com/htm-community/htm.core.git vendor_htm
git -C vendor_htm checkout 96309f4969b1488ede172b3eec649dcb9702557d
cmake -S vendor_htm -B vendor_htm/build/cmake -DBINDING_BUILD=CPP_Only -DPROJECT_VERSION1=2.2.0
cmake --build vendor_htm/build/cmake --config Release --target install --parallel 2
python -m pip install --no-build-isolation --no-deps ./vendor_htm
```

The Python installation must include matching development headers and its
shared library. On installations with nonstandard paths, supply their actual
locations with the CMake `Python_EXECUTABLE`, `Python_INCLUDE_DIR`, and
`Python_LIBRARY` cache variables through `CMAKE_ARGS`. Do not alter the HTM
algorithm source to work around Python discovery.

Recompute and refit into a fresh directory, retaining the supplied evidence:

```bash
python run_unit_sensitivity.py --output-dir "$PWD/reproduced_deg_s" --htm-workers 3
```

Use `--htm-workers 1` for serial seed runs. The entry point computes a shared
encoding cache before launching disjoint seed jobs. It verifies the original
features/checkpoints, trains alternative baselines and full HTM models, then
reconstructs the summary. Existing model results are reused; use a fresh output
directory when a complete independent refit is intended. Runtime depends on
available CPU resources. The aggregate report is regenerated under `report/`.

To verify the supplied metrics without refitting:

```bash
export FARMER_BENCHMARK_ROOT="$PWD/work"
export FARMER_BENCHMARK_RUNS="$PWD/work/runs_deg_per_second"
python summarize_unit_sensitivity.py
```

The verification checks test-label, recording, endpoint and inventory alignment,
reconstructs all 36 model/seed/scenario outcomes from saved probabilities, and
requires exact equality of the three time-only prediction arrays. The original
`common.py`, `run_baselines.py`, `run_htm.py`, `reference_model.py`, and numerical
configuration are unchanged. The working protocol adds unit-sensitivity
metadata. The inherited `reference_config.json` status/hardware fields describe
the earlier reference design; actual execution is documented by the protocol,
per-run runtime metadata and this audit, not that old status string.

Neural checkpoints and selected full-HTM readout weights are included. Serialized
SP/TM backbone states are not included: HTM prediction reproduction requires
retraining the seeded backbone. Readouts alone are not complete HTM checkpoints.
Auxiliary candidate probabilities/results are included, with duplicate large
readouts omitted. The deterministic encoding cache can be regenerated.

The audit ran on CPU, not the author-provided i9-9900K / RTX 3060 / 32 GB
workstation. Per-run timings, including concurrent seeds, are not comparative
speed or deployment-latency benchmarks. See `audit_environment.json` and the
runtime blocks in each result JSON for actual versions/hardware.

## Primary sources

- Dataset institution: https://ibo.certh.gr/open-datasets/
- Source paper: https://doi.org/10.3390/app11052188
- Madgwick API: https://ahrs.readthedocs.io/en/latest/filters/madgwick.html
- Vicon sensor specification: https://vicon-help.atlassian.net/wiki/spaces/LMAC/pages/8880610/Capture.U+Education+-+Learn+More

The original archive SHA-256 is
`401a48da1ad82830b932bf69cc3fc09e99cb41bd7f5849fc3dd8c01bea29378a`.
