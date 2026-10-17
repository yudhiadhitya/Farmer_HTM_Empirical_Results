# Farmer activity recognition empirical evaluation

This supplement contains a new secondary evaluation of **real author-released
agricultural material-handling recordings**. It is not a newly recruited farmer
cohort. No synthetic activity examples or borrowed paper scores contribute to
the empirical tables.

## Data attribution and provenance

Anagnostis, A., Benos, L., Tsaopoulos, D., Tagarakis, A., Tsolakis, N., and
Bochtis, D. (2021). Human Activity Recognition through Recurrent Neural Networks
for Human–Robot Interaction in Agriculture. Applied Sciences 11(5), 2188.
https://doi.org/10.3390/app11052188

Institutional release: https://ibo.certh.gr/open-datasets/

Original archive:
https://ibo.certh.gr/wp-content/uploads/2025/01/Human-activity-recognition-for-human-robot-interaction-in-agriculture.zip

SHA-256:
401a48da1ad82830b932bf69cc3fc09e99cb41bd7f5849fc3dd8c01bea29378a

The original archive and released README, filename mapping, activity mapping,
and GNU GPL version 2 license are included unchanged. Retain the dataset's
attribution and license when redistributing it. The download contains 930 CSVs:
186 trials times five sensors. This benchmark selects only chest sensor 1530,
with 100,579 native frames from all 20 participants. Participants A–L train,
M–P validate, and Q–T test. The native interval is 0.02 s (50 Hz). All native
frames enter orientation and temporal processing; loss/readout endpoints and
reported metrics use every fifth frame (10 Hz). No interpolation is used.
Chest files D101_1530 and D102_1530 contain identical bytes. Both are in
training and are retained consistently across models. There are 185 unique
chest-file hashes, with no identical recording bytes across groups.

The README says gyroscope units are rad/s, while the CSV column headers say
deg/s. Processing follows the README as frozen in the protocol, without unit
conversion. This discrepancy is disclosed in the manuscript. Condition labels
are retained from the source; annotation agreement and source video are not
provided in the download. AHRS Madgwick IMU gain is 0.033 and each recording
starts from its first accelerometer sample. Magnetometer values are unused.

## Actual execution and configurations

Executed on Linux with an AMD EPYC 9V74 CPU, Python 3.12.14, NumPy 2.3.5,
SciPy 1.17.0, scikit-learn 1.8.0, PyTorch 2.5.1+cpu, AHRS 0.4.0, and htm.core
2.2.0 at commit 96309f4969b1488ede172b3eec649dcb9702557d. The supplied
i9-9900K / RTX 3060 / 32 GB workstation was not used or timed. Recorded training
durations describe this execution environment, including concurrent independent
seed runs; they are not fair model speed benchmarks or end-to-end latency.

`empirical_experiment_tools/protocol.json` records the fixed participants,
models, seeds, selection rules, loss weights, label mapping and metrics.
`reference_config.json` retains every SP/TM constructor argument. Four RDSEs
have nominal 21 active bits each; hashing gives 78–84 unique concatenated bits
in the observed recordings. The HTM backbone learns three passes on training
data only, then freezes. TM resets per recording. The readout uses normalized
sparse indicators and 20 SGD epochs at 0.01. The auxiliary horizon is one
native frame, 20 ms. Weights 0, 0.1 and 0.3 are all reported; validation chooses
the full-model weight. No-SP retains the full model's chosen weight and alters
the input density. No-TM uses the same SP columns with current-label loss.

The logistic baseline selects C from 0.1, 1 and 10 using validation macro-F1.
The causal CNN, unidirectional LSTM and causal Transformer have hidden width
32, complete-trial input, training-only standardization and no future-frame
access. They use Adam 0.001, batches of four trials, clipping at norm 1,
unweighted cross-entropy, at most 20 epochs and patience 5. Validation chooses
the checkpoint. Future-perturbation causality checks ran before each neural
training run. Architecture details are defined in `run_baselines.py` and in
the manuscript. The elapsed-time-only forest uses 200 trees, depth 8 and leaf
size at least 30; it has no sensor input or future duration normalization.

## Reproduction

Use Python 3.12.14 and a C++17 build toolchain. Start in the extracted package
root. Install the pinned analysis requirements:

```bash
python -m pip install -r requirements.txt
```

Install the same htm.core source rather than an unrelated PyPI package:

```bash
git clone https://github.com/htm-community/htm.core.git vendor_htm
git -C vendor_htm checkout 96309f4969b1488ede172b3eec649dcb9702557d
python -m pip install cmake==3.31.6 setuptools wheel
cmake -S vendor_htm -B vendor_htm/build/cmake -DBINDING_BUILD=CPP_Only -DPROJECT_VERSION1=2.2.0
cmake --build vendor_htm/build/cmake --config Release --target install --parallel 2
python -m pip install --no-build-isolation ./vendor_htm
```

The included `empirical_runs` directory holds the original measured outputs.
To reproduce without overwriting them, choose a fresh run directory:

```bash
export FARMER_BENCHMARK_RUNS="$PWD/reproduced_runs"
python empirical_experiment_tools/prepare_data.py
python empirical_experiment_tools/run_baselines.py --model classical
python empirical_experiment_tools/run_baselines.py --model CNN
python empirical_experiment_tools/run_baselines.py --model LSTM
python empirical_experiment_tools/run_baselines.py --model Transformer
python empirical_experiment_tools/run_htm.py
python empirical_experiment_tools/summarize_results.py
```

Independent HTM seeds may also be run using `--seed 7`, `--seed 42`, and
`--seed 2026`, on disjoint output directories within the same run root.
Do not launch overlapping serial and individual HTM jobs on the same output
paths. Existing completed results are reused. To check the archived results
without retraining, unset FARMER_BENCHMARK_RUNS and run the summarizer.

The archive includes neural weights and all fitted readout weights. Serialized
SP/TM backbone states are not included; reproducing HTM predictions requires
retraining the seeded backbone from the retained data and exact configuration.
This distinction is explicit rather than calling the readouts complete HTM
model checkpoints.

## Reported files and statistical scope

- `empirical_runs/inventory.json`: participants, recording IDs, source member
  names, hashes, durations, native timing and class coverage.
- `empirical_runs/recordings.npz`: measured, derived quaternion arrays, labels,
  timestamps and selected endpoints; no simulated data.
- `empirical_runs/baselines/*/seed_*/`: test predictions, result JSONs,
  selected settings, histories and neural checkpoint files.
- `empirical_runs/htm/seed_*/`: all loss weights, the validation-selected full
  model, no-SP and no-TM outcomes, predictions and fitted readouts.
- `empirical_runs/summary/aggregate_metrics.csv` and `.json`: all test metrics
  as mean and sample SD across seeds 7, 42, 2026.
- `empirical_runs/summary/verification.json`: reconstruction of all 33 model
  outcomes from saved predictions, participant isolation and source hash checks.
- `empirical_runs/summary/*.png` and `.pdf`: standalone scientific plots.

Metrics pool 4,636 held-out endpoints. Precision, recall and F1 are macro
averages over all six classes, with zero for undefined per-class values.
Probability MAE/MSE average over sample–class pairs against one-hot targets;
RMSE is the square root of MSE. Transition error uses endpoints within ±0.5 s
of supplied boundaries. Unintended switches require both adjacent endpoints
outside all such windows and are normalized by total recorded test minutes.
Neither count crosses a recording boundary.

Three-seed SD describes model variability on **one fixed participant split**.
The selected full-HTM outcome reuses its validation-selected loss-weight fit;
it is not an additional independent training run. There are 30 fitted
model/seed configurations and 33 reported model/seed outcomes.
It is not cross-validation, a population confidence interval, or a significance
test. Trials have the same six-stage activity order. An elapsed-time-only
control assesses this important timing confound. The 20 ms auxiliary target
retains the current label in 99.46% of eligible test endpoint pairs. There is
no separately reported next-step forecasting benchmark, noise experiment,
musculoskeletal outcome, irregular field-task test, or target-workstation
deployment benchmark.
