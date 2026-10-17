# Gyroscope-unit audit and matched sensitivity analysis

The dataset release explicitly conflicts: its README states “Angular velocity is measured in rad/s.”, while all three CSV gyroscope columns carry `deg/s` suffixes. The original study's hardware range and Vicon's sensor specification give ±2000 degrees/s; a hardware range does not establish the units of a subsequently released numeric array. AHRS 0.4.0 Madgwick requires rad/s.

The primary convention remains the dataset README's rad/s interpretation. No independent author clarification or orientation ground truth was available. The alternate CSV-header interpretation is evaluated as a sensitivity hypothesis, not selected according to its test performance. Consequently, this audit resolves the implementation and quantifies the metadata uncertainty without certifying the true physical source unit.

## Verified provenance

- Original dataset archive SHA-256: `401a48da1ad82830b932bf69cc3fc09e99cb41bd7f5849fc3dd8c01bea29378a`.
- All 186 chest recordings and all 100,579 native frames are real released data.
- Fixed groups: train A–L (12), validation M–P (4), test Q–T (4).
- Native filter/TM rate 50 Hz; scored endpoints 10 Hz; 4,636 held-out endpoints.
- Recomputed README-convention float32 quaternion features match the archived arrays bit-for-bit: maximum absolute difference 0.
- All nine archived CNN/LSTM/Transformer checkpoints reproduce their saved test probabilities exactly under the pinned scientific package versions.
- Test labels, trial order, timestamps, endpoints, participant groups, and the retained training duplicate are identical between scenarios.
- Three time-only control prediction arrays are exactly unchanged.
- All 36 reported model/seed/scenario outcomes were verified from saved test probabilities.

## Feature effect

The alternative interprets the same source values as degrees/s and supplies `source_gyro * pi/180` to Madgwick. Filter gain 0.033, initialization, quaternion sign continuity, and sample selection remain unchanged. At held-out endpoints, the sign-invariant geodesic difference between the two estimated orientations has mean 72.52 degrees, median 35.04 degrees, RMS 103.94 degrees, and 95th percentile 177.38 degrees. These are differences between estimates, not errors against ground truth.

## Model effect

The rad/s reference uses the archived fits. Alternative features are used to refit logistic regression, CNN, LSTM, Transformer, and the full HTM model with the original code, architectures, participant split, seeds, training rules, and validation-selection procedure. The full HTM backbone is trained afresh for each alternative seed. Three auxiliary weights are fit on each frozen backbone and validation macro-F1 selects the reported full model. The time-only forest is rerun as a negative control. Module-removal ablations are outside this unit-sensitivity comparison.

| Model | rad/s accuracy (%) | deg/s accuracy (%) | rad/s macro-F1 (%) | deg/s macro-F1 (%) | Δ macro-F1 (pp) |
|---|---:|---:|---:|---:|---:|
| Full HTM | 54.59 ± 0.71 | 35.40 ± 1.34 | 23.58 ± 0.18 | 14.47 ± 0.98 | -9.11 |
| Logistic regression | 64.34 ± 0.00 | 38.70 ± 0.00 | 42.20 ± 0.00 | 16.75 ± 0.00 | -25.45 |
| CNN | 80.82 ± 0.28 | 45.07 ± 0.91 | 75.67 ± 1.01 | 36.65 ± 0.70 | -39.02 |
| LSTM | 81.97 ± 3.16 | 42.49 ± 4.91 | 74.47 ± 6.55 | 28.16 ± 5.69 | -46.31 |
| Transformer | 90.69 ± 0.15 | 84.45 ± 1.59 | 89.29 ± 0.18 | 82.27 ± 1.81 | -7.02 |
| Elapsed time only | 82.67 ± 0.12 | 82.67 ± 0.12 | 79.86 ± 0.26 | 79.86 ± 0.26 | +0.00 |

Values are mean ± sample SD over seeds 7, 42, 2026. Δ is deg/s minus rad/s in percentage points. Full-HTM validation-selected weights are [0.1, 0.0, 0.0] under rad/s and [0.3, 0.0, 0.0] under deg/s. Seed SD describes variation in model fitting on one fixed split; it is not a population confidence interval.

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
