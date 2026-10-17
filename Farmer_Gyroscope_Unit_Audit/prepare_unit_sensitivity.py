"""Reproduce the archived orientation features and vary only gyro-unit interpretation."""
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from ahrs.filters import Madgwick

BASE = Path(__file__).resolve().parent
ORIGINAL = BASE / "original"
WORK = BASE / "work"
ALTERNATIVE = Path(os.environ.get("FARMER_UNIT_RUNS", WORK / "runs_deg_per_second"))


def orientation(gyro, acc):
    q = Madgwick(gyr=gyro, acc=acc, frequency=50.0, gain=0.033).Q
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    if q[0, np.flatnonzero(np.abs(q[0]) > 1e-12)[0]] < 0:
        q[0] *= -1
    for t in range(1, len(q)):
        if np.dot(q[t - 1], q[t]) < 0:
            q[t] *= -1
    return q


def stats(x):
    x = np.asarray(x, dtype=float)
    return {"n": int(x.size), "mean": float(x.mean()),
            "rms": float(np.sqrt(np.mean(x*x))),
            "median": float(np.median(x)), "p95": float(np.quantile(x, .95)),
            "maximum": float(x.max())}


def main():
    WORK.mkdir(exist_ok=True)
    ALTERNATIVE.mkdir(exist_ok=True)
    shutil.copytree(ORIGINAL / "empirical_experiment_tools",
                    WORK / "empirical_experiment_tools", dirs_exist_ok=True)
    protocol_path = WORK / "empirical_experiment_tools/protocol.json"
    original_protocol = json.loads(protocol_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    protocol["experiment_version"] = "1.0.0-unit-sensitivity"
    protocol["orientation"]["gyro_units"] = "sensitivity hypothesis: CSV values in deg/s; converted to rad/s by pi/180"
    protocol["unit_sensitivity"] = {
        "reference": "archived rad/s README interpretation, no conversion",
        "alternative": "deg/s CSV-header interpretation, pi/180 conversion",
        "changes": "gyro input scale only; same splits, filter, endpoints, models, seeds, training and validation selection",
        "unit_choice_not_selected_by_test_performance": True,
    }
    protocol_path.write_text(json.dumps(protocol, indent=2))
    inventory = json.loads((ORIGINAL / "empirical_runs/inventory.json").read_text())
    (ALTERNATIVE / "inventory.json").write_text(json.dumps(inventory, indent=2))
    (ALTERNATIVE / "protocol.json").write_text(json.dumps(protocol, indent=2))
    archive = ORIGINAL / "empirical_source_data/certh_original_activity.zip"
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive_hash == original_protocol["source_zip_sha256"]
    with np.load(ORIGINAL / "empirical_runs/recordings.npz") as z:
        arrays = {key: z[key].copy() for key in z.files}
    angle_groups = {key: [] for key in ["all_native", "test_native", "test_endpoints"]}
    step_a = []; step_b = []; raw_gyro = []; per_recording = []
    max_archived_difference = 0.0
    all_bit_exact = True
    with ZipFile(archive) as z:
        for record in inventory["recordings"]:
            index = record["index"]
            payload = z.read(record["source_member"])
            assert hashlib.sha256(payload).hexdigest() == record["source_sha256"]
            rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
            acc = np.array([[float(r[k]) for k in ("ax_m/s/s", "ay_m/s/s", "az_m/s/s")] for r in rows])
            gyro = np.array([[float(r[k]) for k in ("gx_deg/s", "gy_deg/s", "gz_deg/s")] for r in rows])
            labels = np.array([int(float(r["condition"])) for r in rows])
            np.testing.assert_array_equal(labels, arrays[f"y_{index}"])
            qa = orientation(gyro, acc)
            qb = orientation(np.deg2rad(gyro), acc)
            archived = arrays[f"q_{index}"]
            error = float(np.max(np.abs(qa.astype(np.float32) - archived)))
            max_archived_difference = max(error, max_archived_difference)
            all_bit_exact &= bool(np.array_equal(qa.astype(np.float32), archived))
            assert np.isfinite(qb).all()
            np.testing.assert_allclose(np.linalg.norm(qb, axis=1), 1., atol=1e-12)
            arrays[f"q_{index}"] = qb.astype(np.float32)
            # Quaternion sign is physically immaterial; use geodesic SO(3) distance.
            angle = np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(qa*qb, axis=1)), 0., 1.)))
            angle_groups["all_native"].extend(angle.tolist())
            if record["split"] == "test":
                angle_groups["test_native"].extend(angle.tolist())
                angle_groups["test_endpoints"].extend(angle[arrays[f"e_{index}"]].tolist())
            for q, dest in [(qa, step_a), (qb, step_b)]:
                increments = np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(q[1:]*q[:-1], axis=1)), 0., 1.)))
                dest.extend(increments.tolist())
            raw_gyro.extend(gyro.tolist())
            per_recording.append({"clip_id": record["clip_id"], "split": record["split"],
                                   "orientation_difference_degrees": stats(angle),
                                   "archived_max_abs_feature_error": error})
            if (index+1) % 30 == 0:
                print(f"Processed {index+1}/{len(inventory['recordings'])} real chest recordings", flush=True)
    assert all_bit_exact, "Original rad/s features did not reproduce bit-for-bit; investigate before training."
    np.savez_compressed(ALTERNATIVE / "recordings.npz", **arrays)
    gyro = np.array(raw_gyro)
    summary = {"source_archive_sha256": archive_hash,
               "original_readme_feature_reproduction": {"all_float32_arrays_bit_exact": all_bit_exact,
                                                         "maximum_absolute_difference": max_archived_difference},
               "orientation_difference_degrees": {key: stats(value) for key, value in angle_groups.items()},
               "per_frame_rotation_degrees": {"README_rad_per_second": stats(step_a),
                                               "CSV_deg_per_second": stats(step_b)},
               "raw_gyro_absolute_components": stats(np.abs(gyro).ravel()),
               "raw_gyro_vector_magnitude": stats(np.linalg.norm(gyro, axis=1)),
               "recordings": len(per_recording), "per_recording": per_recording,
               "interpretation": "Geodesic difference between two estimates, not orientation error against ground truth.",
               "runtime_python": platform.python_version(),
               "changed_fields": ["gyro scale before Madgwick"]}
    (WORK / "feature_sensitivity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "per_recording"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
