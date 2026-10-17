"""Prospective quaternion/HTM HAR reference. No empirical farmer results included."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

VERSION = "1.0.0"
FIELDS = ("timestamp_seconds", "qw", "qx", "qy", "qz", "label")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    """Scalar first; canonical initial sign and continuous signs per recording."""
    q = np.array(q, dtype=np.float64, copy=True)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError("Quaternions must be a finite N by 4 scalar-first array.")
    norms = np.linalg.norm(q, axis=1)
    if len(q) == 0 or np.any(norms < 1e-10):
        raise ValueError("Empty or zero-norm quaternion input.")
    q /= norms[:, None]
    first_nonzero = np.flatnonzero(np.abs(q[0]) > 1e-12)[0]
    if q[0, first_nonzero] < 0:
        q[0] *= -1
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1
    return q


@dataclass
class Recording:
    clip_id: str
    group_id: str
    split: str
    times: np.ndarray
    quaternions: np.ndarray
    labels: np.ndarray
    provenance: dict


def resample(times, q, labels, target_hz=50.0, max_gap_seconds=0.1):
    times = np.asarray(times, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    q = normalize_quaternions(q)
    if len(times) != len(q) or len(labels) != len(q) or len(times) < 2:
        raise ValueError("A recording needs at least two aligned samples.")
    dt = np.diff(times)
    if not np.isfinite(times).all() or np.any(dt <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing.")
    if np.any(dt > max_gap_seconds + 1e-9):
        raise ValueError("Split recordings at gaps longer than max_gap_seconds.")
    if target_hz <= 0:
        raise ValueError("target_hz must be positive.")
    # No extrapolation and no assumed native sensor acquisition rate.
    count = int(np.floor((times[-1] - times[0]) * target_hz + 1e-9)) + 1
    grid = times[0] + np.arange(count) / target_hz
    grid = grid[grid <= times[-1]]
    rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    interpolated = Slerp(times, rotation)(grid).as_quat()[:, [3, 0, 1, 2]]
    # Labels describe left-closed sample intervals; never interpolate class IDs.
    index = np.searchsorted(times, grid, side="right") - 1
    timing = {
        "timestamp_median_rate_hz": float(1.0 / np.median(dt)),
        "native_interval_min_seconds": float(dt.min()),
        "native_interval_max_seconds": float(dt.max()),
        "native_sample_count": len(times), "analysis_rate_hz": target_hz,
        "analysis_sample_count": len(grid),
        "resampling": "SLERP; left-closed label intervals; no anti-alias filter"
    }
    return grid, normalize_quaternions(interpolated), labels[index], timing


def load_recordings(manifest_path: Path, config: dict) -> list[Recording]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["recordings"]
    if not entries:
        raise ValueError("Manifest has no recordings.")
    group_split, hash_split, clip_ids = {}, {}, set()
    loaded = []
    for entry in entries:
        clip, group, split = (str(entry[k]) for k in ("clip_id", "group_id", "split"))
        if not clip or not group or split not in {"train", "validation", "test"}:
            raise ValueError("Each recording needs clip_id, group_id and a valid split.")
        if clip in clip_ids:
            raise ValueError(f"Duplicate clip_id: {clip}")
        clip_ids.add(clip)
        if group in group_split and group_split[group] != split:
            raise ValueError(f"Group occurs in multiple splits: {group}")
        group_split[group] = split
        path = (manifest_path.parent / entry["path"]).resolve()
        digest = file_hash(path)
        if digest in hash_split and hash_split[digest] != split:
            raise ValueError("Identical file bytes occur in multiple splits.")
        hash_split[digest] = split
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not set(FIELDS).issubset(reader.fieldnames or []):
                raise ValueError(f"Missing required CSV columns in {path.name}")
            rows = list(reader)
        times = np.array([float(r["timestamp_seconds"]) for r in rows])
        q = np.array([[float(r[k]) for k in FIELDS[1:5]] for r in rows])
        labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
        if np.any(labels < 0):
            raise ValueError("Labels must be nonnegative integers.")
        times, q, labels, timing = resample(
            times, q, labels, config["target_hz"], config["max_gap_seconds"])
        provenance = {"path": entry["path"], "sha256": digest,
                      "source": entry.get("source", {}), **timing}
        loaded.append(Recording(clip, group, split, times, q, labels, provenance))
    present = {r.split for r in loaded}
    if present != {"train", "validation", "test"}:
        raise ValueError("Provide independent train, validation, and test recordings.")
    train_labels = set(np.concatenate([r.labels for r in loaded if r.split == "train"]))
    all_labels = set(np.concatenate([r.labels for r in loaded]))
    if train_labels != all_labels:
        raise ValueError("An evaluation class has no training observation; revise class coverage.")
    if all_labels != set(range(len(all_labels))):
        raise ValueError("Map classes to contiguous integers starting at zero.")
    return loaded


def read_bvh_root(path: Path):
    """Read ROOT rotation only, using declared intrinsic Euler channel order."""
    text = path.read_text(encoding="utf-8-sig")
    hierarchy, motion = re.split(r"\bMOTION\b", text, maxsplit=1)
    root = re.search(r"\bROOT\s+(\S+)", hierarchy)
    channels = re.findall(r"\bCHANNELS\s+(\d+)\s+([^\r\n{}]+)", hierarchy)
    if root is None or not channels:
        raise ValueError("BVH ROOT and CHANNELS declarations are required.")
    channel_lists = []
    for n, names in channels:
        names = names.split()
        if len(names) != int(n):
            raise ValueError("Inconsistent BVH channel declaration.")
        channel_lists.append(names)
    frames = re.search(r"Frames\s*:\s*(\d+)", motion)
    interval = re.search(r"Frame\s+Time\s*:\s*([\d.eE+-]+)", motion)
    if frames is None or interval is None:
        raise ValueError("BVH Frames and Frame Time fields are required.")
    dt = float(interval.group(1))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid BVH Frame Time.")
    values = np.loadtxt(io.StringIO(motion[interval.end():]), ndmin=2)
    if values.shape != (int(frames.group(1)), sum(map(len, channel_lists))):
        raise ValueError("BVH motion dimensions do not match declarations.")
    indexes = [i for i, name in enumerate(channel_lists[0]) if name.endswith("rotation")]
    order = "".join(channel_lists[0][i][0].upper() for i in indexes)
    if len(order) != 3 or set(order) != set("XYZ"):
        raise ValueError("ROOT must specify three distinct rotation axes.")
    # BVH angles are degrees; uppercase SciPy sequences denote intrinsic rotations.
    q = Rotation.from_euler(order, values[:, indexes], degrees=True).as_quat()
    metadata = {"source_type": "BVH animation", "root_joint": root.group(1),
                "rotation_order": order, "rotation_convention": "intrinsic Euler degrees",
                "source_frame_time_seconds": dt, "source_frame_rate_hz": 1.0 / dt,
                "source_frames": len(values), "source_sha256": file_hash(path),
                "feature": "local ROOT orientation only; translation excluded"}
    return np.arange(len(q)) * dt, normalize_quaternions(q[:, [3, 0, 1, 2]]), metadata


def convert_bvh(path: Path, output: Path, label: int):
    if label < 0:
        raise ValueError("label must be nonnegative.")
    times, q, metadata = read_bvh_root(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for t, quat in zip(times, q):
            writer.writerow([float(t), *quat.tolist(), label])
    output.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class HTMFeatures:
    def __init__(self, config: dict):
        from htm.bindings.sdr import SDR
        from htm.bindings.encoders import RDSE, RDSE_Parameters
        from htm.bindings.algorithms import SpatialPooler, TemporalMemory
        self.SDR = SDR
        self.encoders = []
        e = config["encoder"]
        for component in range(4):
            p = RDSE_Parameters()
            p.size, p.activeBits, p.resolution = e["size_per_component"], e["activeBits"], e["resolution"]
            p.seed = config["seed"] + component + 1
            self.encoders.append(RDSE(p))
        sp = copy.deepcopy(config["sp"])
        tm = copy.deepcopy(config["tm"])
        for settings in (sp, tm):
            for key in ("inputDimensions", "columnDimensions"):
                if key in settings:
                    settings[key] = tuple(settings[key])
        if sp["inputDimensions"] != (4 * e["size_per_component"],):
            raise ValueError("Encoder and SP input dimensions disagree.")
        if sp["columnDimensions"] != tm["columnDimensions"]:
            raise ValueError("SP and TM column dimensions disagree.")
        self.sp = SpatialPooler(**sp)
        self.tm = TemporalMemory(**tm)
        self.width = 4 * e["size_per_component"]
        self.cell_count = self.tm.numberOfCells()

    def reset(self):
        self.tm.reset()

    def step(self, q, learn=False):
        parts = [encoder.encode(float(value)) for encoder, value in zip(self.encoders, q)]
        encoded = self.SDR(self.width).concatenate(parts)
        columns = self.SDR(self.sp.getColumnDimensions())
        self.sp.compute(encoded, learn, columns)
        self.tm.compute(columns, learn=learn)
        active = self.tm.getActiveCells().sparse.copy().astype(np.int64)
        self.tm.activateDendrites(learn)
        predictive = self.tm.getPredictiveCells().sparse.copy().astype(np.int64)
        return active, predictive

    def transform(self, recording: Recording, learn=False):
        self.reset()
        return [self.step(q, learn) for q in recording.quaternions]

    def save(self, directory: Path):
        self.reset()
        self.sp.saveToFile(str(directory / "sp.bin"), "BINARY")
        self.tm.saveToFile(str(directory / "tm.bin"), "BINARY")
        for i, encoder in enumerate(self.encoders):
            encoder.saveToFile(str(directory / f"encoder_{i}.bin"), "BINARY")


def probabilities(weights, bias, indices):
    logits = bias.copy()
    if len(indices):
        logits += weights[indices].mean(axis=0)
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def hybrid_gradient(weights, bias, active, label, predictive, next_label, auxiliary_weight):
    """Return both gradients at the same weights, before either update."""
    if auxiliary_weight < 0:
        raise ValueError("Auxiliary weight must be nonnegative.")
    pc = probabilities(weights, bias, active)
    gc = pc.copy()
    gc[label] -= 1
    loss = -float(np.log(max(pc[label], np.finfo(float).tiny)))
    gp = np.zeros_like(bias)
    if next_label is not None and auxiliary_weight:
        pn = probabilities(weights, bias, predictive)
        gp = pn.copy()
        gp[next_label] -= 1
        gp *= auxiliary_weight
        loss -= auxiliary_weight * float(np.log(max(pn[next_label], np.finfo(float).tiny)))
    return loss, gc, gp


def update_readout(weights, bias, active, label, predictive, next_label, alpha, rate):
    loss, gc, gp = hybrid_gradient(weights, bias, active, label, predictive, next_label, alpha)
    if len(active):
        np.add.at(weights, active, -rate * gc / len(active))
    if len(predictive):
        np.add.at(weights, predictive, -rate * gp / len(predictive))
    bias -= rate * (gc + gp)
    return loss


def metrics(labels, probabilities_array, class_count):
    labels = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities_array)
    predicted = p.argmax(axis=1)
    cm = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(cm, (labels, predicted), 1)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(class_count), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(class_count), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(class_count), where=precision + recall > 0)
    represented = cm.sum(1) > 0
    errors = p - np.eye(class_count)[labels]
    mse = float(np.square(errors).mean())
    return {"accuracy": float(tp.sum() / len(labels)),
            "macro_f1_present_classes": float(f1[represented].mean()),
            "class_ids_in_macro": np.flatnonzero(represented).tolist(),
            "class_support": cm.sum(1).tolist(), "precision": precision.tolist(),
            "recall": recall.tolist(), "f1": f1.tolist(), "confusion_matrix": cm.tolist(),
            "probability_mae": float(np.abs(errors).mean()), "probability_mse": mse,
            "probability_rmse": float(np.sqrt(mse)),
            "zero_division_rule": "zero; macro excludes classes absent in ground truth"}


def train_and_evaluate(manifest_path: Path, config_path: Path, output: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    recordings = load_recordings(manifest_path, config)
    class_count = len(set(np.concatenate([r.labels for r in recordings])))
    features = HTMFeatures(config)
    train = [r for r in recordings if r.split == "train"]
    for _ in range(config["training"]["htm_passes"]):
        for recording in train:
            features.transform(recording, learn=True)
    # Frozen HTM for extraction, validation, and testing; labels never enter HTM.
    cached = {r.clip_id: features.transform(r, learn=False) for r in recordings}
    validation = [r for r in recordings if r.split == "validation"]
    settings = config["training"]
    candidates = []
    for alpha in settings["auxiliary_weights"]:
        rng = np.random.default_rng(config["seed"])
        weights = rng.normal(0, 0.001, (features.cell_count, class_count)).astype(np.float64)
        bias = np.zeros(class_count)
        for _ in range(settings["readout_epochs"]):
            for i in rng.permutation(len(train)):
                recording = train[i]
                sequence = cached[recording.clip_id]
                for t, (active, predictive) in enumerate(sequence):
                    following = int(recording.labels[t + 1]) if t + 1 < len(sequence) else None
                    update_readout(weights, bias, active, int(recording.labels[t]), predictive,
                                   following, alpha, settings["learning_rate"])
        vl, vp = [], []
        for recording in validation:
            vl.extend(recording.labels.tolist())
            vp.extend(probabilities(weights, bias, a) for a, _ in cached[recording.clip_id])
        score = metrics(vl, vp, class_count)
        candidates.append((score["macro_f1_present_classes"], -alpha, weights, bias, score))
    chosen = max(candidates, key=lambda item: (item[0], item[1]))
    _, neg_alpha, weights, bias, validation_metrics = chosen
    test_labels, test_p, predictions = [], [], []
    for recording in recordings:
        if recording.split == "test":
            for i, (a, predictive) in enumerate(cached[recording.clip_id]):
                p = probabilities(weights, bias, a)
                test_labels.append(int(recording.labels[i]))
                test_p.append(p)
                predictions.append([recording.clip_id, float(recording.times[i]),
                                    int(recording.labels[i]), int(p.argmax()), *p.tolist()])
    output.mkdir(parents=True, exist_ok=True)
    features.save(output)
    np.savez(output / "readout.npz", weights=weights, bias=bias)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "timestamp_seconds", "label", "prediction"] +
                        [f"probability_{i}" for i in range(class_count)])
        writer.writerows(predictions)
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "htm"):
        versions[name] = importlib.metadata.version(name)
    report = {"reference_version": VERSION, "selected_auxiliary_weight": -neg_alpha,
              "validation": validation_metrics, "test": metrics(test_labels, test_p, class_count),
              "configuration": config, "runtime_versions": versions,
              "runtime_os": platform.platform(), "runtime_machine": platform.machine(),
              "runtime_processor": platform.processor(),
              "manifest_sha256": file_hash(manifest_path),
              "recordings": [{"clip_id": r.clip_id, "group_id": r.group_id,
                              "split": r.split, **r.provenance} for r in recordings],
              "scope": "These metrics describe only the supplied manifest, not an original farmer cohort.",
              "timing_status": "No real-time latency benchmark is computed by this command."}
    (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bvh = sub.add_parser("convert-bvh")
    bvh.add_argument("input", type=Path)
    bvh.add_argument("output", type=Path)
    bvh.add_argument("--label", type=int, required=True)
    run = sub.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "convert-bvh":
            convert_bvh(args.input, args.output, args.label)
        else:
            report = train_and_evaluate(args.manifest, args.config, args.output)
            print(json.dumps({"output": str(args.output),
                              "selected_auxiliary_weight": report["selected_auxiliary_weight"]}, indent=2))
    except (ValueError, KeyError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
