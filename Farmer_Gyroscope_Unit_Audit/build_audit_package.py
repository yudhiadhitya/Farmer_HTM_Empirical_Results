"""Bundle verified unit sensitivity and its minimal reproducible evidence."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

BASE = Path(__file__).resolve().parent
OUT = BASE / "deliverable"


def main():
    report = BASE / "report"
    summary = json.loads((report / "unit_sensitivity_results.json").read_text())
    assert len(summary["verification"]["all_36_outcomes_recomputed_from_saved_predictions"]) == 36
    for seed in [7, 42, 2026]:
        marker = BASE / f"work/runs_deg_per_second/htm/seed_{seed}/unit_sensitivity_completed.json"
        assert json.loads(marker.read_text())["all_planned_unit_sensitivity_runs_completed"]

    paths = {}

    def add(path, relative=None):
        assert path.is_file(), str(path)
        paths[relative or str(path.relative_to(BASE))] = path

    for name in ["document_unit_evidence.py", "prepare_unit_sensitivity.py", "run_htm_unit_sensitivity.py",
                 "verify_reference.py", "run_unit_sensitivity.py",
                 "summarize_unit_sensitivity.py", "build_audit_package.py"]:
        add(BASE / name)
    for p in report.iterdir():
        if p.is_file() and p.suffix in {".md", ".tex", ".csv", ".json", ".pdf"}:
            add(p)
    for name in ["README.md", "requirements.txt", "experiment_manifest.json"]:
        add(BASE / "original" / name)
    for directory in ["original/empirical_source_data", "original/empirical_experiment_tools",
                      "original/empirical_runs/baselines", "work/empirical_experiment_tools",
                      "work/runs_deg_per_second/baselines"]:
        for p in (BASE / directory).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                add(p)
    for directory in ["original/empirical_runs", "work/runs_deg_per_second"]:
        for name in ["inventory.json", "protocol.json", "recordings.npz"]:
            add(BASE / directory / name)
    for seed in [7, 42, 2026]:
        for scenario in ["original/empirical_runs", "work/runs_deg_per_second"]:
            for p in (BASE / scenario / f"htm/seed_{seed}/full").iterdir():
                if p.is_file():
                    add(p)
        # Auxiliary candidates are retained as predictions/metrics. Avoid
        # duplicating the selected full model's large readout weights.
        for alpha in ["0", "0.1", "0.3"]:
            for scenario in ["original/empirical_runs", "work/runs_deg_per_second"]:
                d = BASE / scenario / f"htm/seed_{seed}/aux_{alpha}"
                for name in ["result.json", "test_predictions.npz", "validation_predictions.npz"]:
                    add(d / name)
        add(BASE / f"work/runs_deg_per_second/htm/seed_{seed}/unit_sensitivity_completed.json")
        add(BASE / f"private/htm_seed_{seed}.log", f"work/runs_deg_per_second/execution_logs/htm_seed_{seed}.log")
    for name in ["source_unit_evidence.json", "feature_sensitivity.json", "original_checkpoint_verification.json"]:
        add(BASE / "work" / name)
    for p in (BASE / "work/runs_deg_per_second/execution_logs").glob("*.log"):
        add(p)
    for name in ["README.md", "requirements.txt", "audit_environment.json"]:
        source = BASE / "package_metadata" / name
        add(source if source.exists() else BASE / name, name)

    hashes = {relative: hashlib.sha256(path.read_bytes()).hexdigest()
              for relative, path in sorted(paths.items())}
    manifest = {
        "study": "Gyroscope-unit sensitivity on the fixed CERTH agricultural HAR split",
        "source_archive_sha256": summary["feature_sensitivity"]["source_archive_sha256"],
        "verified_model_seed_scenario_outcomes": 36,
        "reference": "Unchanged archived rad/s results; verified features/checkpoints",
        "alternative": "New matched fits with degree-to-radian conversion",
        "htm_source_commit": "96309f4969b1488ede172b3eec649dcb9702557d",
        "sha256": hashes,
    }
    OUT.mkdir(exist_ok=True)
    destination = OUT / "Farmer_Gyroscope_Unit_Audit.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for relative, path in sorted(paths.items()):
            z.write(path, relative)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    with ZipFile(destination) as z:
        assert z.testzip() is None
        for relative, digest in hashes.items():
            assert hashlib.sha256(z.read(relative)).hexdigest() == digest, relative
    assert destination.stat().st_size < 50 * 2**20
    print(json.dumps({"file": str(destination), "size_bytes": destination.stat().st_size,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                      "included_files": len(paths) + 1}, indent=2))


if __name__ == "__main__":
    main()
