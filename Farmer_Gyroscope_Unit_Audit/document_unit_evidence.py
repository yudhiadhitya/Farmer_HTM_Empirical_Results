"""Record the conflicting labels from the unchanged primary dataset release."""
import csv
import hashlib
import importlib.metadata
import inspect
import io
import json
from pathlib import Path
from zipfile import ZipFile

from ahrs.filters import Madgwick

BASE = Path(__file__).resolve().parent


def main():
    source = BASE / "original/empirical_source_data"
    archive = source / "certh_original_activity.zip"
    inventory = json.loads((BASE / "original/empirical_runs/inventory.json").read_text())
    headers = set()
    with ZipFile(archive) as z:
        readme_member = next(n for n in z.namelist() if Path(n).name == "README.txt")
        readme = z.read(readme_member)
        assert readme == (source / "certh_README.txt").read_bytes()
        text = readme.decode("utf-8")
        quote = "Angular velocity is measured in rad/s."
        assert quote in text
        for record in inventory["recordings"]:
            payload = z.read(record["source_member"])
            header = next(csv.reader(io.StringIO(payload.decode("utf-8"))))
            assert all(k in header for k in ["gx_deg/s", "gy_deg/s", "gz_deg/s"])
            headers.add(tuple(header))
    assert importlib.metadata.version("ahrs") == "0.4.0"
    assert "angular velocity in rad/s" in Madgwick.__doc__
    result = {
        "institutional_release": "https://ibo.certh.gr/open-datasets/",
        "source_paper_doi": "10.3390/app11052188",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "readme_member": readme_member,
        "readme_sha256": hashlib.sha256(readme).hexdigest(),
        "readme_unit_statement": quote,
        "readme_matches_included_copy_exactly": True,
        "csv_headers_in_selected_chest_recordings": [list(h) for h in sorted(headers)],
        "chest_recordings_with_degree_per_second_headers": len(inventory["recordings"]),
        "filter": {"package": "AHRS", "version": "0.4.0", "api": "Madgwick",
                   "gyro_input_unit": "rad/s", "pinned_docstring_confirms_unit": True,
                   "implementation_file_sha256": hashlib.sha256(Path(inspect.getfile(Madgwick)).read_bytes()).hexdigest(),
                   "documentation": "https://ahrs.readthedocs.io/en/latest/filters/madgwick.html"},
        "sensor_documentation": {"url": "https://vicon-help.atlassian.net/wiki/spaces/LMAC/pages/8880610/Capture.U+Education+-+Learn+More",
                                 "hardware_gyroscope_range": "+/-2000 degrees/s",
                                 "inference_limit": "A hardware measurement range does not establish units in the released numeric arrays."},
        "decision": "Retain the archived README rad/s convention as primary; compare the conflicting CSV deg/s convention after pi/180 conversion.",
        "remaining_uncertainty": "No independent orientation ground truth or source-author clarification establishes the physical source unit.",
    }
    (BASE / "work").mkdir(exist_ok=True)
    (BASE / "work/source_unit_evidence.json").write_text(json.dumps(result, indent=2))
    print("Verified primary README and degree/s headers in all", len(inventory["recordings"]), "selected recordings.")


if __name__ == "__main__":
    main()
