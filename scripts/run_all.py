#!/usr/bin/env python
"""Reproduce every result and figure of the repository.

By default this runs offline on the pseudobulk file shipped in data/. With --from-raw
the raw dataset is downloaded (about 700 MB, skipped when cached), the pseudobulk file
and data/SHA256SUMS are rebuilt, and the analysis without the cell filter is added
(results/sensitivity_all_cells.csv). Timings and the software environment are written
to results/runtime.json (offline) or results/runtime_from_raw.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from pbgate.data import content_sha256, default_data_path, load_pseudobulk, sha256_of

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_STEPS = [
    "run_baselines.py",
    "analyze_additivity.py",
    "analyze_ranking.py",
    "analyze_sensitivity.py",
    "gate_demo.py",
    "review_checks.py",
    "make_figures.py",
]
RAW_STEPS = ["download_data.py", "preprocess_pseudobulk.py"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--from-raw", action="store_true", help="download the raw data and rebuild data/ first"
    )
    args = parser.parse_args()

    cache = Path(os.environ.get("PBGATE_CACHE", REPO_ROOT / "data" / "raw"))
    all_cells = cache / "all_cells"
    steps = [(step, []) for step in OFFLINE_STEPS]
    if args.from_raw:
        # the unfiltered pseudobulk is a sensitivity analysis; it stays in the cache folder
        steps = [
            *[(step, []) for step in RAW_STEPS],
            ("preprocess_pseudobulk.py", ["--keep-all-cells", "--out-dir", str(all_cells)]),
            *steps,
        ]
    timings = {}
    start = time.time()
    for step, extra in steps:
        if args.from_raw and step == "analyze_sensitivity.py":
            extra = ["--all-cells-data", str(all_cells / "norman2019_pseudobulk.npz")]
        label = " ".join([step, *extra[:1]])
        print(f"\n=== {label} ===", flush=True)
        tic = time.time()
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / step), *extra], check=True, cwd=REPO_ROOT)
        timings[label] = round(time.time() - tic, 1)
    if args.from_raw:
        sums = REPO_ROOT / "data" / "SHA256SUMS"
        data = load_pseudobulk()
        sums.write_text(
            f"{sha256_of(default_data_path())}  norman2019_pseudobulk.npz\n"
            f"{content_sha256(data)}  norman2019_pseudobulk.npz:content\n"
        )
    total = round(time.time() - start, 1)
    info = {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "from_raw": args.from_raw,
        "seconds_per_step": timings,
        "seconds_total": total,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib")},
    }
    name = "runtime_from_raw.json" if args.from_raw else "runtime.json"
    (REPO_ROOT / "results" / name).write_text(json.dumps(info, indent=1) + "\n")
    print(f"\nall steps finished in {total} s")


if __name__ == "__main__":
    main()
