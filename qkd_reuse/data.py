"""Saved-data integrity, output directories, and execution metadata."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Replace one JSON file atomically; reject nonfinite numeric output."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def new_output(path):
    """Require a new or empty output directory outside package inputs."""
    path = Path(path).resolve()
    protected = [ROOT / name for name in ("reference", "qkd_reuse", "tests", "docs")]
    if path == ROOT or path in ROOT.parents or any(
        path == item or item in path.parents for item in protected
    ):
        raise ValueError("Choose an output directory outside the source and reference folders")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def verify_reference():
    """Check the saved paper records against their SHA-256 digests."""
    manifest = read_json(REFERENCE / "manifest.json")
    for name, record in manifest["files"].items():
        if sha256(REFERENCE / name) != record["sha256"]:
            raise ValueError(f"Reference file has changed: {name}")
    return manifest


def provenance():
    """Record the code, runtime, and RNG used by a new execution."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "rng": "numpy.random.default_rng / PCG64",
        "source_sha256": {
            path.name: sha256(path)
            for path in sorted((ROOT / "qkd_reuse").glob("*.py"))
        },
    }


def load_results(path, reference_name):
    """Load saved paper data or a completed run from this simulator."""
    path = Path(path)
    manifest = verify_reference()
    data = read_json(path)
    original = sha256(path) == manifest["files"][reference_name]["sha256"]
    if not original:
        if data.get("complete") is not True:
            raise ValueError("A fresh run must be complete before generating paper outputs")
        recorded = data.get("provenance", {}).get("source_sha256", {})
        if recorded.get("simulation.py") != sha256(ROOT / "qkd_reuse/simulation.py"):
            raise ValueError("Result source hash does not match this simulator")
    return data
