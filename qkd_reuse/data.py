"""Experiment result loading, output directories, and execution metadata."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


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
    protected = [ROOT / name for name in ("qkd_reuse", "tests")]
    if path == ROOT or path in ROOT.parents or any(
        path == item or item in path.parents for item in protected
    ):
        raise ValueError("Choose an output directory outside the source folders")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def load_results(path):
    """Load a completed run produced by the current simulator."""
    path = Path(path)
    data = read_json(path)
    if not isinstance(data, dict) or data.get("complete") is not True:
        raise ValueError("A run must be complete before generating analysis outputs")
    source_hashes = data.get("provenance", {}).get("source_sha256", {})
    if source_hashes.get("simulation.py") != sha256(ROOT / "qkd_reuse/simulation.py"):
        raise ValueError("Result source hash does not match this simulator")
    return data
