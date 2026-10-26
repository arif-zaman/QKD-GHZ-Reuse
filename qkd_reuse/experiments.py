"""The manuscript's fixed experiment grid and prespecified seed check."""
from __future__ import annotations

import csv
import time

from . import simulation
from .data import REFERENCE, new_output, provenance, read_json, verify_reference, write_json
from .statistics import summarize, validate_record

# This order assigns seeds 20261001 + index to the paper's configurations.
GRID_CONFIGS = [
    (length, strength, noise, "coherent", 1)
    for length in (2, 5, 8, 12)
    for strength in (0.0, 0.5, 1.0)
    for noise in (0.0, 0.005)
] + [(5, 0.5, 0.0, "mixed", 1), (5, 0.5, 0.0, "coherent", 2),
     (5, 0.5, 0.0, "coherent", 4)]
CONFIG_FIELDS = ("length", "strength", "noise", "reset_mode", "max_reset_attempts")
SUMMARY_FIELDS = [
    *CONFIG_FIELDS, "seed", "attempted_blocks", "accepted_blocks",
    "block_acceptance", "raw_bits_per_transmission", "accepted_raw_error_rate",
    "accepted_raw_error_standard_error_by_block", "reset_attempts",
    "quantum_transmissions", "classical_protocol_bits",
]


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_grid(output, attempts=100000):
    """Run all 27 paper settings in order, using their recorded seeds.

    A smaller attempt count is useful for a smoke run but does not reproduce
    the paper. Random draws depend on the batch size; do not split a run.
    Completed rows are saved after each configuration, even if a later run fails.
    """
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    output = new_output(output)
    metadata = provenance()
    data = {
        "schema": 1, "complete": False,
        "interpretation": "Physical readout/reset analysis; no certified secret key",
        "source_sha256": metadata["source_sha256"]["simulation.py"],
        "provenance": metadata, "independent_attempts_per_configuration": attempts,
        "results": [],
    }
    start = time.perf_counter()
    write_json(output / "results.json", data)
    for index, values in enumerate(GRID_CONFIGS):
        config = dict(zip(CONFIG_FIELDS, values))
        record = simulation.experiment(**config, attempts=attempts, seed=20261001 + index)
        data["results"].append(record)
        write_json(output / "results.json", data)
        print(f"Grid {index + 1:02d}/27: L={record['length']}, g={record['strength']}, "
              f"p={record['noise']}, {record['reset_mode']}, R={record['max_reset_attempts']}; "
              f"accepted {record['accepted_blocks']}/{attempts}", flush=True)
    data.update(complete=True, elapsed_seconds=time.perf_counter() - start)
    write_json(output / "results.json", data)
    write_csv(output / "summary.csv", data["results"], SUMMARY_FIELDS)
    return data


def run_seed_check(output):
    """Run all 20 planned seeds once, without changing the reference records."""
    verify_reference()
    plan = read_json(REFERENCE / "seed-plan.json")
    output = new_output(output)
    data = {
        "schema_version": 1, "complete": False, "plan": plan,
        "provenance": provenance(), "runs": [],
    }
    config = {key: value for key, value in plan["config"].items() if key != "delta"}
    start = time.perf_counter()
    write_json(output / "results.json", data)
    for index, seed in enumerate(plan["seeds"], start=1):
        record = simulation.experiment(**config, seed=seed)
        item = {"seed": seed, "result": record}
        data["runs"].append(item)
        # Keep the record even when its accounting check fails.
        write_json(output / "results.json", data)
        item["counter_checks"] = validate_record(record, plan["config"], seed)
        write_json(output / "results.json", data)
        print(f"Seed {index:02d}/20: {seed}; accepted "
              f"{record['accepted_blocks']}/{record['attempted_blocks']}", flush=True)
    probability = ((1 - plan["config"]["delta"] ** 2) / 2) ** (config["length"] - 1)
    data["summary"] = summarize([item["result"] for item in data["runs"]], probability)
    data.update(complete=True, elapsed_seconds=time.perf_counter() - start)
    write_json(output / "results.json", data)
    write_csv(output / "per-seed.csv", [item["result"] for item in data["runs"]], SUMMARY_FIELDS)
    return data
