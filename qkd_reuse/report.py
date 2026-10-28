"""Generate analysis outputs from completed local experiments."""
from __future__ import annotations

from .analysis import amplitude_updated_predictions, fixed_filter_predictions
from .data import ROOT, load_results, new_output, provenance, sha256, write_json
from .experiments import SUMMARY_FIELDS, seed_plan, write_csv
from .statistics import summarize, validate_record
from .tables import generate_tables, validate_grid


def seed_summary(data):
    """Recompute the full planned seed check, including accounting identities."""
    plan = seed_plan()
    if data.get("plan") != plan:
        raise ValueError("Seed-check data must use the configured study plan")
    runs = data.get("runs", [])
    if [item.get("seed") for item in runs] != plan["seeds"]:
        raise ValueError("Seed-check data must contain all 20 seeds in their planned order")
    records = [item["result"] for item in runs]
    for seed, record in zip(plan["seeds"], records):
        validate_record(record, plan["config"], seed)
    config = plan["config"]
    probability = ((1 - config["delta"] ** 2) / 2) ** (config["length"] - 1)
    return records, summarize(records, probability)


def retry_comparison():
    """Compare exact predictions for fixed-filter and amplitude-updated retries."""
    rows = []
    for policy, predict in (("fixed", fixed_filter_predictions),
                            ("amplitude-updated", amplitude_updated_predictions)):
        for attempts in (1, 2, 4):
            config = dict(length=5, delta=0.2, strength=0.5, noise=0.0,
                          max_attempts=attempts)
            rows.append({"policy": policy, **config, **predict(**config)})
    return rows


def reproduce(output, grid_path=None, seed_path=None, include_figure=True):
    """Regenerate tables, plot data, exact retry comparison, and seed summary.

    This reads existing experiment records; it does not run Monte Carlo.
    Run ``grid`` and ``seed-check`` first. The output directory must be new or
    empty; experiment inputs are never replaced.
    """
    grid_path = grid_path or ROOT / "output/grid/results.json"
    seed_path = seed_path or ROOT / "output/seed-check/results.json"
    try:
        grid = load_results(grid_path)
        seed_data = load_results(seed_path)
    except FileNotFoundError as error:
        raise ValueError(
            f"Missing experiment results: {error.filename}. "
            "Run 'python -m qkd_reuse grid' and 'python -m qkd_reuse seed-check' first, "
            "or supply completed results with --grid and --seed-check."
        ) from error
    validate_grid(grid)
    records, summary = seed_summary(seed_data)
    if include_figure:
        from .figure import make_figure
    output = new_output(output)
    generate_tables(grid, output / "tables")
    write_json(output / "seed-summary.json", summary)
    write_csv(output / "per-seed.csv", records, SUMMARY_FIELDS)
    comparison = retry_comparison()
    write_json(output / "retry-comparison.json", comparison)
    write_csv(output / "retry-comparison.csv", comparison, list(comparison[0]))
    tradeoff = [
        {"delta": index / 200, "strength": 0.5,
         "readout_accuracy": (1 + 0.5 * index / 200) / 2,
         "accepted_readout_accuracy": 0.5 if index < 200 else None,
         "reset_acceptance": (1 - (index / 200) ** 2) / 2}
        for index in range(201)
    ]
    write_csv(output / "tradeoff.csv", tradeoff, list(tradeoff[0]))
    if include_figure:
        make_figure(output / "readout-reset-tradeoff.pdf")
    write_json(output / "manifest.json", {
        "complete": True, "operation": "generate from completed experiments",
        "inputs": {"grid_sha256": sha256(grid_path),
                   "seed_check_sha256": sha256(seed_path)},
        "provenance": provenance(),
    })
    return output
