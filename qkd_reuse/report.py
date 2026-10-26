"""Regenerate manuscript outputs from saved or newly simulated records."""
from __future__ import annotations

from .analysis import amplitude_updated_predictions, fixed_filter_predictions
from .data import (REFERENCE, load_results, new_output, provenance, read_json,
                   sha256, write_json)
from .experiments import SUMMARY_FIELDS, write_csv
from .statistics import summarize, validate_record
from .tables import generate_tables, validate_manuscript_grid


def seed_summary(data):
    """Recompute the full planned seed check, including accounting identities."""
    plan = read_json(REFERENCE / "seed-plan.json")
    if data.get("plan") != plan:
        raise ValueError("Seed-check data must use the original prespecified plan")
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
    """The exact fixed-filter versus amplitude-updated comparison in Section V-C."""
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
    The output directory must be new or empty. Paper inputs are never replaced.
    """
    grid_path = grid_path or REFERENCE / "grid.json"
    seed_path = seed_path or REFERENCE / "seed-check.json"
    grid = load_results(grid_path, "grid.json")
    seed_data = load_results(seed_path, "seed-check.json")
    validate_manuscript_grid(grid)
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
        "complete": True, "operation": "regenerate from recorded experiments",
        "inputs": {"grid_sha256": sha256(grid_path),
                   "seed_check_sha256": sha256(seed_path)},
        "provenance": provenance(),
    })
    low, high = summary["pooled_wilson_ci95"]
    text = f"""# Reproduced results

The two LaTeX tables are in `tables/`. `tradeoff.csv` supplies Figure 1's
analytical curves; its conditional endpoint at delta=1 is undefined (blank).
`retry-comparison.csv` and `.json` give the exact policy comparison for L=5,
delta=0.2, g=0.5, p=0, with R=1,2,4. Costs include rejected blocks.

## Independent seed check

- Runs: {summary['number_of_runs']}; total attempts: {summary['total_attempts']:,}.
- Accepted blocks: {summary['total_accepted']:,}.
- Exact acceptance: {summary['expected_acceptance']:.12g}.
- Pooled acceptance: {summary['pooled_acceptance']:.12g}.
- Pooled 95% Wilson interval: [{low:.12g}, {high:.12g}].
- Sample/binomial count variance ratio: {summary['variance_ratio']:.8g}.
- Individual intervals containing theory: {summary['individual_ci95_coverage_count']}/20.

All per-stage and total counter checks passed. Coverage and variance are
descriptive, not pass/fail criteria. Per-seed records are in `per-seed.csv`;
full summary statistics are in `seed-summary.json`.

These outputs concern readout, postselection, and resource costs. They do not
certify secrecy or a positive secret-key rate. The manifest records input and
code hashes; this command does not run fresh simulations.
"""
    (output / "README.md").write_text(text, encoding="utf-8")
    return output
