"""Generate the two sampled manuscript tables from a complete experiment grid.

Validation is performed before writing either file. Source integrity is the
caller's responsibility; this module validates the data needed by the tables.
"""
from __future__ import annotations

import math
from pathlib import Path


_CONFIG_FIELDS = ("length", "strength", "noise", "reset_mode", "max_reset_attempts")
_EXPECTED_CONFIGS = [
    (length, strength, noise, "coherent", 1)
    for length in (2, 5, 8, 12)
    for strength in (0.0, 0.5, 1.0)
    for noise in (0.0, 0.005)
] + [(5, 0.5, 0.0, "mixed", 1), (5, 0.5, 0.0, "coherent", 2),
     (5, 0.5, 0.0, "coherent", 4)]
_ATTEMPTS = 100000


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _number(row, key, prefix, lower, upper):
    value = row.get(key)
    if not _finite(value) or not lower <= value <= upper:
        raise ValueError(f"{prefix}{key} must be a finite number in [{lower}, {upper}]")
    return value


def _integer(row, key, prefix, lower, upper):
    value = row.get(key)
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{prefix}{key} must be an integer in [{lower}, {upper}]")
    return value


def validate_manuscript_grid(data):
    """Return all 27 rows in manuscript order, rejecting unsupported inputs.

    Each configuration must have 100,000 attempted blocks, ideal delivery,
    and no reserved tests. Smoke runs and incomplete results cannot supply
    manuscript tables. All displayed probabilities, counts, intervals, and
    uncertainties must be finite and have valid ranges. A confidence interval
    is not required to contain the exact analytical probability.
    """
    if (not isinstance(data, dict) or type(data.get("schema")) is not int
            or data["schema"] != 1):
        raise ValueError("Expected manuscript-grid schema 1")
    if "complete" in data and data["complete"] is not True:
        raise ValueError("Manuscript tables require a complete experiment grid")
    attempts = data.get("independent_attempts_per_configuration")
    if type(attempts) is not int or attempts != _ATTEMPTS:
        raise ValueError("Tables require 100,000 attempts per configuration; smoke runs are unsupported")
    rows = data.get("results")
    if not isinstance(rows, list) or len(rows) != len(_EXPECTED_CONFIGS):
        raise ValueError("Tables require the complete 27-configuration manuscript grid")

    by_config = {}
    for index, row in enumerate(rows, start=1):
        prefix = f"Manuscript-grid row {index}: "
        if not isinstance(row, dict):
            raise ValueError(prefix + "must be a JSON object")
        length = _integer(row, "length", prefix, 1, 12)
        retries = _integer(row, "max_reset_attempts", prefix, 1, 4)
        _number(row, "strength", prefix, 0, 1)
        _number(row, "noise", prefix, 0, 1)
        if row.get("reset_mode") not in ("coherent", "mixed"):
            raise ValueError(prefix + "reset_mode must be coherent or mixed")
        config = tuple(row[key] for key in _CONFIG_FIELDS)
        if config not in _EXPECTED_CONFIGS:
            raise ValueError(prefix + "unsupported manuscript configuration")
        if config in by_config:
            raise ValueError(prefix + "duplicate manuscript configuration")

        _integer(row, "attempted_blocks", prefix, _ATTEMPTS, _ATTEMPTS)
        _number(row, "test_fraction", prefix, 0, 0)
        _number(row, "link_success", prefix, 1, 1)
        _integer(row, "accepted_blocks", prefix, 1, _ATTEMPTS)
        _integer(row, "quantum_transmissions", prefix, _ATTEMPTS, _ATTEMPTS)
        _integer(row, "reset_attempts", prefix, 0, _ATTEMPTS * (length - 1) * retries)
        _number(row, "block_acceptance", prefix, 0, 1)
        _number(row, "raw_bits_per_transmission", prefix, 0, length)
        _number(row, "accepted_raw_error_rate", prefix, 0, 1)
        _number(row, "accepted_raw_error_standard_error_by_block", prefix, 0, 1)
        interval = row.get("block_acceptance_ci95")
        if (not isinstance(interval, list) or len(interval) != 2
                or not all(_finite(value) for value in interval)
                or not 0 <= interval[0] <= interval[1] <= 1):
            raise ValueError(prefix + "acceptance CI must have two finite, ordered endpoints in [0, 1]")
        if retries == 1:
            _number(row, "exact_block_acceptance", prefix, 0, 1)
            _number(row, "exact_accepted_raw_error_rate", prefix, 0, 1)
        by_config[config] = row
    return [by_config[config] for config in _EXPECTED_CONFIGS]


def _sampled_table(rows):
    lines = [
        r'\begin{table*}[t]',
        r'\caption{Exact and sampled results for single-attempt coherent reset.}',
        r'\label{tab:sampled}', r'\centering',
        r'\begin{tabular}{rrrrrrrrr}', r'\toprule',
        r'$L$ & $p$ & Accepted & $P_{\rm acc}$ & 95\% CI for $P_{\rm acc}$ & $\widehat\eta_{\rm raw}$ & $Q_{\rm raw}$ & $\widehat Q_{\rm raw}$ & SE (pp) \\',
        r'\midrule',
    ]
    for row in rows:
        if (row["strength"], row["reset_mode"], row["max_reset_attempts"]) != (0.5, "coherent", 1):
            continue
        lower, upper = row["block_acceptance_ci95"]
        fields = [
            str(row["length"]), f'{row["noise"]:.3f}',
            f'{row["accepted_blocks"]:,}', f'{row["exact_block_acceptance"]:.6f}',
            f'[{lower:.5f}, {upper:.5f}]', f'{row["raw_bits_per_transmission"]:.5f}',
            f'{100 * row["exact_accepted_raw_error_rate"]:.2f}' + r'\%',
            f'{100 * row["accepted_raw_error_rate"]:.2f}' + r'\%',
            f'{100 * row["accepted_raw_error_standard_error_by_block"]:.2f}',
        ]
        lines.append(" & ".join(fields) + r' \\')
    lines += [
        r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
        r'\begin{minipage}{\textwidth}', r'\footnotesize\raggedright',
        r'$\delta=0.2$, $g=0.5$; $10^5$ independent block attempts per row. $P_{\rm acc}$ and $Q_{\rm raw}$ are exact analytical values; $\widehat\eta_{\rm raw}$ and $\widehat Q_{\rm raw}$ are Monte Carlo estimates. $\eta_{\rm raw}$ counts retained labels per transmitted qubit. CI: pointwise 95\% Wilson confidence interval for acceptance. Raw-error standard errors are computed across independent accepted blocks and reported in percentage points (pp).',
        r'\end{minipage}', r'\end{table*}',
    ]
    return "\n".join(lines) + "\n"


def _retry_table(rows):
    lines = [
        r'\begin{table}[t]',
        r'\caption{Sampled performance under alternative reset policies.}',
        r'\label{tab:retry}', r'\centering', r'\begin{tabular}{lrrrrr}', r'\toprule',
        r'Ancilla & $R$ & $\widehat P_{\rm acc}$ & $\overline n_F$ & $\widehat Q_{\rm raw}$ & SE (pp) \\',
        r'\midrule',
    ]
    for row in rows:
        if (row["length"], row["strength"], row["noise"]) != (5, 0.5, 0.0):
            continue
        fields = [
            row["reset_mode"].capitalize(), str(row["max_reset_attempts"]),
            f'{row["block_acceptance"]:.4f}',
            f'{row["reset_attempts"] / row["quantum_transmissions"]:.3f}',
            f'{100 * row["accepted_raw_error_rate"]:.2f}' + r'\%',
            f'{100 * row["accepted_raw_error_standard_error_by_block"]:.2f}',
        ]
        lines.append(" & ".join(fields) + r' \\')
    lines += [
        r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
        r'\begin{minipage}{\columnwidth}', r'\footnotesize\raggedright',
        r'$\delta=0.2$, $L=5$, $g=0.5$, $p=0$; $10^5$ block attempts per row. Coherent rows use the fixed filter. $R$: maximum reset attempts per stage; $\overline n_F$: sample mean reset attempts per transmitted qubit. Hats denote Monte Carlo estimates. SE uses independent accepted blocks; pp: percentage points.',
        r'\end{minipage}', r'\end{table}',
    ]
    return "\n".join(lines) + "\n"


def generate_tables(data, output_dir):
    """Write Table II and Table III after validating the entire loaded grid.

    Returns the two output paths. Existing files with these names are replaced;
    no other output files are changed. The input data are not modified.
    """
    rows = validate_manuscript_grid(data)
    contents = {
        "diagnostic-results.tex": _sampled_table(rows),
        "reset-policy.tex": _retry_table(rows),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, text in contents.items():
        path = output_dir / filename
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
