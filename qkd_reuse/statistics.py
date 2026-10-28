"""Independent counter and statistical checks for the acceptance seed study.

Confidence-interval coverage and variance ratios are descriptive diagnostics,
not pass/fail tests. This module does not import the physical simulator.
"""

import math
import statistics


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def wilson(successes, trials):
    """Return the pointwise 95% Wilson interval for integer binomial counts."""
    _require(type(trials) is int and trials > 0
             and type(successes) is int and 0 <= successes <= trials,
             "Expected integer binomial counts with 0 <= successes <= trials")
    z = 1.959963984540054
    estimate = successes / trials
    denominator = 1 + z * z / trials
    center = (estimate + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(estimate * (1 - estimate) / trials
                         + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def validate_record(record, config, seed):
    """Audit one seed-study record, returning per-stage accounting checks.

    The accounting assumes one reset attempt per stage, perfect link delivery,
    and no reserved test blocks. Every accepted block reaches the final round.
    Error uncertainty uses independent accepted blocks, not individual bits.
    Aggregate records allow checks of SE availability and finiteness, but do
    not contain the block-level errors needed to recompute its value.

    A counter or statistic inconsistency raises ValueError. An interval that
    misses the theoretical probability is valid data and is never rejected.
    """
    n, length = config["attempts"], config["length"]
    _require(type(n) is int and n > 0 and type(length) is int and length > 0,
             "Attempts and block length must be positive integers")
    _require(config["max_reset_attempts"] == 1 and config["link_success"] == 1
             and config["test_fraction"] == 0,
             "Counter audit requires one reset attempt, perfect delivery, and no test blocks")
    for key in ("length", "strength", "noise", "reset_mode", "max_reset_attempts",
                "test_fraction", "link_success"):
        _require(record[key] == config[key], f"Unexpected result {key}")
    _require(record["seed"] == seed, "Unexpected result seed")
    _require(record["attempted_blocks"] == n, "Unexpected attempted-block count")

    reached = record["reached_by_round"]
    _require(len(reached) == length, "Wrong number of reached-round counters")
    _require(all(type(value) is int and 0 <= value <= n for value in reached),
             "Invalid reached-round counter")
    _require(reached[0] == n, "Every delivered data block must reach round one")
    _require(all(a >= b for a, b in zip(reached, reached[1:])),
             "Reached-round counts must be nonincreasing")
    accepted = record["accepted_blocks"]
    _require(type(accepted) is int and accepted == reached[-1],
             "Accepted blocks must equal final-round entrants")
    rounds, resets = sum(reached), sum(reached[:-1])
    expected = {
        "delivered_blocks": n, "reserved_test_blocks": 0,
        "discarded_data_blocks": n - accepted,
        "accepted_bits": length * accepted, "quantum_transmissions": n,
        "ghz_preparations": n, "ghz_qubits_initialized": (length + 1) * n,
        "bsm_rounds": rounds, "readouts": rounds, "encoding_ancillas": rounds,
        "reset_attempts": resets, "reset_ancillas": resets,
        "successful_reset_stages": sum(reached[1:]),
        "classical_protocol_bits": 2 * rounds + resets,
        "peak_alice_qubits_serial_schedule": length + 1, "peak_bob_qubits": 1,
    }
    for key, value in expected.items():
        _require(type(record[key]) is int and record[key] == value,
                 f"Counter mismatch: {key} (expected {value}, got {record[key]})")
    stages = [{"stage": i + 1, "entered": entered, "accepted": kept,
               "discarded": entered - kept}
              for i, (entered, kept) in enumerate(zip(reached, reached[1:]))]

    errors = record["accepted_error_by_round"]
    observed = record["errors_at_readout_by_round"]
    _require(len(errors) == length and len(observed) == length,
             "Wrong number of error counters")
    for i, (accepted_errors, readout_errors) in enumerate(zip(errors, observed)):
        _require(type(accepted_errors) is int and 0 <= accepted_errors <= accepted,
                 f"Invalid accepted errors in round {i + 1}")
        _require(type(readout_errors) is int and accepted_errors <= readout_errors <= reached[i],
                 f"Invalid readout errors in round {i + 1}")
    _require(type(record["accepted_raw_bit_errors"]) is int
             and record["accepted_raw_bit_errors"] == sum(errors),
             "Accepted bit-error total is inconsistent")
    correct = record["fully_correct_accepted_blocks"]
    _require(type(correct) is int and 0 <= correct <= accepted,
             "Invalid fully-correct-block counter")
    wrong = accepted - correct
    _require(wrong <= sum(errors) <= length * wrong,
             "Fully-correct blocks disagree with the bit-error total")

    for key, value in {"block_acceptance": accepted / n,
                       "raw_bits_per_transmission": accepted * length / n}.items():
        _require(math.isclose(record[key], value, rel_tol=1e-12, abs_tol=1e-14),
                 f"Wrong denominator for {key}")
    error_rate = record["accepted_raw_error_rate"]
    if accepted:
        _require(error_rate is not None
                 and math.isclose(error_rate, sum(errors) / (accepted * length),
                                  rel_tol=1e-12, abs_tol=1e-14), "Wrong raw-error rate")
    else:
        _require(error_rate is None, "Raw-error rate must be null when no block survives")
    se = record["accepted_raw_error_standard_error_by_block"]
    if accepted < 2:
        _require(se is None, "Block SE requires at least two accepted blocks")
    else:
        _require(se is not None and math.isfinite(se) and se >= 0,
                 "Invalid accepted-block SE")
    interval = record["block_acceptance_ci95"]
    _require(len(interval) == 2, "Acceptance interval must have two endpoints")
    _require(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
                 for a, b in zip(interval, wilson(accepted, n))),
             "Acceptance Wilson interval is inconsistent")
    return {"passed": True, "reset_stages": stages,
            "final_round_entered": reached[-1], "expected_totals": expected}


def summarize(records, probability):
    """Pool equal-sized independent runs and compare their count variation.

    ``probability`` is the analytical block-acceptance probability. Between-run
    variance uses the unbiased sample estimate (denominator runs minus one).
    Individual interval coverage is reported without imposing a coverage test.
    """
    _require(len(records) >= 2, "Variance analysis requires at least two runs")
    n = records[0]["attempted_blocks"]
    _require(type(n) is int and n > 0
             and all(type(r["attempted_blocks"]) is int and r["attempted_blocks"] == n
                     for r in records), "All runs must have the same positive attempt count")
    _require(0 < probability < 1, "Expected acceptance must lie strictly between zero and one")
    counts = [r["accepted_blocks"] for r in records]
    _require(all(type(k) is int and 0 <= k <= n for k in counts), "Invalid accepted counts")
    total, accepted = n * len(counts), sum(counts)
    expected_variance = n * probability * (1 - probability)
    variance = statistics.variance(counts)
    intervals = [wilson(k, n) for k in counts]
    coverage = sum(lo <= probability <= hi for lo, hi in intervals)
    return {
        "number_of_runs": len(counts), "attempts_per_run": n,
        "expected_acceptance": probability, "total_attempts": total,
        "total_accepted": accepted, "pooled_acceptance": accepted / total,
        "pooled_wilson_ci95": wilson(accepted, total),
        "pooled_standardized_residual": (accepted - total * probability)
            / math.sqrt(total * probability * (1 - probability)),
        "observed_mean_count": statistics.mean(counts), "expected_mean_count": n * probability,
        "unbiased_sample_variance_counts": variance,
        "expected_binomial_variance_counts": expected_variance,
        "variance_ratio": variance / expected_variance,
        "individual_ci95_coverage_count": coverage,
        "individual_ci95_coverage_fraction": coverage / len(counts),
    }
