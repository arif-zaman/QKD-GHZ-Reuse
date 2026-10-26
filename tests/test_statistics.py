"""Counter identities and descriptive statistics; no simulation runs required."""

import copy
import unittest

from qkd_reuse.statistics import summarize, validate_record, wilson


def fixture():
    """Ten blocks enter; one survives all seven reset stages."""
    config = {
        "length": 8, "strength": .5, "noise": 0., "delta": .2,
        "reset_mode": "coherent", "max_reset_attempts": 1,
        "attempts": 10, "test_fraction": 0., "link_success": 1.,
    }
    record = {
        "length": 8, "strength": .5, "noise": 0., "reset_mode": "coherent",
        "max_reset_attempts": 1, "test_fraction": 0., "link_success": 1.,
        "seed": 7, "attempted_blocks": 10, "delivered_blocks": 10,
        "reserved_test_blocks": 0, "discarded_data_blocks": 9,
        "accepted_blocks": 1, "accepted_bits": 8, "quantum_transmissions": 10,
        "ghz_preparations": 10, "ghz_qubits_initialized": 90,
        "reached_by_round": [10, 6, 4, 2, 2, 1, 1, 1],
        "bsm_rounds": 27, "readouts": 27, "encoding_ancillas": 27,
        "reset_attempts": 26, "reset_ancillas": 26, "successful_reset_stages": 17,
        "classical_protocol_bits": 80, "peak_alice_qubits_serial_schedule": 9,
        "peak_bob_qubits": 1, "accepted_error_by_round": [0, 0, 0, 0, 0, 0, 0, 1],
        "errors_at_readout_by_round": [3, 2, 2, 1, 1, 0, 0, 1],
        "accepted_raw_bit_errors": 1, "fully_correct_accepted_blocks": 0,
        "block_acceptance": .1, "raw_bits_per_transmission": .8,
        "accepted_raw_error_rate": .125,
        "accepted_raw_error_standard_error_by_block": None,
        "block_acceptance_ci95": [0.017876213095072924, 0.4041500267952385],
    }
    return config, record


class WilsonTests(unittest.TestCase):
    def test_known_interval_and_boundary_counts(self):
        lo, hi = wilson(1, 10)
        self.assertAlmostEqual(lo, .017876213095072924)
        self.assertAlmostEqual(hi, .4041500267952385)
        self.assertAlmostEqual(wilson(0, 10)[0], 0.)
        self.assertAlmostEqual(wilson(10, 10)[1], 1.)
        self.assertAlmostEqual(wilson(0, 10)[1], 1 - wilson(10, 10)[0])

    def test_invalid_binomial_counts(self):
        for successes, trials in ((0, 0), (-1, 10), (11, 10), (1.0, 10), (True, 10)):
            with self.subTest(counts=(successes, trials)), self.assertRaises(ValueError):
                wilson(successes, trials)


class CounterTests(unittest.TestCase):
    def test_valid_block_accounting_and_each_stage(self):
        config, record = fixture()
        checks = validate_record(record, config, 7)
        self.assertEqual([s["discarded"] for s in checks["reset_stages"]],
                         [4, 2, 2, 0, 1, 0, 0])
        self.assertEqual(sum(s["discarded"] for s in checks["reset_stages"]), 9)
        self.assertEqual(checks["final_round_entered"], 1)

    def test_counter_corruption_is_rejected(self):
        config, record = fixture()
        for key in ("quantum_transmissions", "reset_attempts", "successful_reset_stages",
                    "classical_protocol_bits", "accepted_raw_bit_errors"):
            bad = copy.deepcopy(record)
            bad[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_record(bad, config, 7)
        record["reached_by_round"][2] = 7
        with self.assertRaisesRegex(ValueError, "nonincreasing"):
            validate_record(record, config, 7)

    def test_wrong_rate_denominators_and_correct_block_count_are_rejected(self):
        config, record = fixture()
        for key, value in (("block_acceptance", .2), ("raw_bits_per_transmission", .1),
                           ("accepted_raw_error_rate", 1.), ("fully_correct_accepted_blocks", 1)):
            bad = dict(record, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_record(bad, config, 7)

    def test_no_accepted_blocks_have_null_error_and_se(self):
        config, record = fixture()
        record.update({
            "reached_by_round": [10, 0, 0, 0, 0, 0, 0, 0],
            "accepted_blocks": 0, "accepted_bits": 0, "discarded_data_blocks": 10,
            "bsm_rounds": 10, "readouts": 10, "encoding_ancillas": 10,
            "reset_attempts": 10, "reset_ancillas": 10, "successful_reset_stages": 0,
            "classical_protocol_bits": 30, "accepted_error_by_round": [0] * 8,
            "errors_at_readout_by_round": [3, 0, 0, 0, 0, 0, 0, 0],
            "accepted_raw_bit_errors": 0, "block_acceptance": 0.,
            "raw_bits_per_transmission": 0., "accepted_raw_error_rate": None,
            "block_acceptance_ci95": [0., 0.2775327998628892],
        })
        self.assertTrue(validate_record(record, config, 7)["passed"])
        for key in ("accepted_raw_error_rate", "accepted_raw_error_standard_error_by_block"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_record(dict(record, **{key: 0.}), config, 7)

    def test_se_requires_two_accepted_blocks(self):
        config, record = fixture()
        record["accepted_raw_error_standard_error_by_block"] = 0.
        with self.assertRaisesRegex(ValueError, "at least two"):
            validate_record(record, config, 7)

        # Two surviving blocks with three and five erroneous bits respectively.
        record.update({
            "reached_by_round": [10, 6, 4, 3, 3, 2, 2, 2],
            "accepted_blocks": 2, "accepted_bits": 16, "discarded_data_blocks": 8,
            "bsm_rounds": 32, "readouts": 32, "encoding_ancillas": 32,
            "reset_attempts": 30, "reset_ancillas": 30, "successful_reset_stages": 22,
            "classical_protocol_bits": 94,
            "accepted_error_by_round": [0, 1, 1, 1, 1, 1, 1, 2],
            "errors_at_readout_by_round": [3, 2, 2, 2, 2, 1, 1, 2],
            "accepted_raw_bit_errors": 8, "block_acceptance": .2,
            "raw_bits_per_transmission": 1.6, "accepted_raw_error_rate": .5,
            "accepted_raw_error_standard_error_by_block": .125,
            "block_acceptance_ci95": wilson(2, 10),
        })
        self.assertTrue(validate_record(record, config, 7)["passed"])
        for invalid in (None, -1., float("nan"), float("inf")):
            record["accepted_raw_error_standard_error_by_block"] = invalid
            with self.subTest(se=invalid), self.assertRaises(ValueError):
                validate_record(record, config, 7)

    def test_interval_and_unsupported_accounting_are_rejected(self):
        config, record = fixture()
        for interval in ([], [.1], [.1, .2]):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                validate_record(dict(record, block_acceptance_ci95=interval), config, 7)
        for key, value in (("max_reset_attempts", 2), ("link_success", .9), ("test_fraction", .1)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Counter audit requires"):
                validate_record(record, dict(config, **{key: value}), 7)


class SummaryTests(unittest.TestCase):
    def test_unbiased_between_run_variance_and_descriptive_coverage(self):
        rows = [{"attempted_blocks": 10, "accepted_blocks": k} for k in (0, 10, 5, 5)]
        result = summarize(rows, .5)
        self.assertEqual(result["pooled_acceptance"], .5)
        self.assertEqual(result["pooled_standardized_residual"], 0)
        self.assertAlmostEqual(result["unbiased_sample_variance_counts"], 50 / 3)
        self.assertAlmostEqual(result["expected_binomial_variance_counts"], 2.5)
        self.assertAlmostEqual(result["variance_ratio"], 20 / 3)
        self.assertEqual(result["individual_ci95_coverage_count"], 2)
        self.assertEqual(result["total_attempts"], 40)

    def test_zero_acceptance_is_valid_even_when_all_intervals_miss_theory(self):
        rows = [{"attempted_blocks": 10, "accepted_blocks": 0} for _ in range(3)]
        result = summarize(rows, .5)
        self.assertEqual(result["pooled_acceptance"], 0)
        self.assertEqual(result["unbiased_sample_variance_counts"], 0)
        self.assertEqual(result["individual_ci95_coverage_count"], 0)
        self.assertAlmostEqual(result["pooled_wilson_ci95"][0], 0)
        self.assertLess(result["pooled_wilson_ci95"][1], .5)

    def test_invalid_sample_sizes_and_expected_probability(self):
        row = {"attempted_blocks": 10, "accepted_blocks": 1}
        for rows, probability in (([row], .5), ([row, dict(row, attempted_blocks=20)], .5),
                                  ([row, row], 0), ([row, row], 1),
                                  ([row, dict(row, accepted_blocks=11)], .5)):
            with self.subTest(rows=rows, probability=probability), self.assertRaises(ValueError):
                summarize(rows, probability)


if __name__ == "__main__":
    unittest.main()
