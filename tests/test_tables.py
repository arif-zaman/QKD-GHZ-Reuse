"""Check table formatting and reject incomplete or malformed experiment data."""
import copy
from pathlib import Path
import tempfile
import unittest

from qkd_reuse.tables import generate_tables, validate_grid


def example_grid():
    """Synthetic formatting inputs, independent of simulation results."""
    configurations = [
        (length, strength, noise, "coherent", 1)
        for length in (2, 5, 8, 12)
        for strength in (0.0, 0.5, 1.0)
        for noise in (0.0, 0.005)
    ] + [(5, 0.5, 0.0, "mixed", 1), (5, 0.5, 0.0, "coherent", 2),
         (5, 0.5, 0.0, "coherent", 4)]
    return {
        "schema": 1, "complete": True,
        "independent_attempts_per_configuration": 100000,
        "results": [{
            "length": length, "strength": strength, "noise": noise,
            "reset_mode": mode, "max_reset_attempts": retries,
            "attempted_blocks": 100000, "test_fraction": 0.0, "link_success": 1.0,
            "accepted_blocks": 1000, "quantum_transmissions": 100000,
            "reset_attempts": 100000, "block_acceptance": 0.01,
            "raw_bits_per_transmission": 0.01 * length,
            "accepted_raw_error_rate": 0.456789,
            "accepted_raw_error_standard_error_by_block": 0.001234,
            "block_acceptance_ci95": [0.0094, 0.0107],
            "exact_block_acceptance": 0.48 ** (length - 1),
            "exact_accepted_raw_error_rate": 0.5 - 0.1 * strength * (1 - noise) ** length / length,
        } for length, strength, noise, mode, retries in configurations],
    }


class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = example_grid()

    def test_table_formats_supplied_values_and_preserves_inputs(self):
        data = copy.deepcopy(self.example)
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_tables(data, directory)
            self.assertEqual({path.name for path in paths},
                             {"diagnostic-results.tex", "reset-policy.tex"})
            self.assertEqual(set(Path(directory).iterdir()), set(paths))
            sampled, retry = [path.read_text(encoding="utf-8") for path in paths]
        self.assertEqual(data, self.example)
        sampled_values = sampled.split('\\midrule\n')[1].split('\\bottomrule')[0].splitlines()
        retry_values = retry.split('\\midrule\n')[1].split('\\bottomrule')[0].splitlines()
        self.assertEqual(len(sampled_values), 8)
        self.assertEqual(len(retry_values), 4)
        self.assertEqual(sampled_values[0],
                         r'2 & 0.000 & 1,000 & 0.480000 & [0.00940, 0.01070] & 0.02000 & 47.50\% & 45.68\% & 0.12 \\')
        self.assertEqual(retry_values[0],
                         r'Coherent & 1 & 0.0100 & 1.000 & 45.68\% & 0.12 \\')
        self.assertEqual([line.split(" & ")[:2] for line in sampled_values],
                         [[str(length), noise] for length in (2, 5, 8, 12)
                          for noise in ("0.000", "0.005")])
        self.assertEqual([line.split(" & ")[:2] for line in retry_values],
                         [["Coherent", "1"], ["Mixed", "1"],
                          ["Coherent", "2"], ["Coherent", "4"]])
        self.assertIn(r'\caption{Exact and sampled results for single-attempt coherent reset.}', sampled)
        self.assertIn(r'\caption{Sampled performance under alternative reset policies.}', retry)
        self.assertIn(r'pointwise 95\% Wilson confidence interval', sampled)
        self.assertIn('independent accepted blocks', sampled)
        self.assertIn('Coherent rows use the fixed filter.', retry)
        # Intervals need not contain theory: accepting input must not select
        # measurements for agreement with the expected probability.
        row = data["results"][0]
        self.assertGreater(row["exact_block_acceptance"], row["block_acceptance_ci95"][1])

    def test_shuffled_grid_has_the_same_configuration_order(self):
        data = copy.deepcopy(self.example)
        data["results"].reverse()
        self.assertEqual(validate_grid(data), self.example["results"])
        with tempfile.TemporaryDirectory() as directory:
            original = generate_tables(self.example, Path(directory) / "original")
            shuffled = generate_tables(data, Path(directory) / "shuffled")
            self.assertEqual([p.read_bytes() for p in original],
                             [p.read_bytes() for p in shuffled])

    def test_rejects_unsupported_metadata_and_missing_duplicate_configurations(self):
        invalid = [None, [], {}, {"schema": True}, {"schema": 2}]
        for key, value in (("complete", False), ("complete", "true"),
                           ("independent_attempts_per_configuration", 100),
                           ("independent_attempts_per_configuration", 100000.0),
                           ("results", []), ("results", None)):
            data = copy.deepcopy(self.example)
            data[key] = value
            invalid.append(data)
        data = copy.deepcopy(self.example)
        data["results"][1] = copy.deepcopy(data["results"][0])
        invalid.append(data)
        data = copy.deepcopy(self.example)
        data["results"][0] = None
        invalid.append(data)
        for index, data in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_grid(data)

    def test_rejects_malformed_numbers_counts_intervals_and_configurations(self):
        cases = [
            ("length", 3), ("length", True), ("length", 2.0),
            ("strength", 0.7), ("strength", [0.5]), ("noise", float("nan")),
            ("reset_mode", "unknown"), ("max_reset_attempts", True),
            ("attempted_blocks", 100), ("attempted_blocks", 100000.0),
            ("test_fraction", 0.1), ("link_success", 0.9),
            ("accepted_blocks", -1), ("accepted_blocks", 100001),
            ("accepted_blocks", 0), ("accepted_blocks", 5.0),
            ("quantum_transmissions", 0), ("quantum_transmissions", True),
            ("reset_attempts", -1), ("reset_attempts", 100001),
            ("block_acceptance", 1.1), ("block_acceptance", -0.1),
            ("raw_bits_per_transmission", 3),
            ("accepted_raw_error_rate", -0.1),
            ("accepted_raw_error_standard_error_by_block", -0.1),
            ("exact_block_acceptance", 2), ("exact_accepted_raw_error_rate", -1),
            ("block_acceptance_ci95", [0.6, 0.4]),
            ("block_acceptance_ci95", [0.4, float("inf")]),
            ("block_acceptance_ci95", [True, 1]),
            ("block_acceptance_ci95", [0.4]),
        ]
        numeric_fields = ("block_acceptance", "raw_bits_per_transmission",
                          "accepted_raw_error_rate", "accepted_raw_error_standard_error_by_block",
                          "exact_block_acceptance", "exact_accepted_raw_error_rate")
        cases += [(key, value) for key in numeric_fields
                  for value in (None, "0.5", True, float("nan"), float("inf"), 10**400)]
        for key, value in cases:
            data = copy.deepcopy(self.example)
            data["results"][0][key] = value
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                validate_grid(data)
        for key in (*numeric_fields, "accepted_blocks", "block_acceptance_ci95"):
            data = copy.deepcopy(self.example)
            del data["results"][0][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_grid(data)

    def test_invalid_input_does_not_create_or_replace_output(self):
        data = copy.deepcopy(self.example)
        # Exercise a late row: validation must finish before either table is written.
        data["results"][-1]["accepted_raw_error_rate"] = None
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tables"
            with self.assertRaises(ValueError):
                generate_tables(data, output)
            self.assertFalse(output.exists())
            output.mkdir()
            existing = output / "diagnostic-results.tex"
            existing.write_text("previous result", encoding="utf-8")
            with self.assertRaises(ValueError):
                generate_tables(data, output)
            self.assertEqual(existing.read_text(encoding="utf-8"), "previous result")
            self.assertFalse((output / "reset-policy.tex").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
