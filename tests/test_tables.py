"""Check published table values and reject incomplete or malformed grid data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from qkd_reuse.tables import generate_tables, validate_manuscript_grid


REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "grid.json"
SAMPLED_ROWS = [
    r'2 & 0.000 & 48,053 & 0.480000 & [0.47743, 0.48363] & 0.96106 & 47.50\% & 47.51\% & 0.16 \\',
    r'2 & 0.005 & 48,301 & 0.480000 & [0.47991, 0.48611] & 0.96602 & 47.52\% & 47.52\% & 0.16 \\',
    r'5 & 0.000 & 5,422 & 0.053084 & [0.05283, 0.05564] & 0.27110 & 49.00\% & 49.11\% & 0.30 \\',
    r'5 & 0.005 & 5,288 & 0.053084 & [0.05151, 0.05428] & 0.26440 & 49.02\% & 48.67\% & 0.31 \\',
    r'8 & 0.000 & 644 & 0.005871 & [0.00596, 0.00696] & 0.05152 & 49.38\% & 50.08\% & 0.69 \\',
    r'8 & 0.005 & 599 & 0.005871 & [0.00553, 0.00649] & 0.04792 & 49.40\% & 48.94\% & 0.72 \\',
    r'12 & 0.000 & 29 & 0.000312 & [0.00020, 0.00042] & 0.00348 & 49.58\% & 49.43\% & 2.58 \\',
    r'12 & 0.005 & 37 & 0.000312 & [0.00027, 0.00051] & 0.00444 & 49.61\% & 54.73\% & 2.16 \\',
]
RETRY_ROWS = [
    r'Coherent & 1 & 0.0542 & 1.829 & 49.11\% & 0.30 \\',
    r'Mixed & 1 & 0.0529 & 1.822 & 48.53\% & 0.31 \\',
    r'Coherent & 2 & 0.2702 & 3.974 & 47.73\% & 0.14 \\',
    r'Coherent & 4 & 0.6935 & 6.854 & 46.13\% & 0.08 \\',
]
SAMPLED_NOTE = (
    r'$\delta=0.2$, $g=0.5$; $10^5$ independent block attempts per row. '
    r'$P_{\rm acc}$ and $Q_{\rm raw}$ are exact analytical values; '
    r'$\widehat\eta_{\rm raw}$ and $\widehat Q_{\rm raw}$ are Monte Carlo estimates. '
    r'$\eta_{\rm raw}$ counts retained labels per transmitted qubit. '
    r'CI: pointwise 95\% Wilson confidence interval for acceptance. '
    r'Raw-error standard errors are computed across independent accepted blocks '
    r'and reported in percentage points (pp).'
)
RETRY_NOTE = (
    r'$\delta=0.2$, $L=5$, $g=0.5$, $p=0$; $10^5$ block attempts per row. '
    r'Coherent rows use the fixed filter. $R$: maximum reset attempts per stage; '
    r'$\overline n_F$: sample mean reset attempts per transmitted qubit. '
    r'Hats denote Monte Carlo estimates. SE uses independent accepted blocks; '
    r'pp: percentage points.'
)


class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(REFERENCE.read_text(encoding="utf-8"))

    def test_saved_grid_reproduces_published_values_captions_and_notes(self):
        data = copy.deepcopy(self.reference)
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_tables(data, directory)
            self.assertEqual({path.name for path in paths},
                             {"diagnostic-results.tex", "reset-policy.tex"})
            self.assertEqual(set(Path(directory).iterdir()), set(paths))
            sampled, retry = [path.read_text(encoding="utf-8") for path in paths]
        self.assertEqual(data, self.reference)
        sampled_values = sampled.split('\\midrule\n')[1].split('\\bottomrule')[0].splitlines()
        retry_values = retry.split('\\midrule\n')[1].split('\\bottomrule')[0].splitlines()
        self.assertEqual(sampled_values, SAMPLED_ROWS)
        self.assertEqual(retry_values, RETRY_ROWS)
        self.assertIn(r'\caption{Exact and sampled results for single-attempt coherent reset.}', sampled)
        self.assertIn(r'\caption{Sampled performance under alternative reset policies.}', retry)
        self.assertIn(SAMPLED_NOTE, sampled)
        self.assertIn(RETRY_NOTE, retry)
        # A pointwise interval can miss the true probability. Preserve that row.
        row = next(row for row in data["results"] if
                   (row["length"], row["strength"], row["noise"], row["max_reset_attempts"])
                   == (8, 0.5, 0.0, 1))
        self.assertLess(row["exact_block_acceptance"], row["block_acceptance_ci95"][0])

    def test_shuffled_grid_has_the_same_manuscript_order(self):
        data = copy.deepcopy(self.reference)
        data["results"].reverse()
        self.assertEqual(validate_manuscript_grid(data), self.reference["results"])
        with tempfile.TemporaryDirectory() as directory:
            original = generate_tables(self.reference, Path(directory) / "original")
            shuffled = generate_tables(data, Path(directory) / "shuffled")
            self.assertEqual([p.read_bytes() for p in original],
                             [p.read_bytes() for p in shuffled])

    def test_rejects_unsupported_metadata_and_missing_duplicate_configurations(self):
        invalid = [None, [], {}, {"schema": True}, {"schema": 2}]
        for key, value in (("complete", False), ("complete", "true"),
                           ("independent_attempts_per_configuration", 100),
                           ("independent_attempts_per_configuration", 100000.0),
                           ("results", []), ("results", None)):
            data = copy.deepcopy(self.reference)
            data[key] = value
            invalid.append(data)
        data = copy.deepcopy(self.reference)
        data["results"][1] = copy.deepcopy(data["results"][0])
        invalid.append(data)
        data = copy.deepcopy(self.reference)
        data["results"][0] = None
        invalid.append(data)
        for index, data in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_manuscript_grid(data)

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
            data = copy.deepcopy(self.reference)
            data["results"][0][key] = value
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                validate_manuscript_grid(data)
        for key in (*numeric_fields, "accepted_blocks", "block_acceptance_ci95"):
            data = copy.deepcopy(self.reference)
            del data["results"][0][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_manuscript_grid(data)

    def test_invalid_input_does_not_create_or_replace_output(self):
        data = copy.deepcopy(self.reference)
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
