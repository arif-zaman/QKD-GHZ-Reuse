"""End-to-end data handling, provenance, and interrupted-run checks."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qkd_reuse import data, experiments, report


class ReproductionTests(unittest.TestCase):
    def test_reference_records_and_rederived_seed_summary(self):
        data.verify_reference()
        saved = data.read_json(data.REFERENCE / "seed-check.json")
        records, summary = report.seed_summary(saved)
        self.assertEqual(len(records), 20)
        self.assertEqual(summary, saved["summary"])
        bad = copy.deepcopy(saved)
        bad["runs"].pop()
        with self.assertRaises(ValueError):
            report.seed_summary(bad)

    def test_default_reproduction_preserves_inputs(self):
        hashes = {path: data.sha256(path) for path in data.REFERENCE.glob("*.json")}
        with tempfile.TemporaryDirectory() as directory:
            output = report.reproduce(directory, include_figure=False)
            summary = data.read_json(output / "seed-summary.json")
            self.assertEqual(summary["total_accepted"], 11558)
            self.assertEqual(summary["total_attempts"], 2000000)
            self.assertTrue((output / "tables/diagnostic-results.tex").is_file())
            comparison = data.read_json(output / "retry-comparison.json")
            self.assertEqual(len(comparison), 6)
            fixed = next(row for row in comparison if row["policy"] == "fixed"
                         and row["max_attempts"] == 4)
            self.assertAlmostEqual(fixed["block_acceptance"], 0.6917980815, places=9)
            self.assertTrue(data.read_json(output / "manifest.json")["complete"])
            with self.assertRaises(ValueError):
                report.reproduce(directory, include_figure=False)
        self.assertEqual(hashes, {path: data.sha256(path) for path in hashes})

    def test_protected_or_existing_outputs_are_rejected(self):
        for path in (data.ROOT, data.REFERENCE / "new-output", data.ROOT / "qkd_reuse/tmp"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                data.new_output(path)
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "keep.txt"
            marker.write_text("existing user data", encoding="utf-8")
            with mock.patch.object(experiments.simulation, "experiment") as simulate:
                with self.assertRaises(ValueError):
                    experiments.run_grid(directory, attempts=10)
                simulate.assert_not_called()
            self.assertEqual(marker.read_text(), "existing user data")

    def test_partial_grid_is_retained_but_cannot_generate_tables(self):
        record = data.read_json(data.REFERENCE / "grid.json")["results"][0]
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(experiments.simulation, "experiment",
                                   side_effect=[record, RuntimeError("interrupted")]):
                with self.assertRaises(RuntimeError):
                    experiments.run_grid(directory)
            path = Path(directory) / "results.json"
            saved = data.read_json(path)
            self.assertFalse(saved["complete"])
            self.assertEqual(saved["results"], [record])
            with self.assertRaises(ValueError):
                data.load_results(path, "grid.json")

    def test_current_source_required_for_fresh_records(self):
        record = data.read_json(data.REFERENCE / "grid.json")
        record.update(complete=True, provenance=data.provenance())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            data.write_json(path, record)
            self.assertEqual(data.load_results(path, "grid.json"), record)
            record["provenance"]["source_sha256"]["simulation.py"] = "stale"
            data.write_json(path, record)
            with self.assertRaises(ValueError):
                data.load_results(path, "grid.json")

    def test_failed_seed_accounting_preserves_the_record(self):
        bad_record = {"length": 0}
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(experiments.simulation, "experiment", return_value=bad_record):
                with self.assertRaises((ValueError, KeyError)):
                    experiments.run_seed_check(directory)
            saved = data.read_json(Path(directory) / "results.json")
            self.assertFalse(saved["complete"])
            self.assertEqual(saved["runs"][0]["result"], bad_record)


if __name__ == "__main__":
    unittest.main()
