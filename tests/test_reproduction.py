"""Run-derived reports, provenance, and interrupted-run checks."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qkd_reuse import data, experiments, report


def small_seed_plan():
    """Reduce only test execution size; retain all 20 study seeds."""
    plan = experiments.seed_plan()
    plan["config"]["attempts"] = 100
    return plan


def table_record(**config):
    """Synthetic table-only input for report plumbing, not a simulation result."""
    length, attempts = config["length"], config["attempts"]
    return {
        **config, "attempted_blocks": attempts, "test_fraction": 0.0,
        "link_success": 1.0, "accepted_blocks": 1000,
        "quantum_transmissions": attempts, "reset_attempts": attempts,
        "block_acceptance": 1000 / attempts,
        "raw_bits_per_transmission": 1000 * length / attempts,
        "accepted_raw_error_rate": 0.5,
        "accepted_raw_error_standard_error_by_block": 0.01,
        "block_acceptance_ci95": [0.0094, 0.0107],
        "exact_block_acceptance": 0.48 ** (length - 1),
        "exact_accepted_raw_error_rate": 0.5 - 0.1 * config["strength"]
            * (1 - config["noise"]) ** length / length,
    }


class ReproductionTests(unittest.TestCase):
    def test_generated_seed_records_and_rederived_summary(self):
        plan = small_seed_plan()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(experiments, "seed_plan", return_value=plan), \
                    contextlib.redirect_stdout(io.StringIO()):
                generated = experiments.run_seed_check(directory)
            self.assertTrue(generated["complete"])
            with mock.patch.object(report, "seed_plan", return_value=plan):
                records, summary = report.seed_summary(generated)
                self.assertEqual(len(records), 20)
                self.assertEqual(summary, generated["summary"])
                self.assertEqual(summary["total_attempts"], 2000)
                self.assertEqual(summary["total_accepted"],
                                 sum(record["accepted_blocks"] for record in records))
                bad = copy.deepcopy(generated)
                bad["runs"].pop()
                with self.assertRaises(ValueError):
                    report.seed_summary(bad)
                bad = copy.deepcopy(generated)
                bad["runs"].reverse()
                with self.assertRaises(ValueError):
                    report.seed_summary(bad)
                bad = copy.deepcopy(generated)
                bad["plan"]["config"]["attempts"] = 1
                with self.assertRaises(ValueError):
                    report.seed_summary(bad)

    def test_default_and_explicit_reproduction_preserve_generated_inputs(self):
        plan = small_seed_plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grid_dir, seed_dir = root / "output/grid", root / "output/seed-check"
            with mock.patch.object(experiments.simulation, "experiment", side_effect=table_record), \
                    contextlib.redirect_stdout(io.StringIO()):
                experiments.run_grid(grid_dir)
            with mock.patch.object(experiments, "seed_plan", return_value=plan), \
                    contextlib.redirect_stdout(io.StringIO()):
                seeds = experiments.run_seed_check(seed_dir)
            paths = [grid_dir / "results.json", seed_dir / "results.json"]
            hashes = {path: data.sha256(path) for path in paths}
            with mock.patch.object(report, "ROOT", root), \
                    mock.patch.object(report, "seed_plan", return_value=plan):
                output = report.reproduce(root / "reproduced", include_figure=False)
                explicit = report.reproduce(root / "explicit", *paths, include_figure=False)
                summary = data.read_json(output / "seed-summary.json")
                self.assertEqual(summary, seeds["summary"])
                self.assertEqual(summary, data.read_json(explicit / "seed-summary.json"))
                for name in ("diagnostic-results.tex", "reset-policy.tex"):
                    self.assertTrue((output / "tables" / name).is_file())
                    self.assertEqual((output / "tables" / name).read_bytes(),
                                     (explicit / "tables" / name).read_bytes())
                comparison = data.read_json(output / "retry-comparison.json")
                self.assertEqual(len(comparison), 6)
                fixed = next(row for row in comparison if row["policy"] == "fixed"
                             and row["max_attempts"] == 4)
                self.assertAlmostEqual(fixed["block_acceptance"], 0.6917980815, places=9)
                manifest = data.read_json(output / "manifest.json")
                self.assertTrue(manifest["complete"])
                self.assertEqual(manifest["inputs"], {
                    "grid_sha256": hashes[paths[0]], "seed_check_sha256": hashes[paths[1]],
                })
                with self.assertRaises(ValueError):
                    report.reproduce(output, include_figure=False)
            self.assertEqual(hashes, {path: data.sha256(path) for path in paths})

    def test_missing_experiments_explain_what_to_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(report, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "grid.*seed-check"):
                    report.reproduce(root / "reproduced", include_figure=False)
            self.assertFalse((root / "reproduced").exists())

    def test_protected_or_existing_outputs_are_rejected(self):
        for path in (data.ROOT, data.ROOT / "tests/new-output", data.ROOT / "qkd_reuse/tmp"):
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
        record = experiments.simulation.experiment(
            length=2, strength=0.0, noise=0.0, attempts=10, seed=20261001)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(experiments.simulation, "experiment",
                                   side_effect=[record, RuntimeError("interrupted")]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    experiments.run_grid(directory, attempts=10)
            path = Path(directory) / "results.json"
            generated = data.read_json(path)
            self.assertFalse(generated["complete"])
            self.assertEqual(generated["results"], [record])
            with self.assertRaises(ValueError):
                data.load_results(path)

    def test_current_source_and_completion_required_for_run_records(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                generated = experiments.run_grid(directory, attempts=10)
            path = Path(directory) / "results.json"
            self.assertEqual(data.load_results(path), generated)
            stale = copy.deepcopy(generated)
            stale["provenance"]["source_sha256"]["simulation.py"] = "stale"
            invalid = [stale, {**generated, "complete": False},
                       {**generated, "provenance": {}}]
            missing_completion = copy.deepcopy(generated)
            del missing_completion["complete"]
            invalid.append(missing_completion)
            for record in invalid:
                with self.subTest(complete=record.get("complete")):
                    data.write_json(path, record)
                    with self.assertRaises(ValueError):
                        data.load_results(path)

    def test_failed_seed_accounting_preserves_the_record(self):
        bad_record = {"length": 0}
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(experiments.simulation, "experiment", return_value=bad_record):
                with self.assertRaises((ValueError, KeyError)):
                    experiments.run_seed_check(directory)
            generated = data.read_json(Path(directory) / "results.json")
            self.assertFalse(generated["complete"])
            self.assertEqual(generated["runs"][0]["result"], bad_record)


if __name__ == "__main__":
    unittest.main()
