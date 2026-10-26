# GHZ reuse: measurement, reset, and resource analysis
The model studies repeated readout and reset of a shared
Greenberger--Horne--Zeilinger (GHZ) state after Alice transmits one qubit to Bob.
It records which blocks survive, what Bob can infer from his outcomes, and
the operations spent on accepted and rejected blocks. The code reproduces the
paper's simulation tables, analytical figure, retry-policy comparison, and
independent seed check. It analyzes protocol requirements; it does not implement
a complete secure QKD protocol.

## Setup

Run all commands from the directory containing this README:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The recorded experiments used Python 3.12.14, NumPy 2.3.5, and NumPy's PCG64
random-number generator. ReportLab 4.4.9 draws the figure. Simulation and tests
require NumPy and the Python standard library.

## Reproduce the paper outputs

```sh
python -m qkd_reuse reproduce
```

This verifies the included reference-file hashes and regenerates the following
under `output/reproduced/`, **without running new simulations**:

| Output | Manuscript content |
|---|---|
| `tables/diagnostic-results.tex` | Table II: exact and sampled single-attempt results |
| `tables/reset-policy.tex` | Table III: sampled reset-policy comparison |
| `readout-reset-tradeoff.pdf`, `tradeoff.csv` | Figure 1: analytical readout/reset tradeoff |
| `retry-comparison.csv`, `.json` | Section V-C: exact fixed versus amplitude-updated retries |
| `seed-summary.json`, `per-seed.csv` | Section V-B: all 20 prespecified validation seeds |
| `README.md`, `manifest.json` | Summary and input/code provenance |

The saved seed check gives 11,558 accepted blocks out of 2,000,000 attempts:
pooled acceptance 0.005779, with a 95% Wilson interval of approximately
[0.005674894, 0.005885005]. These values appear in the generated summary.

Table I lists symbolic Bell-state measurement (BSM) branches, which the circuit
tests verify. Generated LaTeX tables contain the numerical cells, captions,
and notes; float placement and spacing are left to the paper's LaTeX source.
The plot embeds TeX Times fonts when available and otherwise uses standard
PDF Times-Roman. Font availability can change appearance and file hashes.
Use `--skip-figure` to generate everything except the PDF without ReportLab.

## Run the simulations again

```sh
# Small execution check; not manuscript data.
python -m qkd_reuse grid --attempts 1000 --output output/smoke

# All 27 configurations, 100,000 independent blocks each.
python -m qkd_reuse grid

# All 20 prespecified seeds, 100,000 independent blocks each.
python -m qkd_reuse seed-check

# Regenerate outputs using those new experiment records.
python -m qkd_reuse reproduce --grid output/grid/results.json \
    --seed-check output/seed-check/results.json --output output/fresh
```

Each command requires a new or empty output directory. Use `--output` to keep
multiple runs. Completed records remain on disk if execution stops; incomplete
runs cannot generate manuscript tables. A grid with fewer than 100,000 attempts
per setting is for testing and is rejected by the paper-output command.

The grid uses seeds 20261001–20261027 in the recorded configuration order. The
seed check uses 2026101001–2026101020 and pools only those 20 additional runs.
Do not split batches or replace seeds when comparing with the reference data:
random draws depend on batch size and operation order. Dependency versions and
floating-point differences between platforms can also affect sampled results.

## Check the implementation

```sh
python -m unittest discover -s tests -v
```

Tests build three- and four-qubit circuits independently of the compressed
simulator. They check BSM corrections, coherent and mixed reset branches,
sequential conditioning, both retry strategies, the erasure criterion's
assumptions, statistical accounting, and boundary cases. Numerical checks
support the stated model; they do not constitute a security proof.

## Find the relevant code

| File | Purpose |
|---|---|
| `qkd_reuse/simulation.py` | Sampled density-matrix evolution and resource counters |
| `qkd_reuse/analysis.py` | Exact fixed and amplitude-updated retry predictions |
| `qkd_reuse/experiments.py` | Grid settings, seeds, and experiment execution |
| `qkd_reuse/statistics.py` | Wilson intervals, seed pooling, and counter checks |
| `qkd_reuse/tables.py`, `figure.py` | Manuscript tables and analytical figure |
| `qkd_reuse/report.py`, `data.py`, `__main__.py` | Reproduction workflow, provenance, and CLI |
| `tests/` | Independent physical and reproduction checks |
| `reference/` | Original records, seed plan, and integrity manifest |

To follow the implementation, read [the model](docs/model.md), then
`simulation.experiment()` for the full sequence on a batch of independent
blocks. Its helpers implement the BSM, noise, readout, and reset steps in order.
`experiments.py` applies this model to the paper's settings; `analysis.py`
evaluates the exact retry formulas.

The Monte Carlo model fixes population contrast at **delta=0.2** and uses
fixed-filter retries. The exact analysis supports **0 <= delta <= 1** and both
retry policies. Amplitude-updated predictions are analytical, not sampled rows
in Table III.

For example, the same exact comparison can be called from Python:

```python
from qkd_reuse.analysis import amplitude_updated_predictions, fixed_filter_predictions

settings = dict(length=5, delta=0.2, strength=0.5, noise=0.0, max_attempts=4)
print(fixed_filter_predictions(**settings))
print(amplitude_updated_predictions(**settings))
```

## Saved data

`reference/grid.json` contains all 27 experiment records.
`reference/seed-plan.json` specifies the additional 20-seed study, and
`reference/seed-check.json` contains its run records and pooled summary.
`reference/manifest.json` supplies the file hashes used to check these inputs.
New runs record their own environment and source hashes.

The Table II row at L=8, p=0 retains its 644 accepted blocks and pointwise
confidence interval that excludes theory. The 20-seed check provides additional
evidence without replacing that sample. The small L=12 accepted samples are
also retained.
