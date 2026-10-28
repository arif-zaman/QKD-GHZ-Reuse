# GHZ reuse: measurement, reset, and resource analysis

Monte Carlo simulation and exact analysis of repeated readout and reset of a
shared Greenberger–Horne–Zeilinger (GHZ) state. The model tracks accepted and
rejected blocks, readout errors, and resource costs. It does not implement a
complete secure quantum key distribution protocol.

## Setup

Run from the repository root using Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```sh
python -m qkd_reuse grid
python -m qkd_reuse seed-check
python -m qkd_reuse reproduce
```

The grid runs 27 configurations; the seed check runs 20 additional seeds.
Each configuration or seed uses 100,000 independent blocks. Settings and seeds
are defined in [experiments.py](qkd_reuse/experiments.py).

Results are written to `output/grid/` and `output/seed-check/`. The final command
reads those results and writes LaTeX tables, the analytical PDF figure, and
CSV/JSON summaries to `output/reproduced/`.

Each command requires a new or empty output directory. Use `--output` for another
location and `reproduce --grid PATH --seed-check PATH` for custom input files.
Run `python -m qkd_reuse reproduce --help` for options; `--skip-figure` omits the
PDF. Incomplete runs and grids with fewer than 100,000 attempts per setting
cannot generate reports.

Keep the configured seeds, batch sizes, and operation order for reproducible
sampling. Dependency versions and floating-point differences across platforms
can affect sampled results.

## Tests

```sh
python -m unittest discover -s tests -v
```
