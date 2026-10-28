"""Command-line entry point: python -m qkd_reuse --help."""
import argparse
from pathlib import Path

from .data import ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m qkd_reuse",
        description="Run and analyze the GHZ-reuse measurement and reset model.")
    commands = parser.add_subparsers(dest="command", required=True)
    reproduce = commands.add_parser("reproduce", help="Generate outputs after running grid and seed-check")
    reproduce.add_argument("--output", type=Path, default=ROOT / "output/reproduced")
    reproduce.add_argument("--grid", type=Path,
                           help="Completed grid results (default: output/grid/results.json)")
    reproduce.add_argument("--seed-check", type=Path,
                           help="Completed seed results (default: output/seed-check/results.json)")
    reproduce.add_argument("--skip-figure", action="store_true",
                           help="Write plot data without the ReportLab PDF")
    grid = commands.add_parser("grid", help="Run the 27-configuration Monte Carlo grid")
    grid.add_argument("--attempts", type=int, default=100000,
                      help="Independent blocks per setting (default: 100000)")
    grid.add_argument("--output", type=Path, default=ROOT / "output/grid")
    seeds = commands.add_parser("seed-check", help="Run all 20 configured validation seeds")
    seeds.add_argument("--output", type=Path, default=ROOT / "output/seed-check")
    args = parser.parse_args(argv)
    try:
        if args.command == "reproduce":
            from .report import reproduce as run
            output = run(args.output, args.grid, args.seed_check, not args.skip_figure)
        elif args.command == "grid":
            from .experiments import run_grid
            run_grid(args.output, args.attempts)
            output = args.output.resolve()
        else:
            from .experiments import run_seed_check
            run_seed_check(args.output)
            output = args.output.resolve()
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.error(str(error))
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
