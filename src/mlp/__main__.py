"""CLI entrypoint. Data training and serving arrive in subsequent milestones."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mlp.common.config import AppSettings
from mlp.common.environment import environment_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlp", description="MLOps reference platform")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Print the current runtime environment")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    commands.add_parser("config", help="Validate and print non-sensitive configuration")
    build = commands.add_parser("data-build", help="Generate and validate synthetic splits")
    build.add_argument("--output-dir", type=Path, default=Path("data") / "synthetic")
    build.add_argument("--rows", type=int)
    build.add_argument("--seed", type=int)
    verify = commands.add_parser("data-verify", help="Verify hashes and schema in data directory")
    verify.add_argument("--data-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "doctor":
        report = environment_report()
        if arguments.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            for name, value in report.items():
                print(f"{name}: {value if value is not None else 'unknown'}")
        return 0

    settings = AppSettings()
    if arguments.command == "data-build":
        from mlp.data.pipeline import build_data

        manifest = build_data(
            arguments.output_dir,
            rows=arguments.rows if arguments.rows is not None else settings.tiny_rows,
            seed=arguments.seed if arguments.seed is not None else settings.random_seed,
            validation_fraction=settings.validation_fraction,
            test_fraction=settings.test_fraction,
        )
        print(json.dumps({"data_version": manifest["data_version"], "rows": manifest["rows"]}))
        return 0
    if arguments.command == "data-verify":
        from mlp.data.manifest import verify_manifest

        manifest = verify_manifest(arguments.data_dir)
        print(json.dumps({"verified": True, "data_version": manifest["data_version"]}))
        return 0
    print(json.dumps(settings.public_config(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
