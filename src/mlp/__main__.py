"""CLI entrypoint. Data training and serving arrive in subsequent milestones."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from mlp.common.config import AppSettings
from mlp.common.environment import environment_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlp", description="MLOps reference platform")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Print the current runtime environment")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    commands.add_parser("config", help="Validate and print non-sensitive configuration")
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
    print(json.dumps(settings.public_config(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
