"""Dedicated no-fallback CLI for LA-centered real-IQ experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uclada.common import write_json
from uclada.coupled_la import COUPLED_ABLATIONS
from uclada.measured import analyze_measured, load_config, lock_measured, run, tune_measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "lock", "test", "ablations", "analyze", "pilot"])
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/measured.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_config(root / args.config)
    if args.command == "validate":
        tune_measured(root, config)
    elif args.command == "lock":
        print(json.dumps(lock_measured(root, config), indent=2))
    elif args.command == "analyze":
        print(json.dumps(analyze_measured(root), indent=2))
    elif args.command == "pilot":
        config = config.model_copy(
            update={
                "name": "M00-real-validation-pilot",
                "methods": ["coupled_la", "lr_ep", "sw_ucb", "dqn"],
                "seeds": [0],
            }
        )
        outputs = run(root, config, "validation", validation_only=True)
        write_json(root / "reports/measured_validation_pilot.json", outputs)
    else:
        settings = json.loads((root / "results/measured_settings.json").read_text())["settings"]
        if args.command == "ablations":
            config = config.model_copy(
                update={"name": "M02-measured-LA-ablations", "methods": ["coupled_la"] + COUPLED_ABLATIONS}
            )
            settings = {name: settings["coupled_la"] for name in config.methods}
        run(root, config, "test", settings)


if __name__ == "__main__":
    main()
