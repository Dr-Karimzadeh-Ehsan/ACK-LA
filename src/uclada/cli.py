"""Headless research command-line entry points with truthful failure codes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from uclada.common import read_config, source_hash, write_json


def correctness(root: Path) -> None:
    path = root / "reports/correctness"
    path.mkdir(parents=True, exist_ok=True)
    code = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--junitxml", str(path / "pytest.xml")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    (path / "pytest.txt").write_text(code.stdout + code.stderr)
    print(code.stdout)
    if code.returncode:
        raise RuntimeError("Correctness tests failed; performance experiments are prohibited")
    write_json(path / "passed.json", {"source_sha256": source_hash(), "exit_code": 0})


def require_correctness(root: Path) -> None:
    path = root / "reports/correctness/passed.json"
    if not path.exists() or json.loads(path.read_text())["source_sha256"] != source_hash():
        correctness(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auditable U-CLADA research pipeline")
    parser.add_argument(
        "command",
        choices=[
            "data",
            "test",
            "tune",
            "lock",
            "experiment",
            "supplementary",
            "suite",
            "analyze",
            "figures",
            "tables",
            "manuscript",
            "verify",
            "references",
            "report",
        ],
    )
    parser.add_argument("action", nargs="?", default="")
    parser.add_argument("--config", default="configs/experiments/main.yaml")
    parser.add_argument("--dataset", choices=["nist", "electrosense"], default="nist")
    parser.add_argument("--subset", default="pilot")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-root", default="results/")
    parser.add_argument("--from-registry", default="results/")
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--input")
    args = parser.parse_args(argv)
    root = Path.cwd()
    output: Any = None
    try:
        if args.command == "data":
            from uclada.data import audit_electrosense, audit_nist, download_nist, prepare_canonical

            if args.action == "audit":
                output = audit_nist(root) if args.dataset == "nist" else audit_electrosense(root)
            elif args.action == "download":
                output = (
                    download_nist(root) if args.dataset == "nist" else audit_electrosense(root, download=True)
                )
                if output.get("success") is False:
                    print(json.dumps(output, indent=2))
                    return 2
            elif args.action == "prepare":
                if not args.input:
                    raise RuntimeError(
                        "--input must name a verified timestamped dBm CSV; NIST raw access/alignment must first be audited"
                    )
                output = str(prepare_canonical(Path(args.input), root))
            else:
                raise ValueError("data action must be audit, download or prepare")
        elif args.command == "test":
            correctness(root)
        elif args.command == "references":
            from uclada.literature import verify_references

            output = verify_references(root)
        elif args.command in {"tune", "lock", "experiment", "supplementary", "suite"}:
            from uclada.experiments import lock_test, run_jobs, run_supplementary, tune

            config = read_config(args.config)
            if args.workers:
                config.workers = args.workers
            if args.pilot:
                config = config.model_copy(
                    update={"seeds": [0, 1, 2], "slots": 2000, "name": config.name + "-pilot"}
                )
            require_correctness(root)
            if args.command == "tune":
                output = tune(root, config)
            elif args.command == "lock":
                output = lock_test(root, config)
            elif args.command == "experiment":
                settings = {}
                if args.split == "test":
                    lock_test(root, config)
                    settings = json.loads((root / "results/selected_settings.json").read_text())["settings"]
                output = {
                    "runs": len(
                        run_jobs(root, config, args.split, settings, warm_start=args.split != "smoke")
                    )
                }
            elif args.command == "supplementary":
                settings = json.loads((root / "results/selected_settings.json").read_text())["settings"]
                run_supplementary(root, config, settings)
            else:
                settings = tune(root, config)
                lock_test(root, config)
                run_jobs(root, config, "test", settings)
                run_supplementary(root, config, settings)
        elif args.command == "analyze":
            from uclada.analysis import analyze

            output = analyze(root)
        elif args.command == "figures":
            from uclada.visualization import make_figures

            output = make_figures(root)
        elif args.command == "tables":
            from uclada.visualization import make_tables

            output = make_tables(root)
        elif args.command == "manuscript":
            from uclada.reporting import manuscript

            output = manuscript(root)
        elif args.command == "verify":
            from uclada.reporting import verify

            output = verify(root)
            if output["status"] != "PASS":
                print(json.dumps(output, indent=2))
                return 1
        elif args.command == "report":
            from uclada.reporting import completion_report

            output = completion_report(root)
        if output is not None:
            print(json.dumps(output, indent=2, default=str))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
