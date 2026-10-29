"""Measured-occupancy experiments: no synthetic fallback, day-clustered inference.

The radio measurements are real; SU access, sensing corruption and packet outcomes
are explicitly modeled counterfactual replay. No deployed scheduler is claimed.
"""

from __future__ import annotations

import json
import os
import pickle
import resource
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import wilcoxon

from uclada.agents import CORE_METHODS
from uclada.analysis import bootstrap_mean, holm
from uclada.common import (
    EnvConfig,
    MethodConfig,
    digest,
    file_hash,
    machine_metadata,
    source_hash,
    write_json,
)
from uclada.coupled_la import create_agent
from uclada.env import CRNEnvironment, TraceEpisode
from uclada.experiments import candidate_grid, play
from uclada.iq import load_iq_trace


class MeasuredConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "M01-measured-main"
    cohort_manifest: str = "data/manifests/measured_cohort.json"
    methods: list[str] = Field(
        default_factory=lambda: [m for m in CORE_METHODS if m != "u_clada"] + ["coupled_la"]
    )
    seeds: list[int] = Field(default_factory=lambda: list(range(5)))
    validation_days: list[str] = Field(default_factory=lambda: ["Day1", "Day2", "Day3"])
    training_slots: int = Field(2000, ge=100)
    evaluation_slots: int = Field(2000, ge=100)
    workers: int = Field(4, ge=1, le=8)
    bootstrap_draws: int = Field(2000, ge=2000)
    channel_indices: list[int] | None = None
    warm_start: bool = True
    env: EnvConfig = Field(
        default_factory=lambda: EnvConfig(
            agents=3, packet_bits=100, slot_seconds=0.0005, allow_sense_only=True
        )
    )
    method: MethodConfig = Field(default_factory=lambda: MethodConfig(temperature=0, shield=True))


def load_config(path: Path) -> MeasuredConfig:
    return MeasuredConfig.model_validate(yaml.safe_load(path.read_text()))


def cohort(root: Path, config: MeasuredConfig, validation_only: bool = False) -> list[dict[str, Any]]:
    items = json.loads((root / config.cohort_manifest).read_text())["records"]
    result = []
    for item in items:
        if not item.get("primary_inclusion_eligible"):
            continue
        if validation_only and item["day"] not in config.validation_days:
            continue
        if not (root / item["trace_path"]).exists():
            raise FileNotFoundError("An approved measured trace is absent; no synthetic substitution")
        result.append(item)
    if not result:
        raise RuntimeError(
            "No real measured trace passes source and sample-quality eligibility for this cohort"
        )
    return result


def episode(root: Path, item: dict[str, Any], split: str, slots: int) -> tuple[TraceEpisode, int]:
    path = root / item["trace_path"]
    trace = load_iq_trace(path, split)
    if len(trace.occupancy) < slots:
        raise ValueError(
            f"{path.name}/{split}: insufficient measured slots ({len(trace.occupancy)} < {slots})"
        )
    # The first fixed window is prescribed; no high-performance window selection.
    return TraceEpisode(
        trace.occupancy[:slots], trace.trace_id, trace.ground_truth_kind, trace.metadata
    ), int(trace.metadata["split_sample_start_slot"])


def train(
    root: Path, config: MeasuredConfig, item: dict[str, Any], name: str, method: MethodConfig, seed: int
) -> tuple[list[Any], dict[str, Any]]:
    trace, start_index = episode(root, item, "train", config.training_slots)
    trace = select_channels(trace, config.channel_indices)
    if abs(trace.metadata["slot_seconds"] - config.env.slot_seconds) > 1e-12:
        raise ValueError("Packet-slot duration differs from the actual IQ aggregation cadence")
    training_method = method.model_copy(update={"shield": False, "deep_updates": True})
    identity = {
        "source_sha256": source_hash(),
        "trace_sha256": digest(trace.occupancy),
        "record": item,
        "method": name,
        "settings": training_method.model_dump(),
        "seed": seed,
        "env": config.env.model_dump(),
        "training_slots": config.training_slots,
    }
    identifier = digest(identity)[:24]
    folder = root / "results/measured_training" / identifier
    checkpoint = folder / "agents.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as f:
            agents = pickle.load(f)
        return agents, json.loads((folder / "metadata.json").read_text())
    day = int(item["day"].removeprefix("Day"))
    env = CRNEnvironment(trace, config.env, 20261014 + day * 10000 + seed * 10 + 1)
    agents = [
        create_agent(name, trace.occupancy.shape[1], training_method, day * 10000 + seed * 100 + m, True)
        for m in range(config.env.agents)
    ]
    begin = time.perf_counter()
    play(env, agents, diagnostics=False)
    metadata = {
        "training_id": identifier,
        "identity": identity,
        "training_start_index": start_index,
        "training_slots": config.training_slots,
        "seconds": time.perf_counter() - begin,
        "metrics": env.metrics(),
        "risk_training_mode": "unshielded simulated SU training on measured occupancy",
        "information": "Own noisy senses/ACK only; no true failure-cause feedback",
    }
    folder.mkdir(parents=True, exist_ok=True)
    with checkpoint.with_suffix(".partial").open("wb") as f:
        pickle.dump(agents, f, protocol=5)
    os.replace(checkpoint.with_suffix(".partial"), checkpoint)
    for m, a in enumerate(agents):
        if a.deep is not None:
            a.deep.save(str(folder / f"agent_{m}.pt"))
            metadata[f"deep_{m}"] = a.deep.profile()
    write_json(folder / "metadata.json", metadata)
    return agents, metadata


def measured_trial(
    root: Path,
    config: MeasuredConfig,
    item: dict[str, Any],
    name: str,
    method: MethodConfig,
    seed: int,
    split: str,
) -> dict[str, Any]:
    trace, eval_start = episode(root, item, split, config.evaluation_slots)
    trace = select_channels(trace, config.channel_indices)
    identity = {
        "config": config.model_dump(),
        "record": item,
        "method": name,
        "settings": method.model_dump(),
        "seed": seed,
        "split": split,
        "source_sha256": source_hash(),
        "trace_sha256": digest(trace.occupancy),
    }
    run_id = digest(identity)[:24]
    folder = root / "results/measured_runs" / run_id
    summary_path = folder / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    try:
        if config.warm_start:
            agents, training = train(root, config, item, name, method, seed)
        else:
            agents = [
                create_agent(name, trace.occupancy.shape[1], method, seed * 100 + m, True)
                for m in range(config.env.agents)
            ]
            training = {
                "training_start_index": eval_start,
                "training_slots": 0,
                "mode": "cold start",
                "metrics": None,
            }
        elapsed_slots = eval_start - int(training["training_start_index"])
        unobserved_gap = max(0, elapsed_slots - int(training["training_slots"]))
        for m, a in enumerate(agents):
            a.config.shield = method.shield and name not in {"coupled_no_risk", "coupled_no_forecast"}
            a.config.deep_updates = False
            # Real sample-clock gap is included in discounted beliefs, never concatenated away.
            a.begin_evaluation(
                elapsed_slots, int(item["day"].removeprefix("Day")) * 10000 + seed * 100 + m + 500000
            )
            if hasattr(a, "base") and hasattr(a.base, "ts_alpha"):
                a.base.ts_alpha = 1 + a.config.forget**unobserved_gap * (a.base.ts_alpha - 1)
                a.base.ts_beta = 1 + a.config.forget**unobserved_gap * (a.base.ts_beta - 1)
        day = int(item["day"].removeprefix("Day"))
        env = CRNEnvironment(
            trace, config.env, 20261014 + day * 10000 + seed * 10 + (2 if split == "validation" else 3)
        )
        inference = play(env, agents)
        metrics = env.metrics()
        metrics.update(
            inference_us_per_agent_slot=inference * 1e6 / (config.evaluation_slots * config.env.agents),
            inference_us_p50=float(np.quantile(env.decision_times, 0.5) * 1e6),
            inference_us_p95=float(np.quantile(env.decision_times, 0.95) * 1e6),
            worker_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        events = pd.DataFrame(env.rows)
        chosen = events[events["forecast_busy"].notna()]
        metrics["forecast_brier"] = (
            float(np.mean((chosen["forecast_busy"].astype(float) - chosen["hidden_busy"].astype(float)) ** 2))
            if len(chosen)
            else None
        )
        events.to_parquet(folder / "events.parquet", index=False, compression="zstd")
        final = []
        for a in agents:
            controller = a.base if hasattr(a, "base") else a
            final.append(
                {
                    "channel_or_primitive_p": controller.p.tolist(),
                    "access_p": a.access.tolist() if hasattr(a, "access") else None,
                }
            )
        write_json(folder / "final_probabilities.json", final)
        result = {
            "run_id": run_id,
            "identity": identity,
            "method": name,
            "experiment": config.name,
            "dataset": "iraq_rf",
            "day": item["day"],
            "cluster_id": item["day"],
            "seed": seed,
            "split": split,
            "ground_truth_kind": "threshold_measurement_proxy",
            "actual_measured_occupancy": True,
            "su_interactions": "simulated counterfactual packet access",
            "privileged_execution_information": False,
            "trace_sha256": digest(trace.occupancy),
            "sensing_stream_sha256": digest(env.sensing_uniforms),
            "arrival_stream_sha256": digest(env.arrivals),
            "source_prefix_sha256": trace.metadata["source_prefix_sha256"],
            "unobserved_gap_slots_before_evaluation": unobserved_gap,
            "training": training,
            "metrics": metrics,
            "runtime_seconds": time.perf_counter() - begin,
            "status": "COMPLETE",
            "machine": machine_metadata(),
        }
        write_json(folder / "config.json", identity)
        write_json(summary_path, result)
        return result
    except Exception:
        write_json(
            folder / "failure.json",
            {"identity": identity, "status": "FAILED", "exception": traceback.format_exc()},
        )
        raise


def _job(args: tuple) -> dict[str, Any]:
    root, config, item, name, method, seed, split = args
    return measured_trial(
        Path(root),
        MeasuredConfig.model_validate(config),
        item,
        name,
        MethodConfig.model_validate(method),
        seed,
        split,
    )


def run(
    root: Path,
    config: MeasuredConfig,
    split: str,
    settings: dict[str, dict] | None = None,
    validation_only: bool = False,
) -> list[dict[str, Any]]:
    settings = settings or {}
    if split == "test":
        check_measured_lock(root, config)
    items = cohort(root, config, validation_only)
    jobs = [
        (
            str(root.resolve()),
            config.model_dump(),
            item,
            name,
            settings.get(name, config.method.model_dump()),
            seed,
            split,
        )
        for name in config.methods
        for item in items
        for seed in config.seeds
    ]
    outputs = []
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        futures = [pool.submit(_job, j) for j in jobs]
        for future in as_completed(futures):
            outputs.append(future.result())
            if len(outputs) % max(1, len(jobs) // 10) == 0:
                print(config.name, len(outputs), "/", len(jobs), flush=True)
    write_json(
        root / f"reports/experiments/{config.name}_{split}.json",
        {
            "expected_runs": len(jobs),
            "completed_runs": len(outputs),
            "real_day_clusters": len({r["day"] for r in outputs}),
            "policy_seeds_per_day": len(config.seeds),
            "run_ids": [r["run_id"] for r in outputs],
            "config": config.model_dump(),
        },
    )
    return outputs


def tune_measured(root: Path, config: MeasuredConfig) -> dict[str, dict]:
    path = root / "results/measured_settings.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["source_sha256"] == source_hash():
            return previous["settings"]
        if (root / "results/measured_test_lock.json").exists():
            raise RuntimeError("Measured test is locked for a different source")
    config = config.model_copy(deep=True)
    config.method.uncertainty_c = calibrate_measured(root, config)
    all_rows = []
    settings: dict[str, dict] = {}
    for name in config.methods:
        if name == "coupled_la":
            candidates = [
                config.method.model_copy(
                    update={"eta_reward": a, "access_eta_reward": b, "availability_credit": credit}
                )
                for a, b, credit in [(0.05, 0.03, 0.05), (0.1, 0.05, 0.1), (0.2, 0.1, 0.2)]
            ]
        else:
            candidates = candidate_grid(name, config.method)
        scored: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            trial_config = config.model_copy(
                update={"name": f"M00-validation-{name}-{index}", "methods": [name]}
            )
            outputs = run(
                root, trial_config, "validation", {name: candidate.model_dump()}, validation_only=True
            )
            attempts = sum(x["metrics"]["n_attempted"] for x in outputs)
            overlap = sum(x["metrics"]["n_pu_overlap"] for x in outputs)
            risk = overlap / attempts if attempts else None
            row = {
                "method": name,
                "candidate": index,
                "mean_goodput_bps": float(np.mean([x["metrics"]["goodput_bps"] for x in outputs])),
                "overlap_per_attempt": risk,
                "feasible": risk is not None and risk <= 0.1,
                "settings": candidate.model_dump(),
                "run_ids": [x["run_id"] for x in outputs],
            }
            scored.append(row)
            all_rows.append(row)
        feasible = [r for r in scored if r["feasible"]]
        chosen = (
            max(feasible, key=lambda r: r["mean_goodput_bps"])
            if feasible
            else min(
                scored,
                key=lambda r: (
                    r["overlap_per_attempt"] if r["overlap_per_attempt"] is not None else 2,
                    -r["mean_goodput_bps"],
                ),
            )
        )
        settings[name] = chosen["settings"]
        write_json(root / "results/measured_tuning_trials.json", all_rows)
        print("VALIDATION SELECTED", name, chosen["candidate"], "feasible", chosen["feasible"], flush=True)
    write_json(
        path,
        {
            "settings": settings,
            "source_sha256": source_hash(),
            "validation_days": config.validation_days,
            "seeds": config.seeds,
            "test_outcomes_used": False,
            "candidate_budget": 3,
        },
    )
    return settings


def lock_measured(root: Path, config: MeasuredConfig) -> dict[str, Any]:
    settings = root / "results/measured_settings.json"
    if not settings.exists():
        raise RuntimeError("Measured validation tuning required before test lock")
    data = json.loads((root / config.cohort_manifest).read_text())
    if not data.get("acquisition_complete"):
        raise RuntimeError("Complete the fixed source cohort audit before opening any test outcomes")
    if json.loads(settings.read_text())["source_sha256"] != source_hash():
        raise RuntimeError("Validation settings are stale for this source")
    identity = {
        "source_sha256": source_hash(),
        "config": config.model_dump(),
        "settings_sha256": file_hash(settings),
        "cohort_sha256": file_hash(root / config.cohort_manifest),
        "primary_endpoint": "goodput_bps",
        "constraint": "occupied measurement proxy overlaps / attempts <=0.10",
        "unit_of_inference": "recording day; average policy seeds within each day",
        "data_kind": "real measured IQ subband occupancy proxies; simulated SU responses",
    }
    path = root / "results/measured_test_lock.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["identity"] != identity:
            raise RuntimeError("Measured test lock mismatch; use a new preregistered untouched release")
        return old
    lock = {
        "identity": identity,
        "locked_at": machine_metadata()["utc_timestamp"],
        "lock_sha256": digest(identity),
    }
    write_json(path, lock)
    return lock


def analyze_measured(root: Path) -> dict[str, Any]:
    rows = []
    manifest = []
    day_dates = {
        x["day"]: x["recording_date"]
        for x in json.loads((root / "data/manifests/measured_cohort.json").read_text())["records"]
    }
    for p in sorted((root / "results/measured_runs").glob("*/summary.json")):
        r = json.loads(p.read_text())
        rows.append(
            {
                "run_id": r["run_id"],
                "experiment": r["experiment"],
                "method": r["method"],
                "day": r["day"],
                "seed": r["seed"],
                "split": r["split"],
                **{k: v for k, v in r["metrics"].items() if not isinstance(v, list)},
            }
        )
        manifest.append(
            {
                "run_id": r["run_id"],
                "summary": str(p.relative_to(root)),
                "summary_sha256": file_hash(p),
                "events_sha256": file_hash(p.parent / "events.parquet"),
                "trace_sha256": r["trace_sha256"],
                "source_prefix_sha256": r["source_prefix_sha256"],
            }
        )
    f = pd.DataFrame(rows)
    if f.empty:
        raise RuntimeError("No completed measured trials")
    f.to_parquet(root / "results/measured_all_runs.parquet", index=False)
    f.to_csv(root / "results/measured_all_runs.csv", index=False)
    metrics = [
        "goodput_bps",
        "normalized_goodput",
        "pu_overlap_per_attempt",
        "su_contention_per_attempt",
        "switch_rate",
        "jain_fairness",
        "abstention_rate",
        "access_defer_rate",
        "sensing_defer_rate",
        "transmission_duty",
        "forecast_brier",
        "inference_us_per_agent_slot",
        "queue_backlog",
        "drops",
        "delay_p95_s",
    ]
    summaries = []
    contrasts = []
    heldout = f[f["split"] == "test"]
    for experiment, g in heldout.groupby("experiment"):
        # Repeated stochastic seeds on one real recording are not extra RF environments.
        days = g.groupby(["day", "method"], as_index=False)[metrics].mean()
        for method, group in days.groupby("method"):
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                low, high = bootstrap_mean(values, draws=2000)
                summaries.append(
                    {
                        "experiment": experiment,
                        "method": method,
                        "metric": metric,
                        "mean": float(finite.mean()) if len(finite) else None,
                        "sd_between_days": float(finite.std(ddof=1)) if len(finite) > 1 else None,
                        "ci_low": low,
                        "ci_high": high,
                        "n_rf_day_clusters": len(group),
                        "n_valid_day_clusters": len(finite),
                        "policy_seeds_per_day": int(g["seed"].nunique()),
                        "bootstrap_draws": 2000,
                    }
                )
        pivot = days.pivot(index="day", columns="method", values="goodput_bps")
        family = []
        if "coupled_la" in pivot:
            for method in pivot.columns:
                if method == "coupled_la":
                    continue
                pair = pivot[["coupled_la", method]].dropna()
                delta = (pair["coupled_la"] - pair[method]).to_numpy()
                low, high = bootstrap_mean(delta)
                ordered = pair.assign(date=[day_dates[d] for d in pair.index]).sort_values("date")
                block_low, block_high = day_block_bootstrap(
                    (ordered["coupled_la"] - ordered[method]).to_numpy()
                )
                pvalue = float(wilcoxon(delta).pvalue) if len(delta) > 1 and np.any(delta) else 1.0
                family.append(
                    {
                        "experiment": experiment,
                        "baseline": method,
                        "delta_goodput_bps": float(delta.mean()),
                        "calendar_block3_ci_low": block_low,
                        "calendar_block3_ci_high": block_high,
                        "ci_low": low,
                        "ci_high": high,
                        "p_value": pvalue,
                        "n_rf_day_clusters": len(delta),
                        "paired_effect": float(delta.mean() / delta.std(ddof=1))
                        if len(delta) > 1 and delta.std(ddof=1) > 0
                        else None,
                    }
                )
            adjusted = holm(np.array([r["p_value"] for r in family]))
            for r, pvalue in zip(family, adjusted, strict=True):
                r["p_holm"] = float(pvalue)
            contrasts.extend(family)
    for name, data in [
        ("measured_aggregate", pd.DataFrame(summaries)),
        ("measured_paired_contrasts", pd.DataFrame(contrasts)),
    ]:
        data.to_csv(root / f"results/{name}.csv", index=False)
        data.to_parquet(root / f"results/{name}.parquet", index=False)
    write_json(root / "results/measured_results_manifest.json", {"runs": manifest, "count": len(manifest)})
    return {
        "completed_runs": len(f),
        "test_runs": len(heldout),
        "real_day_clusters": int(heldout["day"].nunique()),
        "classification": "Measurement-driven trace replay; counterfactual SU interactions remain simulated",
    }


def calibrate_measured(root: Path, config: MeasuredConfig) -> float:
    """Offline proxy-label bin-mean coverage on real validation only; no safety theorem."""
    target = root / "results/measured_uncertainty_calibration.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if previous["source_sha256"] == source_hash():
            return float(previous["selected_c"])
    rows = []
    method = config.method.model_copy(update={"shield": False})
    for item in cohort(root, config, True):
        for seed in config.seeds:
            agents, training = train(root, config, item, "coupled_la", method, seed)
            trace, start = episode(root, item, "validation", config.evaluation_slots // 2)
            for m, a in enumerate(agents):
                a.begin_evaluation(start - int(training["training_start_index"]), 700000 + seed * 100 + m)
            env = CRNEnvironment(trace, config.env, 800000 + int(item["day"][3:]) * 100 + seed)
            play(env, agents)
            f = pd.DataFrame(env.rows)
            rows.append(f[f.forecast_busy.notna()][["forecast_busy", "forecast_uncertainty", "hidden_busy"]])
    f = pd.concat(rows, ignore_index=True).astype(float)
    f["bin"] = pd.cut(f.forecast_busy, bins=np.linspace(0, 1, 11).tolist(), include_lowest=True)
    candidates = []
    for c in [0.0, 0.5, 1.0, 1.645, 2.0, 3.0, 4.0]:
        f["upper"] = np.minimum(1, f.forecast_busy + c * f.forecast_uncertainty)
        b = f.groupby("bin", observed=True).agg(
            upper=("upper", "mean"), observed=("hidden_busy", "mean"), n=("hidden_busy", "size")
        )
        b = b[b.n >= 30]
        candidates.append(
            {
                "c": c,
                "eligible_bins": len(b),
                "bin_mean_coverage": float((b.upper >= b.observed).mean()) if len(b) else 0.0,
            }
        )
    eligible = [r for r in candidates if r["bin_mean_coverage"] >= 0.95]
    selected = float(eligible[0]["c"]) if eligible else 4.0
    f.drop(columns="bin").to_parquet(root / "results/measured_uncertainty_calibration.parquet", index=False)
    write_json(
        target,
        {
            "source_sha256": source_hash(),
            "selected_c": selected,
            "candidates": candidates,
            "days": config.validation_days,
            "seeds": config.seeds,
            "test_outcomes_used": False,
            "supervision": "Central offline threshold-proxy labels from validation only",
            "qualification": "Empirical bin-mean coverage, neither pointwise uncertainty coverage nor a physical interference guarantee",
        },
    )
    return selected


def check_measured_lock(root: Path, config: MeasuredConfig) -> None:
    path = root / "results/measured_test_lock.json"
    if not path.exists():
        raise RuntimeError("Freeze measured validation settings, source and cohort before test")
    lock = json.loads(path.read_text())["identity"]
    if lock["source_sha256"] != source_hash():
        raise RuntimeError("Source differs from the measured test lock")
    if lock["settings_sha256"] != file_hash(root / "results/measured_settings.json"):
        raise RuntimeError("Measured settings differ from the test lock")
    if lock["cohort_sha256"] != file_hash(root / "data/manifests/measured_cohort.json"):
        raise RuntimeError("Primary measured cohort changed after test lock")
    manifest = json.loads((root / config.cohort_manifest).read_text())
    if not manifest.get("acquisition_complete"):
        raise RuntimeError("Requested measured cohort acquisition is incomplete")
    for item in cohort(root, config):
        if file_hash(root / item["trace_path"]) != item["trace_file_sha256"]:
            raise RuntimeError("A measured occupancy file has changed")


def day_block_bootstrap(
    deltas: np.ndarray, draws: int = 2000, length: int = 3
) -> tuple[float | None, float | None]:
    """Circular blocks in actual acquisition-date order; dependence sensitivity only."""
    if not len(deltas):
        return None, None
    rng = np.random.default_rng(20261014)
    starts = rng.integers(len(deltas), size=(draws, int(np.ceil(len(deltas) / length))))
    indices = (starts[:, :, None] + np.arange(length)) % len(deltas)
    means = deltas[indices.reshape(draws, -1)[:, : len(deltas)]].mean(axis=1)
    interval = np.quantile(means, [0.025, 0.975])
    return float(interval[0]), float(interval[1])


def select_channels(trace: TraceEpisode, indices: list[int] | None) -> TraceEpisode:
    if indices is None:
        return trace
    if (
        not indices
        or len(set(indices)) != len(indices)
        or min(indices) < 0
        or max(indices) >= trace.occupancy.shape[1]
    ):
        raise ValueError("Only distinct available measured resources can be selected")
    return TraceEpisode(
        trace.occupancy[:, indices],
        trace.trace_id + "-resources-" + "-".join(map(str, indices)),
        trace.ground_truth_kind,
        {**trace.metadata, "selected_resource_indices": indices},
    )


def verify_measured(root: Path) -> dict[str, Any]:
    """Validate immutable provenance and replay event accounting, without retuning."""
    manifest = json.loads((root / "results/measured_results_manifest.json").read_text())
    checked = 0
    for run in manifest["runs"]:
        summary_path = root / run["summary"]
        if file_hash(summary_path) != run["summary_sha256"]:
            raise AssertionError("Summary checksum mismatch")
        event_path = summary_path.parent / "events.parquet"
        if file_hash(event_path) != run["events_sha256"]:
            raise AssertionError("Event checksum mismatch")
        r = json.loads(summary_path.read_text())
        events = pd.read_parquet(event_path)
        m = r["metrics"]
        assert int(events.attempted.sum()) == m["n_attempted"]
        assert m["n_attempted"] == m["n_success"] + m["n_pu_overlap"] + m["n_su_contention"]
        assert not (events.attempted & events.sensed_busy.fillna(False).astype(bool)).any()
        assert not (events.access_defer & events.attempted).any()
        assert r["actual_measured_occupancy"] and not r["privileged_execution_information"]
        assert m["pu_overlap_per_attempt"] is None if not m["n_attempted"] else True
        checked += 1
    result = {
        "status": "PASS",
        "verified_runs": checked,
        "checks": [
            "summary/events SHA256",
            "attempt outcome identity",
            "sensing/intent ordering",
            "explicit measured provenance",
        ],
        "qualification": "Byte and accounting checks do not establish physical occupant labels or algorithm superiority",
    }
    write_json(root / "reports/measured_verification.json", result)
    return result
