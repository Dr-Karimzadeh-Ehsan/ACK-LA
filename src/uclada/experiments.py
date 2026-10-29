"""Immutable trial artifacts, train/validation tuning, frozen settings and experiment grids."""

from __future__ import annotations

import copy
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

from uclada.agents import ABLATIONS, LocalAgent
from uclada.common import (
    ExperimentConfig,
    MethodConfig,
    digest,
    file_hash,
    machine_metadata,
    source_hash,
    write_json,
)
from uclada.coupled_la import create_agent
from uclada.data import load_audited_trace
from uclada.env import CRNEnvironment, TraceEpisode, synthetic_trace


def make_trace(config: ExperimentConfig, seed: int, split: str, slots: int | None = None) -> TraceEpisode:
    if config.dataset == "nist":
        assert config.trace_path is not None
        return load_audited_trace(config.trace_path, split)
    scenario = "toy_legacy" if config.dataset == "toy_legacy" else config.scenario
    namespace = {"train": 10000, "validation": 20000, "test": 30000, "smoke": 40000, "synthetic": 50000}[
        split
    ]
    return synthetic_trace(
        slots or config.slots, config.channels, seed + config.root_seed + namespace, scenario
    )


def play(env: CRNEnvironment, agents: list[LocalAgent], diagnostics: bool = True) -> float:
    elapsed = 0.0
    decision_times = []
    for _ in range(env.slots):
        start = time.perf_counter()
        actions = [agent.act(env.observation(m)) for m, agent in enumerate(agents)]
        duration = time.perf_counter() - start
        elapsed += duration
        decision_times.append(duration / len(agents))
        info = [a.diagnostics() for a in agents] if diagnostics else None
        intents = (
            [getattr(agent, "transmit_intent", True) for agent in agents]
            if env.config.allow_sense_only
            else None
        )
        transition = env.step(actions, intents)
        if info is not None:
            for m in range(env.config.agents):
                # Evaluator augments private logs; never passes private records back to a policy.
                env.rows[-env.config.agents + m].update(info[m])
        for agent, feedback in zip(agents, transition.agent_visible, strict=True):
            agent.update(feedback)
    env.decision_times = decision_times
    return elapsed


def train_agents(
    root: Path, config: ExperimentConfig, name: str, method: MethodConfig, train_seed: int
) -> tuple[list[LocalAgent], dict[str, Any]]:
    train_method = method.model_copy(update={"shield": False, "deep_updates": True})
    identity = {
        "name": name,
        "method": train_method.model_dump(),
        "env": config.env.model_dump(),
        "channels": config.channels,
        "train_slots": config.train_slots,
        "seed": train_seed,
        "dataset": config.dataset,
        "scenario": config.scenario,
        "source": source_hash(),
        "trace_path": config.trace_path,
        "root_seed": config.root_seed,
    }
    cache_id = digest(identity)[:20]
    folder = root / "results/training" / cache_id
    checkpoint = folder / "agents.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as file:
            agents = pickle.load(file)
        return agents, json.loads((folder / "metadata.json").read_text())
    trace = make_trace(config, train_seed, "train", config.train_slots)
    env = CRNEnvironment(trace, config.env, train_seed + 171)
    agents = [
        create_agent(
            name, trace.occupancy.shape[1], train_method, train_seed * 100 + m, config.env.allow_sense_only
        )
        for m in range(config.env.agents)
    ]
    start = time.perf_counter()
    inference = play(env, agents, diagnostics=False)
    metadata = {
        "training_id": cache_id,
        "training_config": identity,
        "training_seconds": time.perf_counter() - start,
        "training_inference_seconds": inference,
        "slots": env.slots,
        "training_shield": False,
        "training_supervision": "local noisy sensing and own ACK only",
        "training_risk_warning": "Unshielded simulated training incurs separately reported overlap",
        "metrics": env.metrics(),
        "trace_sha256": digest(trace.occupancy),
    }
    folder.mkdir(parents=True, exist_ok=True)
    with checkpoint.with_suffix(".partial").open("wb") as file:
        pickle.dump(agents, file, protocol=5)
    os.replace(checkpoint.with_suffix(".partial"), checkpoint)
    for m, a in enumerate(agents):
        if a.deep is not None:
            a.deep.save(str(folder / f"agent_{m}.pt"))
            metadata[f"deep_agent_{m}"] = a.deep.profile()
    write_json(folder / "metadata.json", metadata)
    return agents, metadata


def trial(
    root: Path,
    config: ExperimentConfig,
    name: str,
    seed: int,
    split: str,
    method: MethodConfig | None = None,
    warm_start: bool = True,
) -> dict[str, Any]:
    method = method or config.method
    trace = make_trace(config, seed, split)
    train_seed = config.train_seeds[seed % len(config.train_seeds)]
    identity = {
        "config": config.model_dump(),
        "name": name,
        "method": method.model_dump(),
        "seed": seed,
        "split": split,
        "trace_sha256": digest(trace.occupancy),
        "source_sha256": source_hash(),
        "warm_start": warm_start,
        "train_seed": train_seed,
    }
    run_id = digest(identity)[:24]
    folder = root / "results/runs" / run_id
    summary_file = folder / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        if summary["identity_sha256"] != digest(identity):
            raise RuntimeError("Immutable run identity mismatch")
        return summary
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        training: dict[str, Any] = {}
        if warm_start:
            agents, training = train_agents(root, config, name, method, train_seed)
            for m, a in enumerate(agents):
                # Retain training state, restore evaluated control restrictions.
                a.config.shield = method.shield
                if name in {"no_shield", "adaptive_la", "la_only"}:
                    a.config.shield = False
                a.config.deep_updates = method.deep_updates
                a.begin_evaluation(int(training["slots"]), config.root_seed + seed * 1000 + m)
        else:
            agents = [
                LocalAgent(name, config.channels, method, config.root_seed + seed * 1000 + m)
                for m in range(config.env.agents)
            ]
        env = CRNEnvironment(trace, config.env, config.root_seed + seed)
        trace_folder = root / "results/traces"
        trace_folder.mkdir(parents=True, exist_ok=True)
        trace_path = trace_folder / (str(identity["trace_sha256"]) + ".npz")
        if not trace_path.exists():
            temporary = trace_folder / (run_id + ".npz")
            np.savez_compressed(temporary, occupancy=trace.occupancy)
            os.replace(temporary, trace_path)
        inference_seconds = play(env, agents)
        metrics = env.metrics()
        events = pd.DataFrame(env.rows)
        if config.event_logs:
            events.to_parquet(folder / "events.parquet", index=False, compression="zstd")
        forecast = events[events["forecast_busy"].notna()]
        metrics["forecast_brier"] = (
            float(np.mean((forecast["forecast_busy"] - forecast["hidden_busy"].astype(float)) ** 2))
            if len(forecast)
            else None
        )
        metrics["eligible_fraction"] = float(events["eligible_fraction"].mean())
        metrics["drift_alarms"] = int(events["drift_alarm"].sum())
        metrics["inference_seconds"] = inference_seconds
        metrics["inference_us_per_agent_slot"] = inference_seconds * 1e6 / (env.slots * config.env.agents)
        metrics["inference_us_per_agent_slot_p50"] = float(np.quantile(env.decision_times, 0.5) * 1e6)
        metrics["inference_us_per_agent_slot_p95"] = float(np.quantile(env.decision_times, 0.95) * 1e6)
        metrics["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        summary = {
            "run_id": run_id,
            "identity_sha256": digest(identity),
            "identity": identity,
            "method": name,
            "experiment": config.name,
            "scenario": config.scenario,
            "dataset": config.dataset,
            "split_id": split,
            "seed": seed,
            "train_seed": train_seed,
            "cluster_id": f"{config.dataset}-{config.scenario}-{seed}"
            if config.dataset != "nist"
            else trace.trace_id,
            "independent_unit": "synthetic_scenario_seed"
            if config.dataset != "nist"
            else "source_site_session",
            "occupancy_ground_truth_kind": trace.ground_truth_kind,
            "trace_metadata": trace.metadata,
            "exogenous_trace_path": str(trace_path),
            "sensing_stream_sha256": digest(env.sensing_uniforms),
            "arrival_stream_sha256": digest(env.arrivals),
            "privileged_information_used": False,
            "training": training,
            "risk_budget_definition": "forecast busy upper <= rho; empirical overlap conditional on attempts separately evaluated",
            "runtime_seconds": time.perf_counter() - start,
            "status": "COMPLETE",
            "metrics": metrics,
            "metadata": machine_metadata(),
            "output_paths": {
                "summary": str(summary_file),
                "events": str(folder / "events.parquet") if config.event_logs else None,
            },
        }
        write_json(folder / "config.json", identity)
        write_json(summary_file, summary)
        return summary
    except Exception:
        write_json(
            folder / "failure.json",
            {"identity": identity, "status": "FAILED", "exception_stack": traceback.format_exc()},
        )
        raise


def _trial_job(args: tuple[str, dict, str, int, str, dict, bool]) -> dict[str, Any]:
    root, config, name, seed, split, method, warm = args
    return trial(
        Path(root),
        ExperimentConfig.model_validate(config),
        name,
        seed,
        split,
        MethodConfig.model_validate(method),
        warm,
    )


def run_jobs(
    root: Path,
    config: ExperimentConfig,
    split: str,
    settings: dict[str, dict] | None = None,
    warm_start: bool = True,
) -> list[dict[str, Any]]:
    settings = settings or {}
    jobs = [
        (
            str(root.resolve()),
            config.model_dump(),
            name,
            seed,
            split,
            settings.get(name, config.method.model_dump()),
            warm_start,
        )
        for name in config.methods
        for seed in config.seeds
    ]
    results = []
    workers = min(config.workers, len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_names = {pool.submit(_trial_job, job): (job[2], job[3]) for job in jobs}
        for future in as_completed(future_names):
            result = future.result()
            results.append(result)
            if len(results) % max(1, len(jobs) // 10) == 0:
                print(f"{config.name}: {len(results)}/{len(jobs)} completed", flush=True)
    record_experiment(root, config, split, results)
    return results


def candidate_grid(name: str, base: MethodConfig) -> list[MethodConfig]:
    """Three predeclared candidates per method, including modern comparators."""
    changes: dict[str, list[dict[str, Any]]] = {
        "random": [{}, {}, {}],
        "greedy": [{"forget": v} for v in [0.95, 0.98, 0.995]],
        "epsilon_greedy": [{"epsilon": v} for v in [0.05, 0.1, 0.2]],
        "epsilon_decay": [{"epsilon": v} for v in [0.1, 0.2, 0.4]],
        "lri": [{"eta_reward": v} for v in [0.03, 0.1, 0.2]],
        "lr_ep": [
            {"eta_reward": a, "eta_penalty": b} for a, b in [(0.05, 0.005), (0.1, 0.01), (0.15, 0.002)]
        ],
        "q_learning": [{"q_alpha": a, "q_gamma": g} for a, g in [(0.05, 0.5), (0.1, 0.9), (0.2, 0.95)]],
        "sw_ucb": [{"window": w} for w in [50, 100, 300]],
        "discounted_ts": [{"forget": v} for v in [0.95, 0.98, 0.995]],
        "exp3": [{"exp3_gamma": v} for v in [0.03, 0.1, 0.2]],
        "dqn": [{"deep_lr": lr, "deep_hidden": h} for lr, h in [(0.0003, 32), (0.001, 32), (0.001, 64)]],
        "context_only": [{"temperature": t} for t in [0.0, 0.5, 2.0]],
        "u_clada": [{"eta_reward": a, "temperature": t} for a, t in [(0.05, 0.0), (0.1, 0.5), (0.2, 2.0)]],
    }
    return [base.model_copy(update=x) for x in changes.get(name, [{}, {}, {}])]


def calibrate_uncertainty(root: Path, config: ExperimentConfig) -> float:
    """Centralized validation-only bin-mean calibration, explicitly declared.

    Calibration supervision uses simulator states/proxy labels only outside execution.
    Coverage is empirical bin-mean coverage, not pointwise binary-label coverage.
    """
    records = []
    method = config.method.model_copy(update={"shield": False})
    for seed in config.train_seeds:
        trace = make_trace(config, seed + 700, "validation", config.slots)
        # Earlier half exclusively for calibration; tuning uses separate validation seeds.
        half = TraceEpisode(
            trace.occupancy[: config.slots // 2],
            trace.trace_id + "-calibration",
            trace.ground_truth_kind,
            trace.metadata,
        )
        env = CRNEnvironment(half, config.env, seed + 777)
        agents = [
            LocalAgent("u_clada", config.channels, method, seed * 10 + m) for m in range(config.env.agents)
        ]
        play(env, agents)
        f = pd.DataFrame(env.rows)
        records.append(
            f[f["forecast_busy"].notna()][["forecast_busy", "forecast_uncertainty", "hidden_busy"]]
        )
    f = pd.concat(records, ignore_index=True)
    f["bin"] = pd.cut(f["forecast_busy"], bins=np.linspace(0, 1, 11).tolist(), include_lowest=True)
    candidates = []
    for c in [0.0, 0.5, 1.0, 1.645, 2.0, 3.0, 4.0]:
        f["upper"] = np.minimum(1, f["forecast_busy"] + c * f["forecast_uncertainty"])
        bins = f.groupby("bin", observed=True).agg(
            q=("forecast_busy", "mean"),
            upper=("upper", "mean"),
            observed=("hidden_busy", "mean"),
            n=("hidden_busy", "size"),
        )
        bins = bins[bins["n"] >= 30]
        coverage = float((bins["upper"] >= bins["observed"]).mean()) if len(bins) else 0.0
        candidates.append({"c": c, "bin_mean_coverage": coverage, "eligible_bins": len(bins)})
    feasible = [r for r in candidates if r["bin_mean_coverage"] >= 0.95]
    selected = feasible[0]["c"] if feasible else candidates[-1]["c"]
    f.drop(columns="bin").to_parquet(root / "results/uncertainty_calibration.parquet", index=False)
    write_json(
        root / "results/uncertainty_calibration.json",
        {
            "selected_c": selected,
            "candidates": candidates,
            "target": 0.95,
            "definition": "fraction of validation forecast bins with empirical mean busy <= mean upper",
            "supervision": "centralized synthetic simulator labels; no execution-time hidden states",
            "guarantee": False,
            "seed_namespace": "validation seeds train_seed+700",
            "rows": len(f),
        },
    )
    return float(selected)


def tune(root: Path, config: ExperimentConfig) -> dict[str, dict]:
    path = root / "results/selected_settings.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["source_sha256"] == source_hash() and existing["config_sha256"] == digest(config):
            return existing["settings"]
        if (root / "results/test_lock.json").exists():
            raise RuntimeError("A test lock exists for different source/config; use a new release directory")
    root.joinpath("results").mkdir(exist_ok=True)
    c = calibrate_uncertainty(root, config)
    base = config.method.model_copy(update={"uncertainty_c": c})
    rows: list[dict[str, Any]] = []
    settings: dict[str, dict] = {}
    for name in config.methods:
        candidates = candidate_grid(name, base)
        scored: list[dict[str, Any]] = []
        for number, candidate in enumerate(candidates):
            validation = config.model_copy(
                update={"name": f"validation-{name}-c{number}", "methods": [name], "seeds": list(range(5))}
            )
            runs = run_jobs(root, validation, "validation", {name: candidate.model_dump()})
            goodput = float(np.mean([r["metrics"]["goodput_bps"] for r in runs]))
            num = sum(r["metrics"]["n_pu_overlap"] for r in runs)
            den = sum(r["metrics"]["n_attempted"] for r in runs)
            risk = num / den if den else None
            feasible = risk is not None and risk <= 0.1
            row = {
                "method": name,
                "candidate": number,
                "goodput_bps": goodput,
                "overlap_per_attempt": risk,
                "feasible": feasible,
                "config": candidate.model_dump(),
                "run_ids": [r["run_id"] for r in runs],
            }
            scored.append(row)
            rows.append(row)
        eligible = [row for row in scored if row["feasible"]]
        chosen = (
            max(eligible, key=lambda x: x["goodput_bps"])
            if eligible
            else min(
                scored,
                key=lambda x: (
                    x["overlap_per_attempt"] if x["overlap_per_attempt"] is not None else 2,
                    -x["goodput_bps"],
                ),
            )
        )
        settings[name] = chosen["config"]
        print(f"Selected {name}: candidate {chosen['candidate']}, feasible={chosen['feasible']}", flush=True)
        write_json(root / "results/tuning_trials.json", rows)
    write_json(
        path,
        {
            "source_sha256": source_hash(),
            "config_sha256": digest(config),
            "settings": settings,
            "selection": "validation feasible goodput; otherwise minimum risk",
            "validation_seeds": list(range(5)),
            "train_seeds": config.train_seeds,
        },
    )
    return settings


def lock_test(root: Path, config: ExperimentConfig) -> dict[str, Any]:
    selected = root / "results/selected_settings.json"
    if not selected.exists():
        raise RuntimeError("Run validation tuning before locking test")
    identity = {
        "source_sha256": source_hash(),
        "settings_sha256": file_hash(selected),
        "experiment_config_sha256": digest(config),
        "test_seeds": config.seeds,
        "train_seeds": config.train_seeds,
        "primary_endpoint": "goodput_bps",
        "primary_risk": "pu_overlap_per_attempt <= 0.10",
        "dataset": config.dataset,
        "real_primary_status": "BLOCKED" if config.dataset == "synthetic" else "PENDING",
        "online_adaptation": True,
        "deep_online_updates": config.method.deep_updates,
    }
    path = root / "results/test_lock.json"
    if path.exists():
        lock = json.loads(path.read_text())
        if lock["identity"] != identity:
            raise RuntimeError(
                "Test lock mismatch: do not retune after test; create a distinct preregistered release"
            )
        return lock
    lock = {
        "identity": identity,
        "locked_at": machine_metadata()["utc_timestamp"],
        "lock_sha256": digest(identity),
    }
    write_json(path, lock)
    return lock


def record_experiment(root: Path, config: ExperimentConfig, split: str, runs: list[dict]) -> None:
    row = {
        "id": config.name,
        "question": config.name,
        "dataset": config.dataset,
        "scenario": config.scenario,
        "methods": ";".join(config.methods),
        "N": config.channels,
        "M": config.env.agents,
        "T": config.slots,
        "sensor": f"PFA={config.env.p_false_alarm};PMD={config.env.p_missed_detection}",
        "traffic": config.env.traffic,
        "shift": config.scenario,
        "split": split,
        "seeds": ";".join(map(str, config.seeds)),
        "repetitions": len(runs),
        "metrics": "goodput;overlap;contention;switching;fairness;queue;compute",
        "status": "EXECUTED",
        "artifact_paths": ";".join(r["run_id"] for r in runs),
        "limitations": "Uncalibrated synthetic dynamics; no measured RF environment"
        if config.dataset != "nist"
        else "Measurement proxy replay with simulated SU interactions",
    }
    path = root / "experiment_registry.csv"
    if path.exists():
        old = pd.read_csv(path)
        old = old[~((old["id"] == config.name) & (old["split"] == split))]
    else:
        old = pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def supplementary_configs(base: ExperimentConfig) -> list[ExperimentConfig]:
    """Predeclared broad CPU grid; short-horizon scale/sensor screens are labeled."""
    configs = []
    methods = ["lr_ep", "sw_ucb", "u_clada", "no_shield"]

    def add(name: str, **changes: Any) -> None:
        updates = {"name": name, "methods": methods, "slots": 600, "train_slots": 600, **changes}
        configs.append(base.model_copy(update=updates))

    add(
        "E01-toy_legacy",
        dataset="toy_legacy",
        scenario="toy_legacy",
        slots=600,
        methods=["random", "lri", "lr_ep", "epsilon_greedy"],
        method=base.method.model_copy(update={"eta_reward": 0.15, "eta_penalty": 0.002}),
    )
    for scenario in [
        "stationary",
        "abrupt",
        "gradual",
        "periodic",
        "mixed",
        "adversarial",
        "correlated",
        "bursty",
    ]:
        add(f"E03-drift-{scenario}", scenario=scenario, slots=2000, train_slots=1000)
    for channels in [3, 5, 10, 20, 40]:
        for agents in [2, 3, 5, 10, 20, 40]:
            add(
                f"E04-scale-N{channels}-M{agents}",
                channels=channels,
                env=base.env.model_copy(update={"agents": agents}),
                slots=300,
                train_slots=300,
                methods=["lr_ep", "sw_ucb", "u_clada"],
            )
    for pfa in [0, 0.05, 0.1, 0.2]:
        for pmd in [0, 0.05, 0.1, 0.2]:
            add(
                f"E05-sensing-fa{pfa}-md{pmd}",
                env=base.env.model_copy(update={"p_false_alarm": pfa, "p_missed_detection": pmd}),
                method=base.method.model_copy(update={"sensor_pfa": pfa, "sensor_pmd": pmd}),
            )
    for rho in [0.05, 0.1, 0.2, 0.4, 0.6, 1.0]:
        add(
            f"E06-risk-{rho}",
            method=base.method.model_copy(update={"rho": rho, "shared_shield": True}),
            slots=2000,
            methods=["u_clada", "context_only", "lr_ep", "sw_ucb"],
        )
    add("E07-ablation", methods=["u_clada", "context_only"] + ABLATIONS, slots=2000, train_slots=1000)
    for scenario in ["iid", "correlated", "bursty"]:
        add(f"E08-ood-{scenario}", scenario=scenario, slots=2000)
    for pmd_assumed in [0, 0.05, 0.2]:
        add(
            f"E08-misspecified-sensor-{pmd_assumed}",
            env=base.env.model_copy(update={"p_missed_detection": 0.2}),
            method=base.method.model_copy(update={"sensor_pmd": pmd_assumed}),
        )
    for traffic in ["bernoulli", "bursty"]:
        for load in [0.05, 0.2, 0.5, 0.8, 1.0]:
            add(
                f"E09-qos-{traffic}-{load}",
                slots=2000,
                env=base.env.model_copy(update={"traffic": traffic, "arrival_rate": load}),
                methods=["lr_ep", "sw_ucb", "u_clada", "no_queue_reward"],
            )
    for cost in [0, 0.05, 0.2, 0.5]:
        add(f"E10-switch-{cost}", env=base.env.model_copy(update={"switch_cost": cost}), slots=2000)
    for scenario in ["always_idle", "always_busy"]:
        add(f"E13-control-{scenario}", scenario=scenario, methods=["random", "lr_ep", "u_clada", "no_shield"])
    for a in [0.03, 0.1, 0.2]:
        for tau in [0, 0.5, 2.0]:
            add(
                f"E15-sensitivity-a{a}-tau{tau}",
                methods=["u_clada", "no_shield"],
                method=base.method.model_copy(update={"eta_reward": a, "temperature": tau}),
            )
    return configs


def run_supplementary(root: Path, base: ExperimentConfig, settings: dict[str, dict]) -> None:
    for config in supplementary_configs(base):
        # Method/environment changes are prescribed interventions, not test-selected tuning.
        merged = {}
        for name in config.methods:
            selected = copy.deepcopy(settings.get(name, settings.get("u_clada", base.method.model_dump())))
            for key, value in config.method.model_dump().items():
                if value != base.method.model_dump()[key]:
                    selected[key] = value
            merged[name] = selected
        print(f"Running {config.name}: {len(config.methods) * len(config.seeds)} trials", flush=True)
        run_jobs(root, config, "synthetic", merged)
