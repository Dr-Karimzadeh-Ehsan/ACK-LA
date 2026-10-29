"""Configuration, stable hashes and atomic evidence persistence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psutil
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agents: int = Field(3, ge=1)
    p_false_alarm: float = Field(0.05, ge=0, le=1)
    p_missed_detection: float = Field(0.05, ge=0, le=1)
    packet_bits: int = Field(12000, gt=0)
    slot_seconds: float = Field(0.01, gt=0)
    switch_cost: float = Field(0.05, ge=0, le=1)
    sensing_cost: float = Field(0.01, ge=0, le=1)
    idle_cost: float = Field(0.02, ge=0, le=1)
    queue_weight: float = Field(0.01, ge=0)
    traffic: Literal["saturated", "bernoulli", "bursty"] = "saturated"
    arrival_rate: float = Field(0.2, ge=0, le=1)
    queue_capacity: int = Field(100, ge=1)
    deadline_slots: int = Field(100, ge=1)
    sensing_mode: Literal["select_then_sense"] = "select_then_sense"
    tier: Literal["A"] = "A"
    allow_sense_only: bool = False


class MethodConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    eta_reward: float = Field(0.1, ge=0, le=1)
    eta_penalty: float = Field(0.01, ge=0, le=1)
    forget: float = Field(0.98, gt=0, le=1)
    epsilon: float = Field(0.1, ge=0, le=1)
    window: int = Field(100, ge=2)
    temperature: float = Field(0.5, ge=0)
    rho: float = Field(0.1, ge=0, le=1)
    uncertainty_c: float = Field(1.0, ge=0)
    uncertainty_weight: float = Field(0.5, ge=0)
    drift_threshold: float = Field(8.0, gt=0)
    drift_tolerance: float = Field(0.02, ge=0)
    shield: bool = True
    shared_shield: bool = False
    adaptive: bool = True
    drift: bool = True
    context: bool = True
    uncertainty: bool = True
    sensor_correction: bool = True
    sensor_pfa: float = Field(0.05, ge=0, le=1)
    sensor_pmd: float = Field(0.05, ge=0, le=1)
    predictor: Literal["pseudo", "ewma", "persistence", "markov"] = "pseudo"
    q_alpha: float = Field(0.1, gt=0, le=1)
    q_gamma: float = Field(0.9, ge=0, le=1)
    exp3_gamma: float = Field(0.1, gt=0, le=1)
    deep_lr: float = Field(0.001, gt=0)
    deep_hidden: int = Field(32, ge=4)
    deep_batch: int = Field(32, ge=2)
    deep_history: int = Field(4, ge=1)
    deep_train_interval: int = Field(8, ge=1)
    deep_updates: bool = False
    la_mode: Literal["normal", "frozen", "shuffled"] = "normal"
    shield_coldstart: Literal["strict"] = "strict"
    availability_credit: float = Field(0.1, ge=0, le=1)
    access_eta_reward: float = Field(0.05, ge=0, le=1)
    access_eta_penalty: float = Field(0.02, ge=0, le=1)
    reset_mix: float = Field(0.1, ge=0, le=1)
    freeze_channel_la: bool = False
    freeze_access_la: bool = False
    shuffle_channel_la: bool = False
    # V3: local evidence drives LA updates; no privileged collision labels.
    evidence_window: int = Field(32, ge=2)
    evidence_max_age: int = Field(128, ge=1)
    evidence_delta: float = Field(0.1, gt=0, lt=1)
    evidence_horizon: int = Field(16000, ge=100)
    evidence_penalty: float = Field(0.5, ge=0, le=1)
    evidence_enabled: bool = True
    primitive_epsilon_min: float = Field(0.001, gt=0, le=1)
    primitive_epsilon_max: float = Field(0.03, gt=0, le=1)
    adaptive_exploration: bool = True
    pressure_rate: float = Field(0.02, gt=0, le=1)
    access_feedback: Literal["utility", "success_margin"] = "success_margin"
    access_target: float = Field(0.36787944117144233, gt=0, lt=1)
    ack_forget: float = Field(0.995, gt=0, le=1)
    ack_prior_strength: float = Field(2.0, gt=0)
    access_initial_send: float = Field(0.8, gt=0, lt=1)
    deep_eval_epsilon: float | None = Field(None, ge=0, le=1)
    baseline_la_floor: float = Field(0.01, ge=0, le=1)
    defer_ack_value: bool = True
    defer_channel_penalty: float = Field(0.1, ge=0, le=1)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "smoke"
    dataset: Literal["synthetic", "toy_legacy", "nist"] = "synthetic"
    scenario: str = "abrupt"
    channels: int = Field(5, ge=1)
    slots: int = Field(2000, ge=10)
    train_slots: int = Field(1000, ge=10)
    seeds: list[int] = Field(default_factory=lambda: list(range(20)))
    train_seeds: list[int] = Field(default_factory=lambda: [101, 102, 103, 104, 105])
    methods: list[str] = Field(default_factory=lambda: ["random", "lr_ep", "sw_ucb", "u_clada"])
    env: EnvConfig = Field(default_factory=lambda: EnvConfig())
    method: MethodConfig = Field(default_factory=lambda: MethodConfig())
    workers: int = Field(2, ge=1, le=8)
    bootstrap_iterations: int = Field(2000, ge=100)
    root_seed: int = 20261014
    trace_path: str | None = None
    event_logs: bool = True

    @model_validator(mode="after")
    def real_data_path(self) -> ExperimentConfig:
        if self.dataset == "nist" and not self.trace_path:
            raise ValueError("NIST suite requires an audited trace_path; no synthetic fallback")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Repeated seeds are not independent clusters")
        return self


def read_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    raise TypeError(type(obj).__name__)


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=jsonable, allow_nan=False)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, default=jsonable, allow_nan=False) + "\n")
    os.replace(tmp, path)


def source_hash() -> str:
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))})


def machine_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ["numpy", "scipy", "pandas", "pyarrow", "matplotlib", "pydantic", "torch", "uclada"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]))
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    return {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine_cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "library_versions": packages,
        "git_commit": commit,
        "dirty_tree_flag": dirty,
        "source_sha256": source_hash(),
        "gpu_model": None,
        "device": "cpu",
    }
