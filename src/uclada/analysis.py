"""Cluster-aware estimates, paired inference, calibration and artifact aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

from uclada.common import file_hash, write_json

METRICS = [
    "goodput_bps",
    "normalized_goodput",
    "pu_overlap_per_attempt",
    "su_contention_per_attempt",
    "switch_rate",
    "jain_fairness",
    "abstention_rate",
    "sensing_defer_rate",
    "opportunity_capture",
    "delay_mean_s",
    "delay_p95_s",
    "queue_backlog",
    "drops",
    "forecast_brier",
    "eligible_fraction",
    "inference_us_per_agent_slot",
    "peak_process_rss_bytes",
    "drift_alarms",
]


def bootstrap_mean(
    values: np.ndarray, seed: int = 42, draws: int = 2000
) -> tuple[float | None, float | None]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(draws, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def hierarchical_ci(frame: pd.DataFrame, column: str, draws: int = 2000) -> tuple[float | None, float | None]:
    """Resample training seeds, then independent scenario clusters within each model."""
    f = frame[np.isfinite(pd.to_numeric(frame[column], errors="coerce"))]
    groups = [x[column].to_numpy(dtype=float) for _, x in f.groupby("train_seed")]
    if len(groups) < 2:
        return bootstrap_mean(f[column].to_numpy(dtype=float), draws=draws)
    rng = np.random.default_rng(42)
    selected = rng.integers(len(groups), size=(draws, len(groups)))
    numerator = np.zeros(draws)
    denominator = np.zeros(draws)
    for i, values in enumerate(groups):
        sums = rng.choice(values, size=(draws, len(groups), len(values)), replace=True).sum(axis=2)
        used = selected == i
        numerator += (sums * used).sum(axis=1)
        denominator += used.sum(axis=1) * len(values)
    samples = numerator / denominator
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))  # type: ignore[return-value]


def holm(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def wilson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float | None, float | None]:
    """Diagnostic only: independent Bernoulli assumption does not hold for correlated slots."""
    if trials == 0:
        return None, None
    z = norm.ppf((1 + confidence) / 2)
    p = successes / trials
    center = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return float(max(0, center - half)), float(min(1, center + half))


def collect(root: Path) -> pd.DataFrame:
    rows, manifest = [], []
    for path in sorted((root / "results/runs").glob("*/summary.json")):
        run = json.loads(path.read_text())
        if run.get("status") != "COMPLETE":
            continue
        rows.append(
            {
                "run_id": run["run_id"],
                "experiment": run["experiment"],
                "method": run["method"],
                "dataset": run["dataset"],
                "scenario": run["scenario"],
                "seed": run["seed"],
                "train_seed": run["train_seed"],
                "cluster_id": run["cluster_id"],
                "split_id": run["split_id"],
                "runtime_seconds": run["runtime_seconds"],
                "ground_truth_kind": run["occupancy_ground_truth_kind"],
                **{k: v for k, v in run["metrics"].items() if not isinstance(v, (dict, list))},
            }
        )
        event = path.parent / "events.parquet"
        manifest.append(
            {
                "run_id": run["run_id"],
                "summary": str(path.relative_to(root)),
                "summary_sha256": file_hash(path),
                "events_sha256": file_hash(event) if event.exists() else None,
                "config_sha256": run["identity_sha256"],
                "trace_sha256": run["identity"]["trace_sha256"],
                "source_sha256": run["metadata"]["source_sha256"],
                "split": run["split_id"],
                "dataset": run["dataset"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No completed trial artifacts; run experiments first")
    frame.to_parquet(root / "results/all_runs.parquet", index=False)
    frame.to_csv(root / "results/all_runs.csv", index=False)
    write_json(root / "results/results_manifest.json", {"runs": manifest, "count": len(manifest)})
    return frame


def aggregate(frame: pd.DataFrame, draws: int = 2000) -> pd.DataFrame:
    rows = []
    for (experiment, method, split), group in frame.groupby(["experiment", "method", "split_id"]):
        # Real RF replications aggregate by source session before inference.
        if group["dataset"].iloc[0] == "nist":
            group = group.groupby("cluster_id", as_index=False).agg(
                {**{m: "mean" for m in METRICS}, "train_seed": "first"}
            )
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            lo, hi = hierarchical_ci(group, metric, draws) if len(finite) >= 2 else (None, None)
            rows.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "split_id": split,
                    "metric": metric,
                    "mean": float(finite.mean()) if len(finite) else None,
                    "sd": float(finite.std(ddof=1)) if len(finite) >= 2 else None,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_clusters": len(group),
                    "n_valid": len(finite),
                    "n_training_seed_groups": group["train_seed"].nunique(),
                    "interval": "hierarchical training-seed / scenario bootstrap, 2000 resamples",
                }
            )
    return pd.DataFrame(rows)


def paired_contrasts(frame: pd.DataFrame, draws: int = 2000) -> pd.DataFrame:
    rows = []
    eligible = frame[frame["split_id"].isin(["test", "synthetic"])]
    for experiment, group in eligible.groupby("experiment"):
        reference = group[group["method"] == "u_clada"]
        if reference.empty:
            continue
        family = []
        for method in sorted(set(group["method"]) - {"u_clada"}):
            comparison = group[group["method"] == method]
            paired = reference.merge(
                comparison, on=["seed", "cluster_id"], suffixes=("_u", "_b"), validate="one_to_one"
            )
            if len(paired) < 2:
                continue
            delta = paired["goodput_bps_u"] - paired["goodput_bps_b"]
            infer = pd.DataFrame({"delta": delta, "train_seed": paired["train_seed_u"]})
            lo, hi = hierarchical_ci(infer, "delta", draws)
            training_means = infer.groupby("train_seed")["delta"].mean().to_numpy()
            # Conservative test at training-seed group level; avoids treating shared models as independent.
            p = float(wilcoxon(training_means).pvalue) if np.any(training_means != 0) else 1.0
            mean = float(delta.mean())
            baseline = float(paired["goodput_bps_b"].mean())
            sd = float(delta.std(ddof=1))
            row = {
                "experiment": experiment,
                "reference": "u_clada",
                "baseline": method,
                "metric": "goodput_bps",
                "delta": mean,
                "ci_low": lo,
                "ci_high": hi,
                "relative_delta_percent": 100 * mean / abs(baseline) if baseline else None,
                "paired_standardized_effect": mean / sd if sd > 0 else None,
                "p_value": p,
                "paired_scenario_clusters": len(paired),
                "inference_training_groups": len(training_means),
                "family": str(experiment) + "/goodput",
                "test": "two-sided Wilcoxon on training-seed group mean paired differences",
            }
            family.append(row)
        adjusted = holm(np.array([r["p_value"] for r in family]))
        for row, p in zip(family, adjusted, strict=True):
            row["p_holm"] = float(p)
        rows.extend(family)
    return pd.DataFrame(rows)


def calibration_table(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = frame[(frame["split_id"] == "test") & (frame["method"].isin(["u_clada", "no_shield", "greedy"]))]
    for method, group in subset.groupby("method"):
        buffers = []
        for run_id in group["run_id"]:
            path = root / "results/runs" / run_id / "events.parquet"
            if path.exists():
                f = pd.read_parquet(path, columns=["forecast_busy", "hidden_busy", "forecast_uncertainty"])
                buffers.append(f.dropna())
        if not buffers:
            continue
        f = pd.concat(buffers, ignore_index=True)
        f["bin"] = pd.cut(f["forecast_busy"], np.linspace(0, 1, 11).tolist(), include_lowest=True)
        for interval, b in f.groupby("bin", observed=True):
            q, y = b["forecast_busy"].to_numpy(dtype=float), b["hidden_busy"].astype(float).to_numpy()
            rows.append(
                {
                    "method": method,
                    "bin": str(interval),
                    "predicted": float(q.mean()),
                    "observed": float(y.mean()),
                    "count": len(b),
                    "brier": float(np.mean((q - y) ** 2)),
                    "nll": float(
                        -np.mean(
                            y * np.log(np.maximum(q, 1e-12)) + (1 - y) * np.log(np.maximum(1 - q, 1e-12))
                        )
                    ),
                    "sampling_note": "Chosen sensed channels only; selection-biased diagnostics, no independent slot CI",
                }
            )
    return pd.DataFrame(rows)


def analyze(root: Path, draws: int = 2000) -> dict[str, Any]:
    frame = collect(root)
    groups = aggregate(frame, draws)
    paired = paired_contrasts(frame, draws)
    calibration = calibration_table(root, frame)
    adaptation = adaptation_table(root, frame)
    for name, f in [
        ("aggregate_metrics", groups),
        ("paired_contrasts", paired),
        ("calibration", calibration),
        ("adaptation", adaptation),
    ]:
        f.to_csv(root / f"results/{name}.csv", index=False)
        f.to_parquet(root / f"results/{name}.parquet", index=False)
    test = frame[frame["split_id"] == "test"]
    report = {
        "completed_runs": len(frame),
        "test_runs": len(test),
        "synthetic_scenario_clusters_test": int(test["cluster_id"].nunique()),
        "independent_real_sessions": int(test.loc[test["dataset"] == "nist", "cluster_id"].nunique()),
        "bootstrap_resamples": draws,
        "paired_test": "Wilcoxon at training-seed group level",
        "correction": "Holm within each experiment/goodput family",
        "overlap_intervals": "Cluster intervals; no independent Bernoulli assumption for correlated slots",
    }
    write_json(root / "reports/statistics/analysis.json", report)
    return report


def adaptation_table(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluator-only change diagnostics with a prespecified censored recovery rule."""
    rows = []
    subset = frame[frame["experiment"].str.startswith("E03-") & (frame["split_id"] == "synthetic")]
    for _, run in subset.iterrows():
        folder = root / "results/runs" / run["run_id"]
        summary = json.loads((folder / "summary.json").read_text())
        points = summary["trace_metadata"].get("change_points", [])
        if not points or not (folder / "events.parquet").exists():
            continue
        events = pd.read_parquet(folder / "events.parquet")
        service = events.groupby("slot")["success"].sum()
        alarms = np.sort(events.loc[events["drift_alarm"], "slot"].unique())
        for point in points:
            stop = min(point + 200, len(service))
            pre = float(service.iloc[max(0, point - 100) : point].mean())
            after = service.iloc[point:stop]
            recover = after.rolling(50, min_periods=50).mean()
            eligible: Any = recover.index[recover >= 0.9 * pre] if pre > 0 else []
            hits = alarms[(alarms >= point) & (alarms < stop)]
            rows.append(
                {
                    "run_id": run["run_id"],
                    "experiment": run["experiment"],
                    "method": run["method"],
                    "seed": run["seed"],
                    "change_slot": point,
                    "pre_packets_per_slot": pre,
                    "post_200_packets": int(after.sum()),
                    "post_window_slots": len(after),
                    "recovery_delay_slots": int(eligible[0] - point) if len(eligible) else None,
                    "recovery_censored": not bool(len(eligible)),
                    "first_alarm_delay_slots": int(hits[0] - point) if len(hits) else None,
                    "definition": "50-slot service mean >= 90% of preceding 100-slot service; censor after 200 slots; evaluator-only boundaries",
                }
            )
    return pd.DataFrame(rows)
