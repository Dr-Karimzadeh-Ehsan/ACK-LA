"""Publication exports drawn exclusively from saved trial artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from uclada.common import file_hash, source_hash, write_json

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "figure.dpi": 120,
    }
)


def save_figure(
    root: Path, name: str, fig: Any, source: pd.DataFrame, caption: str, run_ids: list[str] | None = None
) -> None:
    for extension in ["png", "pdf", "svg"]:
        folder = root / "figures" / extension
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.{extension}", dpi=600 if extension == "png" else None)
    plt.close(fig)
    data = root / "figures/source_data" / f"{name}.csv"
    data.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(data, index=False)
    write_json(
        data.with_suffix(".provenance.json"),
        {
            "figure": name,
            "source_data_sha256": file_hash(data),
            "run_ids": run_ids or [],
            "code_sha256": source_hash(),
            "caption": caption,
            "label": "UNCALIBRATED SYNTHETIC SIMULATION" if run_ids else "Architecture / source audit",
            "smoothing": "none",
        },
    )
    caption_path = root / "figures/captions.md"
    with caption_path.open("a") as file:
        file.write(f"\n**{name}.** {caption}\n")


def diagram(root: Path, name: str, labels: list[str], caption: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")
    positions = [(0.25 + (i % 4) * 2.5, 1.8 if i < 4 else 0.3) for i in range(len(labels))]
    for i, ((x, y), label) in enumerate(zip(positions, labels, strict=True)):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), 2.05, 0.8, boxstyle="round,pad=0.05", facecolor="#EEF5F8", edgecolor=PALETTE[0]
            )
        )
        ax.text(x + 1.025, y + 0.4, label, ha="center", va="center", fontsize=8)
        if i and i != 4:
            px, py = positions[i - 1]
            ax.add_patch(
                FancyArrowPatch((px + 2.1, py + 0.4), (x - 0.05, y + 0.4), arrowstyle="->", mutation_scale=12)
            )
    if len(labels) > 4:
        ax.text(
            5,
            1.35,
            "Evaluator truth is private; policies receive only local sensing and ACK",
            ha="center",
            color=PALETTE[3],
            fontsize=8,
        )
    save_figure(root, name, fig, pd.DataFrame({"node": labels}), caption)


def bar_metric(
    root: Path,
    name: str,
    frame: pd.DataFrame,
    aggregate: pd.DataFrame,
    experiment: str,
    metric: str,
    caption: str,
) -> bool:
    selected = aggregate[(aggregate["experiment"] == experiment) & (aggregate["metric"] == metric)]
    selected = selected[selected["mean"].notna()].sort_values("mean", ascending=False)
    if selected.empty:
        return False
    fig, ax = plt.subplots(figsize=(max(6, len(selected) * 0.42), 3.4))
    y = selected["mean"].to_numpy(dtype=float)
    lo = selected["ci_low"].fillna(selected["mean"]).to_numpy(dtype=float)
    hi = selected["ci_high"].fillna(selected["mean"]).to_numpy(dtype=float)
    ax.bar(
        np.arange(len(y)), y, color=[PALETTE[3] if m == "u_clada" else PALETTE[0] for m in selected["method"]]
    )
    ax.errorbar(
        np.arange(len(y)),
        y,
        yerr=np.maximum(0, np.vstack([y - lo, hi - y])),
        fmt="none",
        color="black",
        capsize=2,
    )
    ax.set_xticks(np.arange(len(y)), selected["method"], rotation=60, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(bottom=0)
    ax.set_title("Uncalibrated synthetic dynamics; hierarchical 95% intervals", fontsize=9)
    runs = frame.loc[frame["experiment"] == experiment, "run_id"].tolist()
    save_figure(root, name, fig, selected, caption, runs)
    return True


def make_figures(root: Path) -> dict[str, Any]:
    frame = pd.read_parquet(root / "results/all_runs.parquet")
    aggregate = pd.read_parquet(root / "results/aggregate_metrics.parquet")
    frame = frame[frame["split_id"].isin(["test", "synthetic"])].copy()
    main_names = frame.loc[frame["split_id"] == "test", "experiment"].unique()
    if not len(main_names):
        raise RuntimeError("No locked test results for figures")
    main = main_names[0]
    (root / "figures/captions.md").write_text("# Evidence-linked figure captions\n")
    generated, omitted = [], []
    for name, labels, caption in [
        (
            "fig01_framework_architecture",
            [
                "Exogenous occupancy\n(evaluator only)",
                "Per-SU local belief\nfrom past sensing",
                "LA probabilities +\ncontext + risk mask",
                "Candidate selection\nthen noisy sensing",
                "Simultaneous attempts\nprivate collision causes",
                "Own ACK + reward\nnext-slot LA update",
            ],
            "Causal distributed Tier-A architecture; training and global evaluation are declared separately.",
        ),
        (
            "fig02_data_pipeline",
            [
                "Official sources\nrequest receipts",
                "License + integrity\narray/header audit",
                "Timestamp + alignment\neligibility gate",
                "Chronological split\ntrain-only fitting",
                "Real replay if eligible\notherwise blocked",
                "Separately labeled\nsynthetic experiments",
            ],
            "Provenance and eligibility pipeline; raw-data access failure does not authorize synthetic substitution in a real suite.",
        ),
        (
            "fig05_algorithm_flow",
            [
                "Predict from\nlocal past only",
                "Temporary risk mask\nretain LA mass",
                "Sample LA × context\n0 means abstain",
                "Sense selected channel\nbusy => defer",
                "Resolve attempts\nprivate truth",
                "Update own belief\nLA and detector",
            ],
            "Actual implemented select-then-sense order. No ACK exists without a transmission attempt.",
        ),
    ]:
        diagram(root, name, labels, caption)
        generated.append(name)
    for name, exp, metric, caption in [
        (
            "fig06_main_goodput",
            main,
            "goodput_bps",
            "Main synthetic locked test. Absolute delivered payload rate with hierarchical 95% bootstrap intervals; no real RF performance claim.",
        ),
        (
            "fig11_ablation",
            "E07-ablation",
            "goodput_bps",
            "Paired prescribed component removals under the same synthetic exogenous occupancy. A service collapse is a reportable negative result.",
        ),
        (
            "fig18_runtime_efficiency",
            main,
            "inference_us_per_agent_slot",
            "Measured CPU inference time in microseconds per SU-slot; excludes training and includes policy decision calls. Concurrent worker contention applies.",
        ),
    ]:
        if bar_metric(root, name, frame, aggregate, exp, metric, caption):
            generated.append(name)
    # Distinct counters on the same physical denominator for visualization only.
    f = frame[frame["experiment"] == main]
    rates = f.groupby("method")[
        ["pu_overlap_per_attempt", "su_contention_per_attempt", "sensing_defer_rate", "abstention_rate"]
    ].mean()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    rates.plot.bar(ax=ax, width=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate (denominators differ; see caption)")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(fontsize=6)
    save_figure(
        root,
        "fig08_collisions",
        fig,
        rates.reset_index(),
        "Incumbent overlap and SU contention are conditional on attempts; sensing defer and abstention use all SU-slots. Undefined no-attempt risks are omitted, not zeroed.",
        f["run_id"].tolist(),
    )
    generated.append("fig08_collisions")
    risk = frame[frame["experiment"].str.startswith("E06-")]
    if not risk.empty:
        source = risk.groupby(["experiment", "method"], as_index=False).agg(
            goodput_bps=("goodput_bps", "mean"),
            risk=("pu_overlap_per_attempt", "mean"),
            abstention=("abstention_rate", "mean"),
        )
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for method, plotted_group in source.groupby("method"):
            ax.scatter(plotted_group["risk"], plotted_group["goodput_bps"], label=method, alpha=0.8)
        ax.axvline(0.1, ls="--", color="gray", label="empirical target 0.10")
        ax.set(xlabel="Incumbent overlap / attempt", ylabel="Goodput (bit/s)", xlim=(0, None), ylim=(0, None))
        ax.legend(fontsize=7)
        save_figure(
            root,
            "fig07_risk_throughput_pareto",
            fig,
            source,
            "Observed trade-off points, not a proven Pareto frontier. Empty-attempt episodes have undefined risk; forecast thresholds are not risk guarantees.",
            risk["run_id"].tolist(),
        )
        generated.append("fig07_risk_throughput_pareto")
    sensor = frame[frame["experiment"].str.startswith("E05-")].copy()
    if not sensor.empty:
        sensor["pfa"] = sensor["experiment"].str.extract(r"fa([0-9.]+)", expand=False).astype(float)
        sensor["pmd"] = sensor["experiment"].str.extract(r"md([0-9.]+)", expand=False).astype(float)
        source = sensor.groupby(["method", "pfa", "pmd"], as_index=False).agg(
            normalized_goodput=("normalized_goodput", "mean")
        )
        fig, axes = plt.subplots(1, source["method"].nunique(), figsize=(10, 2.8), squeeze=False)
        for ax, (method, group) in zip(axes.flat, source.groupby("method"), strict=True):
            p = group.pivot(index="pfa", columns="pmd", values="normalized_goodput")
            image = ax.imshow(p.to_numpy(), origin="lower", vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(p.columns)), p.columns)
            ax.set_yticks(range(len(p.index)), p.index)
            ax.set(xlabel="P_MD", ylabel="P_FA", title=method)
        fig.colorbar(image, ax=list(axes.flat), label="Successes / SU-slot", shrink=0.7)
        save_figure(
            root,
            "fig12_sensing_robustness",
            fig,
            source,
            "Full prespecified sensor-error cross-grid on synthetic occupancy, with sensor calibration matched in this panel.",
            sensor["run_id"].tolist(),
        )
        generated.append("fig12_sensing_robustness")
    for prefix, name, xcol, ycol, caption in [
        (
            "E04-",
            "fig13_scale_density",
            "agents",
            "normalized_goodput",
            "Short-horizon synthetic scalability screen, not an asymptotic convergence study. Distinct channel counts and overload configurations are retained.",
        ),
        (
            "E09-",
            "fig14_fairness_qos",
            "arrivals",
            "delay_p95_s",
            "Delivered-packet p95 delay under explicit queues; censored undelivered packets, drops and backlog must be read with the companion tables.",
        ),
        (
            "E10-",
            "fig17_switching_tradeoff",
            "switch_rate",
            "goodput_bps",
            "Switching/service response to prespecified modeled switching costs. This is not measured radio energy.",
        ),
        (
            "E15-",
            "fig19_risk_and_threshold_sensitivity",
            "eligible_fraction",
            "goodput_bps",
            "Synthetic controller sensitivity; RF occupancy-threshold sensitivity is unavailable without eligible raw RF time series.",
        ),
    ]:
        subset = frame[frame["experiment"].str.startswith(prefix)]
        if subset.empty:
            continue
        source = subset.groupby(["experiment", "method"], as_index=False)[
            list(dict.fromkeys([xcol, ycol, "channels", "agents"]))
        ].mean()
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for method, plotted_group in source.groupby("method"):
            ax.scatter(plotted_group[xcol], plotted_group[ycol], label=method, alpha=0.65)
        ax.set(xlabel=xcol.replace("_", " "), ylabel=ycol.replace("_", " "), ylim=(0, None))
        ax.legend(fontsize=7)
        save_figure(root, name, fig, source, caption, subset["run_id"].tolist())
        generated.append(name)
    calibration = pd.read_parquet(root / "results/calibration.parquet")
    if not calibration.empty:
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        for method, calibration_group in calibration.groupby("method"):
            ax.plot(calibration_group["predicted"], calibration_group["observed"], "o-", label=method)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(
            xlim=(0, 1), ylim=(0, 1), xlabel="Mean forecast busy", ylabel="Measured synthetic busy fraction"
        )
        ax.legend(fontsize=7)
        save_figure(
            root,
            "fig16_uncertainty_calibration",
            fig,
            calibration,
            "Chosen-channel reliability diagnostic on held-out synthetic evaluation. Points aggregate correlated observations and have no slot-independent confidence claim.",
            f["run_id"].tolist(),
        )
        generated.append("fig16_uncertainty_calibration")
    # Full raw cumulative trajectories, with pointwise across-scenario intervals for display.
    selected = frame[
        (frame["experiment"] == main) & frame["method"].isin(["u_clada", "lr_ep", "sw_ucb", "dqn"])
    ]
    curves = []
    for _, run in selected.iterrows():
        path = root / "results/runs" / run["run_id"] / "events.parquet"
        if not path.exists():
            continue
        events = pd.read_parquet(path, columns=["slot", "success"])
        counts = events.groupby("slot")["success"].sum().cumsum()
        curves.append(
            pd.DataFrame(
                {
                    "slot": counts.index,
                    "cumulative_successes": counts.values,
                    "method": run["method"],
                    "seed": run["seed"],
                }
            )
        )
    if curves:
        source = pd.concat(curves, ignore_index=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for method, plotted_group in source.groupby("method"):
            g = plotted_group.groupby("slot")["cumulative_successes"]
            mean, lo, hi = g.mean(), g.quantile(0.025), g.quantile(0.975)
            ax.plot(mean.index, mean, label=method)
            ax.fill_between(mean.index, lo, hi, alpha=0.12)
        ax.set(xlabel="Slot", ylabel="Cumulative delivered packets", ylim=(0, None))
        ax.legend(fontsize=7)
        save_figure(
            root,
            "fig09_online_learning",
            fig,
            source,
            "Unsmoothed cumulative packets; shaded bands are empirical 2.5–97.5% scenario quantiles, not confidence intervals of the mean.",
            selected["run_id"].tolist(),
        )
        generated.append("fig09_online_learning")
    # Representative evaluator-known change points and raw per-slot service.
    drift = frame[(frame["experiment"] == "E03-drift-abrupt") & (frame["seed"] == 0)]
    if not drift.empty:
        curves = []
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for _, run in drift.iterrows():
            events = pd.read_parquet(root / "results/runs" / run["run_id"] / "events.parquet")
            counts = events.groupby("slot")["success"].sum().cumsum()
            curves.append(
                pd.DataFrame({"slot": counts.index, "successes": counts.values, "method": run["method"]})
            )
            ax.plot(counts.index, counts, label=run["method"])
        for boundary in [666, 1332, 1998]:
            ax.axvline(boundary, color="gray", ls=":", lw=0.8)
        ax.set(xlabel="Slot", ylabel="Cumulative delivered packets", ylim=(0, None))
        ax.legend(fontsize=7)
        save_figure(
            root,
            "fig10_drift_adaptation",
            fig,
            pd.concat(curves),
            "Representative seed 0 abrupt-drift trace; evaluator-only regime boundaries are dotted. This single-seed illustration is not an adaptation significance test.",
            drift["run_id"].tolist(),
        )
        generated.append("fig10_drift_adaptation")
    contrasts = pd.read_parquet(root / "results/paired_contrasts.parquet")
    if not contrasts.empty:
        failure = contrasts.sort_values("delta").head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        failure_labels = failure["experiment"] + " / " + failure["baseline"]
        ax.barh(np.arange(len(failure)), failure["delta"], color=PALETTE[3])
        ax.set_yticks(np.arange(len(failure)), failure_labels, fontsize=6)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("U-CLADA minus comparator goodput (bit/s)")
        save_figure(
            root,
            "fig20_failure_cases",
            fig,
            failure,
            "Post hoc failure-case discovery from stored paired contrasts. Selection highlights losses and does not create an additional confirmatory test.",
            frame["run_id"].tolist(),
        )
        generated.append("fig20_failure_cases")
    omitted += [
        {
            "figure": "fig03_dataset_occupancy",
            "reason": "No eligible timestamped RF occupancy replay; never plot fake synchronized measured channels.",
        },
        {
            "figure": "fig04_dataset_dynamics",
            "reason": "RF chronology/cadence unverified; measured on/off durations cannot be inferred.",
        },
        {
            "figure": "fig15_cross_site_generalization",
            "reason": "No eligible independent measured replay sessions; synthetic OOD results are in tables.",
        },
    ]
    report = {"generated": generated, "omitted": omitted, "raster_dpi": 600, "vector": ["pdf", "svg"]}
    write_json(root / "reports/figure_inventory.json", report)
    return report


def make_tables(root: Path) -> dict[str, Any]:
    aggregate = pd.read_parquet(root / "results/aggregate_metrics.parquet")
    runs = pd.read_parquet(root / "results/all_runs.parquet")
    pairs = pd.read_parquet(root / "results/paired_contrasts.parquet")
    selected = aggregate[aggregate["split_id"].isin(["test", "synthetic"])]
    tables = {
        "T05_main_results": selected[selected["split_id"] == "test"],
        "T06_paired_statistics": pairs,
        "T07_ablations": selected[selected["experiment"].str.startswith("E07-")],
        "T08_drift": selected[selected["experiment"].str.startswith("E03-")],
        "T09_sensing": selected[selected["experiment"].str.startswith("E05-")],
        "T10_scale": selected[selected["experiment"].str.startswith("E04-")],
        "T11_synthetic_ood": selected[selected["experiment"].str.startswith("E08-")],
        "T12_qos": selected[selected["experiment"].str.startswith("E09-")],
        "T13_calibration": pd.read_parquet(root / "results/calibration.parquet"),
        "T15_compute": runs[runs["split_id"].isin(["test", "synthetic"])][
            [
                "run_id",
                "experiment",
                "method",
                "channels",
                "agents",
                "slots",
                "runtime_seconds",
                "inference_us_per_agent_slot",
                "peak_process_rss_bytes",
            ]
        ],
    }
    for name, path in [
        ("T01_verified_literature", root / "data/manifests/verified_references.csv"),
        ("T02_data_eligibility", root / "data/manifests/data_eligibility_matrix.csv"),
    ]:
        if path.exists():
            tables[name] = pd.read_csv(path)
    for name, table in tables.items():
        for extension in ["csv", "tex", "html"]:
            (root / "tables" / extension).mkdir(parents=True, exist_ok=True)
        table.to_csv(root / f"tables/csv/{name}.csv", index=False)
        table.to_latex(
            root / f"tables/tex/{name}.tex",
            index=False,
            float_format="%.6g",
            escape=True,
            longtable=len(table) > 40,
            caption="Artifact-generated results; consult protocol for units, denominators and independent clusters.",
        )
        table.to_html(
            root / f"tables/html/{name}.html", index=False, float_format=lambda value: f"{value:.6g}"
        )
    write_json(
        root / "reports/table_inventory.json",
        {
            "generated": {name: len(f) for name, f in tables.items()},
            "unavailable": {
                "T11_measured_cross_site": "No eligible measured RF sessions",
                "T12_rf_threshold_sensitivity": "No verified RF cadence/units",
            },
        },
    )
    return {name: len(f) for name, f in tables.items()}
