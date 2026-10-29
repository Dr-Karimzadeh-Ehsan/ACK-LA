"""Evidence-linked manuscript blocks, release gates and independent replay checks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from uclada.common import ExperimentConfig, MethodConfig, digest, file_hash, source_hash, write_json
from uclada.env import CRNEnvironment
from uclada.experiments import make_trace, play, train_agents


def manuscript(root: Path) -> dict[str, Any]:
    aggregate = pd.read_parquet(root / "results/aggregate_metrics.parquet")
    test = aggregate[(aggregate["split_id"] == "test") & (aggregate["metric"] == "goodput_bps")]
    full = pd.read_parquet(root / "results/all_runs.parquet")
    heldout = full[full["split_id"] == "test"]
    if test.empty:
        raise RuntimeError("Evidence-backed drafting requires completed held-out runs")
    main = test["experiment"].iloc[0]
    test = test[test["experiment"] == main]
    lines = [
        "# Results (evidence-backed synthetic study)",
        "",
        "The qualifying real multichannel RF experiment is **NOT RUN / UNAVAILABLE**.",
        "The following values describe uncalibrated synthetic occupancy with simulated SU interactions.",
        "They must not be described as real-world radio performance or measured PU receiver interference.",
        "",
        "| Method | Goodput (bit/s) | Hierarchical 95% CI | Scenario clusters | Training seeds |",
        "|---|---:|---|---:|---:|",
    ]
    for _, row in test.sort_values("mean", ascending=False).iterrows():
        lo = f"{row['ci_low']:.1f}" if pd.notna(row["ci_low"]) else "NA"
        hi = f"{row['ci_high']:.1f}" if pd.notna(row["ci_high"]) else "NA"
        lines.append(
            f"| {row['method']} | {row['mean']:.1f} | [{lo}, {hi}] | {int(row['n_clusters'])} | {int(row['n_training_seed_groups'])} |"
        )
    lines += [
        "",
        "Source: `results/aggregate_metrics.parquet`, filter split_id=test, metric=goodput_bps.",
        "Risk, abstention, contention, fairness and all denominators accompany these values in T05.",
        "Paired contrasts and Holm-adjusted tests are generated in T06; a higher service mean alone",
        "does not establish constrained superiority. No settings were selected from test results.",
    ]
    out = root / "manuscript"
    out.mkdir(exist_ok=True)
    (out / "results.md").write_text("\n".join(lines) + "\n")
    blocks = {
        "outline.md": """# Working article structure
1. Abstract (bounded by synthetic evidence)
2. Introduction and falsifiable objectives
3. Verified related work and adaptation boundaries
4. System and information model
5. Uncertainty-aware LA controller
6. Source eligibility audit and experimental protocol
7. Results and negative controls
8. Discussion, limitations and threats to validity
9. Reproducibility and data availability

The working measurement-driven title cannot be used as a validated empirical claim
until qualifying measured traces are obtained. Journal quartile/acceptance is not established.
""",
        "introduction.md": """# Introduction
Distributed dynamic spectrum access requires secondary users to choose channels from
local, delayed and imperfect evidence while avoiding occupied resources and mutual
contention. Success rate alone cannot identify incumbent overlap: a failed service
opportunity can also result from another secondary user, abstention or sensing defer.
We study a local learning-automata controller that combines continuous-magnitude
reinforcement, discounted occupancy beliefs, a causal change detector and a forecast
threshold shield. Each component has antecedents; their integration is a candidate
contribution rather than an established novelty claim.

The implemented contribution is an auditable common-simulator evaluation, explicit
causal information boundary, paired scenario design and a failure analysis of the
service/risk trade-off. Measurement-source accessibility and temporal alignment are
treated as eligibility requirements, not inferred from dataset names. The current
empirical study is synthetic; measurement-driven generalization remains open.
""",
        "experimental_setup.md": """# Experimental setup
All methods use overlay select-then-sense packet access with one sensed candidate,
hard one-service-per-idle-channel collisions and exogenous occupancy. Evaluator truth
is absent from policy observations. Sensor corruption is relative to synthetic state
or a threshold-derived proxy, never independently verified physical PU ground truth.
Payload size, slot duration, rewards and queues are recorded per run.

Three bounded validation candidates per implemented core method use five training
seeds. Unshielded simulated training is explicitly accounted separately and does not
constitute a safe cold start. Test seeds and selected configurations are hashed in
results/test_lock.json. Online belief/LA/bandit adaptation continues during evaluation;
DQN uses frozen pretrained weights by default with local history features and replay
only during training. Its training budget is finite and no native-paper reproduction
or universal convergence claim is made. Uncertainty calibration uses centralized
validation labels and an empirical bin-mean target; execution requires no messages.

Inference uses independent synthetic scenario clusters and hierarchical training-seed
resampling. The paired Wilcoxon test uses training-seed group means, with Holm
adjustment per experiment. Multiple policy seeds on one RF session would not count
as multiple independent measured environments. Read the registry for actual executed
families and the completion report for unavailable measured experiments.
""",
        "discussion.md": """# Discussion
Forecast thresholding is not a guarantee about incumbent overlap conditional on
transmission. After a noisy idle sense, the relevant risk depends on the prior and
sensor likelihoods; empirical overlap additionally depends on selection and drift.
The strict preselection shield can remove every channel before any new observation
can be collected. Without a declared safe sensing-only action, this is an absorbing
information starvation failure. Offline unshielded training can initialize beliefs
but does not remove that online failure mechanism or provide safe exploration.

The proposed integration should be assessed against simple adaptive bandits and
context-only controls at comparable observed risk, including lost service. Null and
negative results remain in the saved artifacts. Any algorithm revision motivated by
these tests needs a new version and untouched future test scenarios; it must not
silently replace the locked results. Exact reproduction of contemporary published
DSA/MARL algorithms remains separate from our independent DQN implementation.
""",
        "limitations.md": """# Limitations and threats to validity
The current primary real RF replay is unavailable. NIST catalog availability is not
raw time-series availability. ElectroSense sequential sweeps and numeric matrix axes
do not establish simultaneous channel states or sample cadence. Its PSD is measured
RF, not observed secondary-user allocations or receiver-level harmful interference.
Synthetic generators are uncalibrated unless a separate training-data fit is recorded.

The Tier-A collision model omits propagation, capture, hidden terminals, within-slot
incumbent changes and sensing corruption caused by simultaneous SU transmitters.
Absolute bit rates use declared packet and simulated slot parameters. No measured
energy or hardware deployment result is claimed. Short scale/sensor screens are
finite-horizon sensitivity evidence, not a full asymptotic convergence analysis.
Finite DQN tuning/training may not match more specialized or better trained models.
Novelty requires a fuller equation-level literature review and a submission-ready
measurement-driven claim requires stronger source data. No Q1 acceptance guarantee exists.
""",
        "reproducibility.md": """# Reproducibility and availability
The repository stores exact dependency versions, source hashes, test lock, training
checkpoints, per-trial configurations, private event logs, public/local observations,
scalar numerators/denominators, bootstrap tables and vector/600-dpi figures. Raw RF
data remain outside Git and follow source-specific licenses. This local workspace
has not been published publicly. Follow README and run_research_full.sh to rerun.
The final verification report distinguishes deterministic independent replay from a
fresh-machine/full-dataset reproduction, which remains unestablished when source data
cannot be downloaded. Author names, funding, conflicts and target venue are not invented.
""",
    }
    for name, content in blocks.items():
        (out / name).write_text(content)
    for source, dest in [
        ("docs/methodology.md", "methods.md"),
        ("docs/related_work_2022_2026.md", "related_work.md"),
        ("figures/captions.md", "captions.md"),
        ("references.bib", "references.bib"),
    ]:
        if (root / source).exists():
            shutil.copyfile(root / source, out / dest)
    u = test[test["method"] == "u_clada"]
    abstract = "# Abstract (draft; synthetic evidence only)\n\n"
    abstract += "We implement and audit a distributed learning-automata spectrum-access controller with noisy local sensing, uncertainty-aware forecast masking and nonstationary adaptation. "
    abstract += "A deterministic packet-level environment separates incumbent overlap, SU contention, sensing defer and abstention, and enforces causal observation boundaries. "
    if not u.empty:
        abstract += f"In the locked synthetic benchmark, the proposed method delivered {u.iloc[0]['mean']:.1f} bit/s averaged over {int(u.iloc[0]['n_clusters'])} scenario clusters. "
    abstract += "Complete baseline and risk trade-offs are reported without assuming superiority. Strict risk masking can produce information starvation and service loss. Official RF data were audited for replay eligibility; qualifying simultaneous measured-channel evaluation remains unavailable. These findings support simulator-level conclusions only.\n"
    (out / "abstract.md").write_text(abstract)
    order = [
        "abstract.md",
        "introduction.md",
        "related_work.md",
        "methods.md",
        "experimental_setup.md",
        "results.md",
        "discussion.md",
        "limitations.md",
        "reproducibility.md",
    ]
    (out / "article_draft.md").write_text(
        "\n\n".join((out / file).read_text() for file in order if (out / file).exists())
    )
    write_json(
        root / "reports/manuscript_evidence.json",
        {
            "source_table": "results/aggregate_metrics.parquet",
            "source_sha256": file_hash(root / "results/aggregate_metrics.parquet"),
            "test_runs": heldout["run_id"].tolist(),
            "measurement_claim_supported": False,
        },
    )
    return {"draft": "manuscript/article_draft.md", "test_rows": len(heldout)}


def independent_replay(root: Path, summary_path: Path) -> dict[str, Any]:
    saved = json.loads(summary_path.read_text())
    identity = saved["identity"]
    config = ExperimentConfig.model_validate(identity["config"])
    method = MethodConfig.model_validate(identity["method"])
    trace = make_trace(config, saved["seed"], saved["split_id"])
    agents, training = train_agents(root, config, saved["method"], method, saved["train_seed"])
    for m, agent in enumerate(agents):
        agent.config.shield = method.shield
        if saved["method"] in {"no_shield", "adaptive_la", "la_only"}:
            agent.config.shield = False
        agent.config.deep_updates = method.deep_updates
        agent.begin_evaluation(training["slots"], config.root_seed + saved["seed"] * 1000 + m)
    env = CRNEnvironment(trace, config.env, config.root_seed + saved["seed"])
    play(env, agents)
    measured = env.metrics()
    expected = {key: saved["metrics"][key] for key in measured}
    equal = digest(measured) == digest(expected)
    return {
        "run_id": saved["run_id"],
        "metrics_identical": equal,
        "recomputed_hash": digest(measured),
        "saved_hash": digest(expected),
        "procedure": "Reconstruct exogenous episode; reload training checkpoint; rerun policy decisions, sensing and feedback without trial-result cache",
    }


def verify(root: Path, replay: bool = True) -> dict[str, Any]:
    manifest_path = root / "results/results_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Analyze results before verifying the manifest")
    manifest = json.loads(manifest_path.read_text())
    errors, replays = [], []
    for record in manifest["runs"]:
        path = root / record["summary"]
        if file_hash(path) != record["summary_sha256"]:
            errors.append(f"Summary integrity failed: {record['run_id']}")
        event = path.parent / "events.parquet"
        if record.get("events_sha256") and (
            not event.exists() or file_hash(event) != record["events_sha256"]
        ):
            errors.append(f"Event integrity failed: {record['run_id']}")
    if replay:
        for name in ["lr_ep", "sw_ucb", "u_clada", "dqn"]:
            candidate = None
            for item in manifest["runs"]:
                path = root / item["summary"]
                run = json.loads(path.read_text())
                if run["method"] == name and run["split_id"] == "test":
                    candidate = path
                    break
            if candidate is not None:
                result = independent_replay(root, candidate)
                replays.append(result)
                if not result["metrics_identical"]:
                    errors.append(f"Replay mismatch {name}")
    for path in (root / "figures/source_data").glob("*.provenance.json"):
        entry = json.loads(path.read_text())
        data = path.with_name(path.name.replace(".provenance.json", ".csv"))
        if not data.exists() or file_hash(data) != entry["source_data_sha256"]:
            errors.append(f"Figure data hash failed: {path.name}")
    lockpath = root / "results/test_lock.json"
    if lockpath.exists():
        lock = json.loads(lockpath.read_text())
        if lock["identity"]["source_sha256"] != source_hash():
            errors.append("Source changed since test lock")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "integrity_runs_checked": len(manifest["runs"]),
        "independent_replays": replays,
        "errors": errors,
        "fresh_machine_reproduction": "NOT RUN",
        "source_sha256": source_hash(),
    }
    write_json(root / "reports/verification.json", report)
    lines = [
        "# Reproduction report",
        "",
        f"Artifact integrity and local independent replay: **{report['status']}**.",
        f"Checked {len(manifest['runs'])} immutable trial summaries and their stored event hashes.",
        "",
        "This is a local re-execution using stored training checkpoints, not a claim of fresh-machine or real-data reproduction.",
        "",
        "```json",
        json.dumps(report, indent=2),
        "```",
    ]
    (root / "reports/reproduction_report.md").write_text("\n".join(lines) + "\n")
    return report


def completion_report(root: Path) -> dict[str, Any]:
    runs_path = root / "results/all_runs.parquet"
    frame = pd.read_parquet(runs_path) if runs_path.exists() else pd.DataFrame()
    registry = (
        pd.read_csv(root / "experiment_registry.csv")
        if (root / "experiment_registry.csv").exists()
        else pd.DataFrame()
    )
    verification = (
        json.loads((root / "reports/verification.json").read_text())
        if (root / "reports/verification.json").exists()
        else {}
    )
    gates = [
        {
            "gate": "G0 sources & feasibility",
            "status": "PARTIAL",
            "evidence": "reports/dataset/; data/manifests/verified_references.csv",
            "blocker": "NIST raw server timeout; novelty full-text audit incomplete",
        },
        {
            "gate": "G1 real data & schemas",
            "status": "PARTIAL",
            "evidence": "data/manifests/data_eligibility_matrix.csv",
            "blocker": "No verified contemporaneous multichannel RF trace; no real-calibrated generator",
        },
        {
            "gate": "G2 simulator correctness",
            "status": "PASS" if (root / "reports/correctness/passed.json").exists() else "PARTIAL",
            "evidence": "reports/correctness/pytest.xml",
            "blocker": "Tier A only; PHY/feedback-delay extensions unavailable",
        },
        {
            "gate": "G3 LA & baselines",
            "status": "PASS" if len(frame) else "PARTIAL",
            "evidence": "results/selected_settings.json; results/training/; reports/baseline_fidelity/",
            "blocker": "Independent DQN; no exact recent-paper reproduction claimed",
        },
        {
            "gate": "G4 experimental grid",
            "status": "PARTIAL",
            "evidence": "experiment_registry.csv; results/results_manifest.json",
            "blocker": "E02 measured primary and E12 real RF threshold sensitivity unavailable; finite-horizon synthetic grids",
        },
        {
            "gate": "G5 statistical artifacts",
            "status": "PASS" if (root / "results/paired_contrasts.parquet").exists() else "PARTIAL",
            "evidence": "results/aggregate_metrics.parquet; tables/; figures/",
            "blocker": "Synthetic inference only; measured-population claims unavailable",
        },
        {
            "gate": "G6 manuscript & reproduction",
            "status": "PARTIAL",
            "evidence": "manuscript/article_draft.md; reports/reproduction_report.md",
            "blocker": "Not submission ready: measured validation, fuller novelty review and fresh-machine reproduction remain open",
        },
    ]
    report = {
        "overall_status": "UNFINISHED_FOR_Q1_SUBMISSION",
        "gates": gates,
        "completed_trial_runs": len(frame),
        "registered_experiments": len(registry),
        "verification": verification.get("status", "NOT RUN"),
    }
    write_json(root / "reports/completion.json", report)
    lines = [
        "# Research completion report",
        "",
        "**Overall: unfinished for Q1 submission.**",
        "Runnable synthetic research artifacts are distinguished from unavailable measurement-driven evidence.",
        "",
        "| Gate | Status | Evidence | Exact open issue |",
        "|---|---|---|---|",
    ]
    for gate in gates:
        lines.append(f"| {gate['gate']} | {gate['status']} | {gate['evidence']} | {gate['blocker']} |")
    lines += [
        "",
        f"Completed immutable trials: {len(frame)}. Registered experiment configurations: {len(registry)}.",
        "",
        "E00 source audit executed; E01 toy sanity and E03–E11/E13/E15 synthetic families are listed individually",
        "in the registry. E10 online trajectories and E11 inference profiling reuse actual event/compute logs.",
        "E14 uses independent DQN in the common simulator; no original-paper numerical reproduction is asserted.",
        "E02, measured E08, E12 and F03/F04/F15 remain unavailable without source-eligible RF time series.",
        "Sparse sensing budgets >1, delayed ACK, spatial SINR and primal–dual control are not implemented;",
        "they must not be represented as executed extensions.",
    ]
    (root / "RESEARCH_COMPLETION_REPORT.md").write_text("\n".join(lines) + "\n")
    with (root / "progress.md").open("a") as file:
        file.write("\n## Latest evidence audit\n\n" + "\n".join(lines[4:]) + "\n")
    return report
