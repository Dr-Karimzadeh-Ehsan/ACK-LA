"""Tier-A packet access simulator with strictly separate local and evaluator views."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from uclada.common import EnvConfig, digest


@dataclass(frozen=True)
class TraceEpisode:
    occupancy: np.ndarray
    trace_id: str
    ground_truth_kind: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        b = np.array(self.occupancy, dtype=np.int8, copy=True)
        if b.ndim != 2 or not np.isin(b, [0, 1]).all():
            raise ValueError("Occupancy must be a binary slots × channels matrix, without imputed gaps")
        b.flags.writeable = False
        object.__setattr__(self, "occupancy", b)


@dataclass(frozen=True)
class LocalObservation:
    slot: int
    queue_length: int
    last_action: int
    last_ack: bool | None


@dataclass(frozen=True)
class LocalFeedback:
    slot: int
    action: int
    sensed_busy: bool | None
    attempted: bool
    ack: bool | None
    switched: bool
    queue_length: int
    reward: float
    queue_length_next: int = 0


@dataclass(frozen=True)
class Transition:
    agent_visible: tuple[LocalFeedback, ...]
    evaluator_private: tuple[dict[str, Any], ...]


def assert_no_leakage(value: LocalObservation | LocalFeedback) -> None:
    """Reject nonlocal or evaluator fields at the policy boundary."""
    if type(value) not in (LocalObservation, LocalFeedback):
        raise TypeError("Policy boundary only permits whitelisted frozen local records")
    forbidden = {"occupancy", "pu_overlap", "su_contention", "future", "other_actions", "hidden_busy"}
    if forbidden.intersection(asdict(value)):
        raise AssertionError("Hidden evaluator information crossed the policy boundary")


def synthetic_trace(slots: int, channels: int, seed: int, scenario: str = "abrupt") -> TraceEpisode:
    """Explicitly synthetic occupancy; parameters are not fitted to RF measurements."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 91]))
    rates = np.linspace(0.08, 0.78, channels)
    b = np.empty((slots, channels), dtype=np.int8)
    changes: list[int] = []
    if scenario == "toy_legacy":
        if channels != 5:
            raise ValueError("toy_legacy has exactly five channels")
        b[:] = rng.random((slots, channels)) >= np.array([0.25, 0.70, 0.90, 0.35, 0.80])
    elif scenario in {"always_idle", "always_busy"}:
        b[:] = int(scenario == "always_busy")
    else:
        state = rng.random(channels) < rates
        for t in range(slots):
            current = rates.copy()
            if scenario in {"abrupt", "mixed", "adversarial"}:
                length = max(10, slots // (20 if scenario == "adversarial" else 3))
                regime = t // length
                current = np.roll(rates, regime % channels)
                if t > 0 and t % length == 0:
                    changes.append(t)
            if scenario in {"gradual", "mixed"}:
                mix = t / max(1, slots - 1)
                current = (1 - mix) * current + mix * rates[::-1]
            if scenario == "periodic":
                current = np.clip(rates + 0.2 * np.sin(2 * np.pi * t / max(20, slots // 3)), 0.02, 0.98)
            if scenario == "bursty":
                current = np.clip(current + 0.25 * (np.sin(t / 30) > 0.8), 0.01, 0.99)
            persistence = 0.85 if scenario != "iid" else 0.0
            refresh = rng.random(channels) > persistence
            fresh = rng.random(channels) < current
            if scenario == "correlated":
                common = rng.random() < 0.25
                fresh = fresh | common
            state = np.where(refresh, fresh, state)
            b[t] = state
    return TraceEpisode(
        b,
        f"synthetic-{scenario}-{seed}",
        "synthetic_binary_state",
        {
            "dataset": "synthetic",
            "calibrated_from_real": False,
            "seed": seed,
            "scenario": scenario,
            "busy_rates_base": rates.tolist(),
            "change_points": changes,
            "persistence": 0.85 if scenario not in {"iid", "toy_legacy"} else 0,
            "timestamp_quality": "declared_simulation_slots",
            "source_sha256": digest(b),
        },
    )


class CRNEnvironment:
    """Simultaneous actions, per-(slot,user,channel) shared sensing uniforms and queues."""

    def __init__(self, trace: TraceEpisode, config: EnvConfig, seed: int) -> None:
        self.trace, self.config, self.seed = trace, config, seed
        self.slots, self.channels = trace.occupancy.shape
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1729]))
        self.sensing_uniforms = rng.random((self.slots, config.agents, self.channels))
        arrival_uniforms = rng.random((self.slots, config.agents))
        rates = np.full((self.slots, config.agents), config.arrival_rate)
        if config.traffic == "bursty":
            rates *= np.where((np.arange(self.slots)[:, None] // 50) % 2 == 0, 0.2, 1.8)
        self.arrivals = (arrival_uniforms < np.clip(rates, 0, 1)).astype(int)
        if config.traffic == "saturated":
            self.arrivals[:] = 0
        self.queues: list[deque[int]] = [deque() for _ in range(config.agents)]
        self.last_actions = np.zeros(config.agents, dtype=int)
        self.last_ack: list[bool | None] = [None] * config.agents
        self.rows: list[dict[str, Any]] = []
        self.delays: list[float] = []
        self.decision_times: list[float] = []
        self.slot = 0
        self.total_arrivals = self.drops = self.delivered = self.deadline_misses = 0

    def observation(self, agent: int) -> LocalObservation:
        backlog = 1 if self.config.traffic == "saturated" else len(self.queues[agent])
        obs = LocalObservation(self.slot, backlog, int(self.last_actions[agent]), self.last_ack[agent])
        assert_no_leakage(obs)
        return obs

    def step(self, actions: list[int], transmit_intents: list[bool] | None = None) -> Transition:
        c, t = self.config, self.slot
        if t >= self.slots:
            raise RuntimeError("Episode exhausted")
        a = np.asarray(actions, dtype=int)
        if a.shape != (c.agents,) or (a < 0).any() or (a > self.channels).any():
            raise ValueError("One valid action per SU required; 0 means idle")
        busy = self.trace.occupancy[t]
        intents = (
            np.ones(c.agents, dtype=bool)
            if transmit_intents is None
            else np.asarray(transmit_intents, dtype=bool)
        )
        if intents.shape != (c.agents,):
            raise ValueError("One local transmission intent per SU required")
        if not c.allow_sense_only and np.any((a > 0) & ~intents):
            raise ValueError("Sensing-only access must be explicitly enabled for the comparison protocol")
        sensed: list[bool | None] = []
        attempts = np.zeros(c.agents, dtype=bool)
        for m, action in enumerate(a):
            backlog = self.observation(m).queue_length
            if action and backlog == 0:
                raise ValueError("Empty queue must select idle")
            z = None
            if action:
                probability_busy = 1 - c.p_missed_detection if busy[action - 1] else c.p_false_alarm
                z = bool(self.sensing_uniforms[t, m, action - 1] < probability_busy)
                attempts[m] = not z and intents[m]
            sensed.append(z)
        counts = np.bincount(a[attempts], minlength=self.channels + 1)
        local: list[LocalFeedback] = []
        private: list[dict[str, Any]] = []
        for m, action in enumerate(a):
            selected = bool(action)
            attempt = bool(attempts[m])
            pu = bool(attempt and busy[action - 1])
            contention = bool(attempt and counts[action] >= 2)
            success = bool(attempt and not pu and not contention)
            ack = success if attempt else None
            switched = bool(t > 0 and action != self.last_actions[m])
            backlog_before = self.observation(m).queue_length
            delay: float | None = None
            if success:
                self.delivered += 1
                if c.traffic != "saturated":
                    entered = self.queues[m].popleft()
                    delay = (t - entered + 1) * c.slot_seconds
                    self.delays.append(delay)
                    self.deadline_misses += int(t - entered + 1 > c.deadline_slots)
            arrivals = int(self.arrivals[t, m])
            drops = 0
            if c.traffic != "saturated":
                self.total_arrivals += arrivals
                if arrivals and len(self.queues[m]) < c.queue_capacity:
                    self.queues[m].append(t + 1)
                elif arrivals:
                    drops = arrivals
                    self.drops += drops
            reward = float(success) - float(attempt and not success)
            reward -= c.switch_cost * switched + c.sensing_cost * selected
            reward -= c.idle_cost * (not selected and backlog_before > 0)
            reward -= c.queue_weight * min(backlog_before / c.queue_capacity, 1) * (not success)
            reward = float(np.clip(reward, -1, 1))
            fb = LocalFeedback(
                t,
                int(action),
                sensed[m],
                attempt,
                ack,
                switched,
                backlog_before,
                reward,
                1 if c.traffic == "saturated" else len(self.queues[m]),
            )
            assert_no_leakage(fb)
            local.append(fb)
            row = {
                "slot": t,
                "agent": m,
                "action": int(action),
                "sensed_busy": sensed[m],
                "attempted": attempt,
                "ack": ack,
                "success": success,
                "pu_overlap": pu,
                "su_contention": contention,
                "hidden_busy": bool(busy[action - 1]) if selected else None,
                "sensing_defer": bool(selected and sensed[m]),
                "access_defer": bool(selected and sensed[m] is False and not attempt),
                "abstain": not selected,
                "switched": switched,
                "sensed": selected,
                "reward": reward,
                "queue_before": backlog_before,
                "queue_after": len(self.queues[m]),
                "arrivals": arrivals,
                "drops": drops,
                "delay_s": delay,
            }
            private.append(row)
            self.rows.append(row)
            self.last_ack[m] = ack
        self.last_actions = a.copy()
        self.slot += 1
        return Transition(tuple(local), tuple(private))

    def metrics(self) -> dict[str, Any]:
        rows = pd.DataFrame(self.rows)
        count_fields = [
            "success",
            "attempted",
            "pu_overlap",
            "su_contention",
            "sensing_defer",
            "access_defer",
            "abstain",
            "switched",
            "sensed",
        ]
        counts = {f"n_{key}": int(rows[key].sum()) for key in count_fields}
        attempts = counts["n_attempted"]
        per_su = rows.groupby("agent")["success"].sum().to_numpy(dtype=float)
        fairness = (
            float(per_su.sum() ** 2 / (self.config.agents * np.sum(per_su**2))) if per_su.any() else None
        )
        t = self.slot
        busy_exposure = int(self.trace.occupancy[:t].sum()) * self.config.agents
        idle_opportunities = int((1 - self.trace.occupancy[:t]).sum())
        max_serviceable = int(
            np.minimum((1 - self.trace.occupancy[:t]).sum(axis=1), self.config.agents).sum()
        )

        def ratio(num, den):
            return float(num / den) if den else None

        return {
            **counts,
            "slots": t,
            "agents": self.config.agents,
            "channels": self.channels,
            "goodput_bps": counts["n_success"] * self.config.packet_bits / (t * self.config.slot_seconds),
            "normalized_goodput": counts["n_success"] / (t * self.config.agents),
            "pu_overlap_per_attempt": ratio(counts["n_pu_overlap"], attempts),
            "su_contention_per_attempt": ratio(counts["n_su_contention"], attempts),
            "pu_exposure_rate": ratio(counts["n_pu_overlap"], busy_exposure),
            "busy_exposure": busy_exposure,
            "idle_channel_slots": idle_opportunities,
            "oracle_success_bound": max_serviceable,
            "opportunity_capture": ratio(counts["n_success"], max_serviceable),
            "switch_rate": ratio(counts["n_switched"], self.config.agents * max(0, t - 1)),
            "switch_denominator": self.config.agents * max(0, t - 1),
            "abstention_rate": counts["n_abstain"] / (t * self.config.agents),
            "sensing_defer_rate": counts["n_sensing_defer"] / (t * self.config.agents),
            "access_defer_rate": counts["n_access_defer"] / (t * self.config.agents),
            "transmission_duty": counts["n_attempted"] / (t * self.config.agents),
            "jain_fairness": fairness,
            "per_user_successes": per_su.astype(int).tolist(),
            "delay_mean_s": float(np.mean(self.delays)) if self.delays else None,
            "delay_p95_s": float(np.quantile(self.delays, 0.95)) if self.delays else None,
            "n_delay_observations": len(self.delays),
            "deadline_misses_delivered": self.deadline_misses,
            "queue_backlog": sum(map(len, self.queues)),
            "arrivals": self.total_arrivals,
            "drops": self.drops,
            "traffic": self.config.traffic,
            "duration_s": t * self.config.slot_seconds,
        }
