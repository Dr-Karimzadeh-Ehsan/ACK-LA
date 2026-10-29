"""Evidence-directed channel and access learning automata (V3 research candidate).

The sole channel selector is the LA vector. Sensing evidence can strengthen a
bounded LA penalty; ACK history can reinforce a binary access LA. Neither module
receives the environment, true occupancy or a true packet-failure cause.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from uclada.agents import OccupancyBelief, la_update
from uclada.common import MethodConfig
from uclada.coupled_la import post_idle_risk
from uclada.env import LocalFeedback, LocalObservation, assert_no_leakage


class EvidenceLA:
    def __init__(self, name: str, channels: int, config: MethodConfig, seed: int):
        self.name, self.channels = name, channels
        self.config = config.model_copy(deep=True)
        c = self.config
        if c.primitive_epsilon_max < c.primitive_epsilon_min:
            raise ValueError("Exploration bounds must be ordered")
        self.rng = np.random.default_rng(seed)
        self.p = np.full(channels, 1 / channels)
        self.access = np.tile([1 - c.access_initial_send, c.access_initial_send], (channels, 1))
        self.belief = OccupancyBelief(channels, c)
        self.predicted_q = np.full(channels, 0.5)
        self.predicted_u = np.full(channels, 0.3)
        self.evidence: list[deque[tuple[int, float]]] = [
            deque(maxlen=c.evidence_window) for _ in range(channels)
        ]
        self.ack_success = np.zeros(channels)
        self.ack_failure = np.zeros(channels)
        self.ack_last = np.full(channels, -1, dtype=int)
        self.pressure = 0.2
        self.last_action = self.policy_action = 0
        self.access_action = 1
        self.risk_allowed = self.transmit_intent = True
        self.evidence_alarm = False
        self.evidence_score = 0.0
        self.eta_plus, self.eta_minus = c.eta_reward, c.eta_penalty
        self.epsilon = c.primitive_epsilon_min
        self.deep = None
        self.conditional_success_estimate = 1.0

    @property
    def evidence_threshold(self) -> float:
        # Union bound over N channels, <=T observed endpoints and <=W lengths.
        c = self.config
        return float(np.log(self.channels * c.evidence_horizon * c.evidence_window / c.evidence_delta))

    def primitive_distribution(self) -> np.ndarray:
        c = self.config
        self.epsilon = (
            c.primitive_epsilon_min + (c.primitive_epsilon_max - c.primitive_epsilon_min) * self.pressure
            if c.adaptive_exploration
            else c.primitive_epsilon_max
        )
        p = self.rng.permutation(self.p) if c.shuffle_channel_la else self.p
        distribution = np.r_[0.0, p * self.access[:, 1], p * self.access[:, 0]]
        return (1 - self.epsilon) * distribution + self.epsilon / len(distribution)

    def act(self, obs: LocalObservation) -> int:
        assert_no_leakage(obs)
        q, u = self.belief.predict(obs.slot)
        self.predicted_q, self.predicted_u = q, u
        if obs.queue_length == 0:
            self.last_action = self.policy_action = 0
            self.transmit_intent = False
            return 0
        encoded = int(self.rng.choice(2 * self.channels + 1, p=self.primitive_distribution()))
        self.policy_action = encoded
        if encoded == 0:
            self.last_action = 0
            self.transmit_intent = False
            return 0
        k = (encoded - 1) % self.channels
        self.last_action = k + 1
        self.access_action = int(encoded <= self.channels)
        self.risk_allowed = (
            not self.config.shield or post_idle_risk(float(q[k]), float(u[k]), self.config) <= self.config.rho
        )
        self.transmit_intent = bool(self.access_action and self.risk_allowed)
        return self.last_action

    def _busy_evidence(self, k: int, busy: bool, slot: int) -> bool:
        c = self.config
        pd, pfa = 1 - c.sensor_pmd, c.sensor_pfa
        # Exact zero likelihoods are handled with infinities, without invented observations.
        numerator, denominator = (pd, pfa) if busy else (1 - pd, 1 - pfa)
        if denominator == 0:
            value = np.inf if numerator > 0 else 0.0
        elif numerator == 0:
            value = -np.inf
        else:
            value = float(np.log(numerator / denominator))
        history = self.evidence[k]
        history.append((slot, value))
        while history and slot - history[0][0] > c.evidence_max_age:
            history.popleft()
        # Max over contiguous suffixes. +/-inf occur only in perfect/degenerate sensors;
        # impossible under both hypotheses is neutral; an idle observation ends a busy run.
        cumulative = 0.0
        score = 0.0
        for _, ell in reversed(history):
            if ell == -np.inf:
                break
            cumulative += ell
            score = max(score, cumulative)
        self.evidence_score = float(score) if np.isfinite(score) else 1e6
        return bool(c.evidence_enabled and busy and score >= self.evidence_threshold)

    def update(self, fb: LocalFeedback) -> None:
        assert_no_leakage(fb)
        c = self.config
        if fb.queue_length > 0:
            self.pressure += c.pressure_rate * ((not bool(fb.ack)) - self.pressure)
        if fb.action == 0:
            return
        if fb.sensed_busy is None:
            raise ValueError("Selected resource requires sensing feedback")
        k = fb.action - 1
        self.belief.observe(k, fb.sensed_busy, fb.slot)
        self.evidence_alarm = self._busy_evidence(k, fb.sensed_busy, fb.slot)
        multiplier = (
            1 + c.uncertainty_weight * float(self.predicted_u[k]) / 0.3 * c.uncertainty if c.adaptive else 1.0
        )
        self.eta_plus = float(np.clip(c.eta_reward * multiplier, 0.001, 0.3))
        self.eta_minus = float(np.clip(c.eta_penalty * multiplier, 0.0005, 0.1))
        reward = fb.reward
        if self.evidence_alarm:
            reward, self.eta_minus = -1.0, c.evidence_penalty
        elif not fb.attempted and not fb.sensed_busy:
            reward = float(np.clip(reward + c.availability_credit, -1, 1))
        if not c.freeze_channel_la:
            self.p = la_update(self.p, k, reward, self.eta_plus, self.eta_minus)
        # Exponential discount is based on actual elapsed slots, including unobserved gaps.
        gap = max(0, fb.slot - int(self.ack_last[k]))
        discount = c.ack_forget**gap
        self.ack_success[k] *= discount
        self.ack_failure[k] *= discount
        self.ack_last[k] = fb.slot
        if fb.attempted:
            self.ack_success[k] += int(bool(fb.ack))
            self.ack_failure[k] += int(not bool(fb.ack))
        if c.freeze_access_la or fb.sensed_busy or not self.risk_allowed:
            return
        if c.access_feedback == "utility":
            reinforcement = fb.reward if fb.attempted else -0.02
        else:
            success = (c.ack_prior_strength + self.ack_success[k]) / (
                c.ack_prior_strength + self.ack_success[k] + self.ack_failure[k]
            )
            # Auxiliary, model-dependent correction; never a revealed failure-cause label.
            risk = post_idle_risk(float(self.predicted_q[k]), float(self.predicted_u[k]), c)
            success = float(np.clip(success / max(1 - risk, 1e-6), 0, 1))
            self.conditional_success_estimate = success
            reinforcement = (success - c.access_target) * (2 * self.access_action - 1)
        self.access[k] = la_update(
            self.access[k], self.access_action, reinforcement, c.access_eta_reward, c.access_eta_penalty
        )

    def diagnostics(self) -> dict[str, Any]:
        k = self.last_action - 1
        return {
            "forecast_busy": float(self.predicted_q[k]) if self.last_action else None,
            "forecast_uncertainty": float(self.predicted_u[k]) if self.last_action else None,
            "la_entropy": float(-np.sum(self.p * np.log(np.maximum(self.p, 1e-300)))),
            "channel_la_selected_mass": float(self.p[k]) if self.last_action else None,
            "access_la_send_mass": float(self.access[k, 1]) if self.last_action else None,
            "eta_reward": self.eta_plus,
            "eta_penalty": self.eta_minus,
            "drift_alarm": self.evidence_alarm,
            "evidence_score": self.evidence_score,
            "evidence_threshold": self.evidence_threshold,
            "primitive_exploration": self.epsilon,
            "own_service_pressure": self.pressure,
            "conditional_ack_success_estimate": self.conditional_success_estimate,
            "transmit_intent": self.transmit_intent,
            "policy_action": self.policy_action,
            "forecast_context_mixing": 0.0,
        }

    def begin_evaluation(self, training_slots: int, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.belief.rebase(training_slots)
        self.ack_last -= training_slots
        self.evidence = [
            deque(((slot - training_slots, ell) for slot, ell in h), maxlen=self.config.evidence_window)
            for h in self.evidence
        ]
        self.evidence_alarm = False
