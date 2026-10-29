"""LA-centered channel/access learning and a fair shared primitive action interface.

Candidate novelty is unestablished: coupled/hierarchical automata have antecedents.
No neural predictor or forecast-context score controls the proposed channel selection.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from uclada.agents import LocalAgent, OccupancyBelief, PageHinkley, la_update, masked_distribution
from uclada.common import MethodConfig
from uclada.env import LocalFeedback, LocalObservation, assert_no_leakage


def post_idle_risk(q: float, u: float, config: MethodConfig) -> float:
    upper = float(np.clip(q + config.uncertainty_c * u * config.uncertainty, 0, 1))
    numerator = upper * config.sensor_pmd
    denominator = numerator + (1 - upper) * (1 - config.sensor_pfa)
    return numerator / max(denominator, 1e-12)


COUPLED_ABLATIONS = [
    "coupled_frozen_channel",
    "coupled_frozen_access",
    "coupled_fixed_rates",
    "coupled_no_drift",
    "coupled_no_risk",
    "coupled_no_availability_credit",
    "coupled_shuffled_channel",
    "coupled_no_uncertainty",
    "coupled_no_forecast",
    "coupled_ewma",
    "coupled_no_discount",
    "coupled_no_sensor_correction",
]


class CoupledLA:
    """One channel LA and one binary access LA per channel, using legitimate feedback."""

    def __init__(self, name: str, channels: int, config: MethodConfig, seed: int):
        self.name, self.channels = name, channels
        self.config = config.model_copy(deep=True)
        overrides: dict[str, dict[str, Any]] = {
            "coupled_frozen_channel": {"freeze_channel_la": True},
            "coupled_frozen_access": {"freeze_access_la": True},
            "coupled_fixed_rates": {"adaptive": False},
            "coupled_no_drift": {"drift": False},
            "coupled_no_risk": {"shield": False},
            "coupled_no_availability_credit": {"availability_credit": 0.0},
            "coupled_shuffled_channel": {"shuffle_channel_la": True},
            "coupled_no_uncertainty": {"uncertainty": False},
            "coupled_no_forecast": {"shield": False, "adaptive": False, "drift": False, "uncertainty": False},
            "coupled_ewma": {"predictor": "ewma"},
            "coupled_no_discount": {"forget": 1.0},
            "coupled_no_sensor_correction": {"sensor_correction": False},
        }
        for key, value in overrides.get(name, {}).items():
            setattr(self.config, key, value)
        self.rng = np.random.default_rng(seed)
        self.p = np.full(channels, 1 / channels)
        self.access = np.tile([0.2, 0.8], (channels, 1)).astype(float)
        self.belief = OccupancyBelief(channels, self.config)
        self.detector = PageHinkley(self.config.drift_threshold, self.config.drift_tolerance)
        self.predicted_q = np.full(channels, 0.5)
        self.predicted_u = np.full(channels, 0.3)
        self.last_action = 0
        self.access_action = 1
        self.transmit_intent = True
        self.risk_allowed = True
        self.eta_plus = self.config.eta_reward
        self.eta_minus = self.config.eta_penalty
        self.deep = None

    def act(self, obs: LocalObservation) -> int:
        assert_no_leakage(obs)
        q, u = self.belief.predict(obs.slot)
        self.predicted_q, self.predicted_u = q, u
        if obs.queue_length == 0:
            self.last_action = 0
            self.transmit_intent = False
            return 0
        p = self.rng.permutation(self.p) if self.config.shuffle_channel_la else self.p
        distribution = masked_distribution(p, np.ones(self.channels, dtype=bool), np.zeros(self.channels), 0)
        k = int(self.rng.choice(self.channels, p=distribution))
        self.last_action = k + 1
        self.access_action = int(self.rng.choice(2, p=0.98 * self.access[k] + 0.01))
        self.risk_allowed = (
            not self.config.shield or post_idle_risk(float(q[k]), float(u[k]), self.config) <= self.config.rho
        )
        self.transmit_intent = bool(self.access_action and self.risk_allowed)
        return self.last_action

    def update(self, fb: LocalFeedback) -> None:
        assert_no_leakage(fb)
        c = self.config
        if fb.action == 0:
            return
        k = fb.action - 1
        if fb.sensed_busy is None:
            raise ValueError("Selected candidate requires legitimate sensing feedback")
        previous_alarm = self.detector.alarm
        self.detector.alarm = False
        prediction = c.sensor_pfa + (1 - c.sensor_pmd - c.sensor_pfa) * self.predicted_q[k]
        self.belief.observe(k, fb.sensed_busy, fb.slot)
        if c.drift:
            self.detector.update(float(fb.sensed_busy) - prediction)
        multiplier = 1.0
        if c.adaptive:
            multiplier += c.uncertainty_weight * float(self.predicted_u[k]) / 0.3 * c.uncertainty + int(
                previous_alarm
            )
        self.eta_plus = float(np.clip(c.eta_reward * multiplier, 0.001, 0.3))
        self.eta_minus = float(np.clip(c.eta_penalty * multiplier, 0.0005, 0.1))
        # A sensed idle resource is evidence of availability, never a fabricated successful ACK.
        channel_reward = fb.reward
        if not fb.attempted and not fb.sensed_busy:
            channel_reward = float(np.clip(fb.reward + c.availability_credit, -1, 1))
        if not c.freeze_channel_la:
            self.p = la_update(self.p, k, channel_reward, self.eta_plus, self.eta_minus)
            if self.detector.alarm:
                self.p = (1 - c.reset_mix) * self.p + c.reset_mix / self.channels
        if not c.freeze_access_la and not fb.sensed_busy and self.risk_allowed:
            access_reward = fb.reward if fb.attempted else -0.02
            self.access[k] = la_update(
                self.access[k], self.access_action, access_reward, c.access_eta_reward, c.access_eta_penalty
            )

    def diagnostics(self) -> dict[str, Any]:
        a = self.last_action
        k = a - 1
        return {
            "forecast_busy": float(self.predicted_q[k]) if a else None,
            "forecast_uncertainty": float(self.predicted_u[k]) if a else None,
            "eligible_fraction": float(
                np.mean(
                    [
                        post_idle_risk(float(q), float(u), self.config) <= self.config.rho
                        for q, u in zip(self.predicted_q, self.predicted_u, strict=True)
                    ]
                )
            ),
            "la_entropy": float(-np.sum(self.p * np.log(np.maximum(self.p, 1e-300)))),
            "channel_la_selected_mass": float(self.p[k]) if a else None,
            "access_la_send_mass": float(self.access[k, 1]) if a else None,
            "eta_reward": self.eta_plus,
            "eta_penalty": self.eta_minus,
            "drift_alarm": bool(self.detector.alarm),
            "transmit_intent": self.transmit_intent,
            "policy_action": a,
            "forecast_context_mixing": 0.0,
        }

    def begin_evaluation(self, training_slots: int, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.belief.rebase(training_slots)
        self.detector = PageHinkley(self.config.drift_threshold, self.config.drift_tolerance)


class ActionBelief:
    """Share channel observations across transmit/sense-only copies of the same resource."""

    def __init__(self, channels: int, config: MethodConfig):
        self.channels = channels
        self.base = OccupancyBelief(channels, config)

    def predict(self, slot: int) -> tuple[np.ndarray, np.ndarray]:
        q, u = self.base.predict(slot)
        return np.tile(q, 2), np.tile(u, 2)

    def observe(self, channel: int, z: bool, slot: int) -> None:
        self.base.observe(channel % self.channels, z, slot)

    def rebase(self, slots: int) -> None:
        self.base.rebase(slots)


class PrimitiveBaseline:
    """Baseline action set: idle, N sense+transmit actions, N sense-only actions.

    DQN/Q-learning/adaptive bandits/classical LA can learn the same capabilities as
    the factorized proposed controller; no privileged action is reserved for it.
    """

    def __init__(self, name: str, channels: int, config: MethodConfig, seed: int):
        self.name, self.channels = name, channels
        self.base = LocalAgent(name, 2 * channels, config, seed)
        self.base.belief = ActionBelief(channels, self.base.config)  # type: ignore[assignment]
        self.config = self.base.config
        self.deep = self.base.deep
        self.last_encoded = 0
        self.transmit_intent = True

    def act(self, obs: LocalObservation) -> int:
        c = self.config
        shield = c.shield
        c.shield = False
        encoded = self.base.act(replace(obs, last_action=self.last_encoded))
        c.shield = shield
        if obs.queue_length > 0 and self.name in {
            "greedy",
            "epsilon_greedy",
            "epsilon_decay",
            "context_only",
        }:
            q = self.base.predicted_q[: self.channels]
            scores = np.r_[-0.02, 1 - 2 * q, np.full(self.channels, -0.03)]
            if self.name == "context_only":
                distribution = np.exp(c.temperature * (scores - scores.max()))
                distribution /= distribution.sum()
            else:
                best = np.isclose(scores, scores.max())
                eps = 0 if self.name == "greedy" else c.epsilon
                if self.name == "epsilon_decay":
                    eps /= np.sqrt(1 + obs.slot / 100)
                distribution = (1 - eps) * best / best.sum() + eps / len(scores)
            encoded = int(self.base.rng.choice(len(scores), p=distribution))
            self.base.last_action = encoded
            self.base.distribution = distribution
        self.last_encoded = encoded
        if encoded == 0:
            self.transmit_intent = False
            return 0
        physical = (encoded - 1) % self.channels + 1
        risk = post_idle_risk(
            float(self.base.predicted_q[physical - 1]), float(self.base.predicted_u[physical - 1]), c
        )
        self.transmit_intent = bool(encoded <= self.channels and (not shield or risk <= c.rho))
        return physical

    def update(self, fb: LocalFeedback) -> None:
        self.base.update(replace(fb, action=self.last_encoded))

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.base.diagnostics(),
            "policy_action": self.last_encoded,
            "transmit_intent": self.transmit_intent,
        }

    def begin_evaluation(self, training_slots: int, seed: int) -> None:
        self.base.begin_evaluation(training_slots, seed)
        self.last_encoded = 0


def create_agent(
    name: str, channels: int, config: MethodConfig, seed: int, expanded_actions: bool = False
) -> Any:
    if name == "ack_value_la":
        from uclada.ack_value_la import AckValueLA

        return AckValueLA(name, channels, config, seed)
    if name == "evidence_la":
        from uclada.evidence_la import EvidenceLA

        return EvidenceLA(name, channels, config, seed)
    if name == "coupled_la" or name in COUPLED_ABLATIONS:
        return CoupledLA(name, channels, config, seed)
    if expanded_actions:
        return PrimitiveBaseline(name, channels, config, seed)
    return LocalAgent(name, channels, config, seed)
