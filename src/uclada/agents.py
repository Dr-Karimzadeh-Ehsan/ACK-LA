"""Local causal policies, learning automata, risk heuristics and adaptive bandits."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from uclada.common import MethodConfig
from uclada.env import LocalFeedback, LocalObservation, assert_no_leakage


def la_update(p: np.ndarray, action: int, reward: float, eta_plus: float, eta_minus: float) -> np.ndarray:
    """Continuous-magnitude L_R-eP; eta_minus=0 gives reward–inaction."""
    if not (0 <= eta_plus <= 1 and 0 <= eta_minus <= 1 and -1 <= reward <= 1):
        raise ValueError("LA rates and bounded reward are outside their domains")
    if not 0 <= action < len(p):
        raise ValueError("Invalid action")
    if len(p) == 1:
        return np.ones(1)
    out = p.copy()
    if reward > 0:
        eta = eta_plus * reward
        out *= 1 - eta
        out[action] += eta
    elif reward < 0:
        b = eta_minus * -reward
        out *= 1 - b
        out += b / (len(p) - 1)
        out[action] -= b / (len(p) - 1)
    return out / out.sum()


def masked_distribution(
    p: np.ndarray, mask: np.ndarray, context: np.ndarray, tau: float, floor: float = 0.01
) -> np.ndarray:
    """Temporary mask; exploration cannot re-enable infeasible actions."""
    if p.shape != mask.shape or p.shape != context.shape or not mask.any():
        raise ValueError("Nonempty feasible support and matching dimensions required")
    logits = np.log(np.maximum(p, 1e-300)) + tau * (context - np.max(context[mask]))
    weights = np.where(mask, np.exp(np.clip(logits, -700, 700)), 0.0)
    weights /= weights.sum()
    return (1 - floor) * weights + floor * mask / mask.sum()


class OccupancyBelief:
    """Discounted excess pseudo-counts with noisy-sensing likelihood correction."""

    def __init__(self, channels: int, config: MethodConfig) -> None:
        self.config = config
        self.busy = np.zeros(channels)
        self.idle = np.zeros(channels)
        self.last = np.full(channels, -1, dtype=int)
        self.last_z = np.full(channels, 0.5)
        self.transitions = np.ones((channels, 2, 2))
        self.ewma = np.full(channels, 0.5)

    def predict(self, slot: int) -> tuple[np.ndarray, np.ndarray]:
        gap = np.maximum(0, slot - self.last)
        decay = self.config.forget**gap
        b, i = self.busy * decay, self.idle * decay
        q = (1 + b) / (2 + b + i)
        if self.config.predictor == "ewma":
            q = 0.5 + decay * (self.ewma - 0.5)
        elif self.config.predictor == "persistence":
            q = np.where(self.last >= 0, self.last_z, 0.5)
        elif self.config.predictor == "markov":
            q = np.empty_like(b)
            for k in range(len(b)):
                matrix = self.transitions[k] / self.transitions[k].sum(axis=1, keepdims=True)
                prior = np.array([1 - self.last_z[k], self.last_z[k]])
                q[k] = (prior @ np.linalg.matrix_power(matrix, int(min(gap[k], 1000))))[1]
        u = np.sqrt(np.maximum(q * (1 - q), 0) / (3 + b + i))
        return np.clip(q, 1e-8, 1 - 1e-8), u

    def observe(self, channel: int, sensed_busy: bool, slot: int) -> None:
        q, _ = self.predict(slot)
        z = int(sensed_busy)
        probability = q[channel]
        if self.config.sensor_correction:
            pd, pfa = 1 - self.config.sensor_pmd, self.config.sensor_pfa
            lb, li = (pd, pfa) if z else (1 - pd, 1 - pfa)
            den = probability * lb + (1 - probability) * li
            posterior = probability * lb / max(den, 1e-12)
        else:
            posterior = float(z)
        gap = max(0, slot - int(self.last[channel]))
        decay = self.config.forget**gap
        self.busy[channel] = self.busy[channel] * decay + posterior
        self.idle[channel] = self.idle[channel] * decay + 1 - posterior
        if self.last[channel] >= 0 and gap == 1:
            self.transitions[channel, int(self.last_z[channel]), z] += 1
        self.ewma[channel] += (1 - self.config.forget) * (posterior - self.ewma[channel])
        self.last[channel] = slot
        self.last_z[channel] = z

    def rebase(self, slots: int) -> None:
        self.last -= slots


class PageHinkley:
    """Two-sided causal change detector over observed sensing innovations."""

    def __init__(self, threshold: float, tolerance: float) -> None:
        self.threshold, self.tolerance = threshold, tolerance
        self.count = 0
        self.mean = self.up = self.down = 0.0
        self.alarm = False
        self.alarms = 0

    def update(self, value: float) -> bool:
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.up = max(0.0, self.up + value - self.mean - self.tolerance)
        self.down = min(0.0, self.down + value - self.mean + self.tolerance)
        self.alarm = self.up > self.threshold or -self.down > self.threshold
        if self.alarm:
            self.alarms += 1
            self.count = 0
            self.mean = self.up = self.down = 0.0
        return self.alarm


CORE_METHODS = [
    "random",
    "greedy",
    "epsilon_greedy",
    "epsilon_decay",
    "lri",
    "lr_ep",
    "q_learning",
    "sw_ucb",
    "discounted_ts",
    "exp3",
    "context_only",
    "u_clada",
    "dqn",
]
ABLATIONS = [
    "la_only",
    "adaptive_la",
    "fixed_rates",
    "no_drift",
    "no_uncertainty",
    "no_shield",
    "no_context",
    "no_discount",
    "no_sensor_correction",
    "ewma",
    "frozen_la",
    "shuffled_la",
    "mean_shield",
    "no_switch_reward",
    "no_queue_reward",
]


class LocalAgent:
    """One policy instance per SU; has no environment or other-agent references."""

    def __init__(self, name: str, channels: int, config: MethodConfig, seed: int) -> None:
        if name not in CORE_METHODS + ABLATIONS:
            raise ValueError(f"Unknown method {name}")
        self.name, self.channels = name, channels
        self.config = config.model_copy(deep=True)
        c = self.config
        overrides: dict[str, dict[str, Any]] = {
            "la_only": {"adaptive": False, "context": False, "shield": False, "drift": False},
            "adaptive_la": {"context": False, "shield": False},
            "fixed_rates": {"adaptive": False},
            "no_drift": {"drift": False},
            "no_uncertainty": {"uncertainty": False},
            "no_shield": {"shield": False},
            "no_context": {"context": False},
            "no_discount": {"forget": 1.0},
            "no_sensor_correction": {"sensor_correction": False},
            "ewma": {"predictor": "ewma"},
            "frozen_la": {"la_mode": "frozen"},
            "shuffled_la": {"la_mode": "shuffled"},
            "mean_shield": {"uncertainty": False},
        }
        for key, value in overrides.get(name, {}).items():
            setattr(c, key, value)
        self.rng = np.random.default_rng(seed)
        k = channels + 1
        self.p = np.full(k, 1 / k)
        self.belief = OccupancyBelief(channels, c)
        self.detector = PageHinkley(c.drift_threshold, c.drift_tolerance)
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.ts_alpha = np.ones(k)
        self.ts_beta = np.ones(k)
        self.logweights = np.zeros(k)
        self.window: deque[tuple[int, int, float]] = deque()
        self.qtable = np.zeros((k * 2, k))
        self.last_state = 0
        self.distribution = self.p.copy()
        self.predicted_q = np.full(channels, 0.5)
        self.predicted_u = np.full(channels, 0.3)
        self.mask = np.ones(k, dtype=bool)
        self.last_action = 0
        self.eta_plus, self.eta_minus = c.eta_reward, c.eta_penalty
        self.deep: Any = None
        self.last_observation: LocalObservation | None = None
        if name == "dqn":
            from uclada.deep import DQNLearner

            self.deep = DQNLearner(channels, c, seed)

    def act(self, obs: LocalObservation) -> int:
        assert_no_leakage(obs)
        self.last_observation = obs
        c, k = self.config, self.channels + 1
        q, u = self.belief.predict(obs.slot)
        self.predicted_q, self.predicted_u = q, u
        self.mask[:] = True
        is_proposed = self.name == "u_clada" or self.name in ABLATIONS or self.name == "context_only"
        if (is_proposed or c.shared_shield) and c.shield:
            upper = np.clip(q + c.uncertainty_c * u * c.uncertainty, 0, 1)
            self.mask[1:] = upper <= c.rho
        if obs.queue_length == 0:
            self.mask[1:] = False
        self.last_state = obs.last_action * 2 + int(bool(obs.last_ack))
        context = np.r_[-c.epsilon, 1 - 2 * q]
        if obs.last_action > 0:
            context[1:] -= 0.05 * (np.arange(1, k) != obs.last_action)
        if self.name == "random":
            distribution = self.mask / self.mask.sum()
        elif self.name in {"lri", "lr_ep"}:
            distribution = masked_distribution(self.p, self.mask, np.zeros(k), 0, c.baseline_la_floor)
        elif self.name in {
            "greedy",
            "epsilon_greedy",
            "epsilon_decay",
            "q_learning",
            "sw_ucb",
            "discounted_ts",
        }:
            if self.name in {"greedy", "epsilon_greedy", "epsilon_decay"}:
                scores = np.r_[0, 1 - q]
            elif self.name == "q_learning":
                scores = self.qtable[self.last_state].copy()
            elif self.name == "sw_ucb":
                while self.window and self.window[0][0] < obs.slot - c.window:
                    self.window.popleft()
                ns, sums = np.zeros(k), np.zeros(k)
                for _, a, reward in self.window:
                    ns[a] += 1
                    sums[a] += reward
                scores = sums / np.maximum(ns, 1) + np.sqrt(
                    2 * np.log(max(2, len(self.window))) / np.maximum(ns, 1e-12)
                )
            else:
                scores = self.rng.beta(self.ts_alpha, self.ts_beta)
            epsilon = 0 if self.name in {"greedy", "sw_ucb", "discounted_ts"} else c.epsilon
            if self.name == "epsilon_decay":
                epsilon /= np.sqrt(1 + obs.slot / 100)
            scores = np.where(self.mask, scores, -np.inf)
            best = self.mask & np.isclose(scores, np.max(scores))
            distribution = (1 - epsilon) * best / best.sum() + epsilon * self.mask / self.mask.sum()
        elif self.name == "exp3":
            base = np.exp(self.logweights - np.max(self.logweights))
            distribution = masked_distribution(base / base.sum(), self.mask, np.zeros(k), 0, c.exp3_gamma)
        elif self.name == "dqn":
            distribution = self.deep.distribution(obs, q, self.mask, self.rng)
        else:
            base = np.ones(k) / k if self.name == "context_only" else self.p.copy()
            if c.la_mode == "shuffled":
                base = self.rng.permutation(base)
            distribution = masked_distribution(
                base, self.mask, context if c.context else np.zeros(k), c.temperature
            )
        self.distribution = distribution
        self.last_action = int(self.rng.choice(k, p=distribution))
        return self.last_action

    def update(self, fb: LocalFeedback) -> None:
        assert_no_leakage(fb)
        c, a = self.config, fb.action
        self.detector.alarm = False
        if fb.sensed_busy is not None:
            k = a - 1
            predicted_z = c.sensor_pfa + (1 - c.sensor_pmd - c.sensor_pfa) * self.predicted_q[k]
            self.belief.observe(k, fb.sensed_busy, fb.slot)
            if c.drift:
                self.detector.update(float(fb.sensed_busy) - predicted_z)
        reward = fb.reward
        if self.name == "no_switch_reward":
            reward = float(np.clip(reward + 0.05 * fb.switched, -1, 1))
        if self.name == "no_queue_reward":
            reward = float(
                np.clip(reward + 0.01 * min(fb.queue_length / 100, 1) * (fb.ack is not True), -1, 1)
            )
        multiplier = 1.0
        if c.adaptive:
            uncertainty = float(np.mean(self.predicted_u)) / 0.3 if c.uncertainty else 0
            multiplier += c.uncertainty_weight * uncertainty + int(self.detector.alarm and c.drift)
        self.eta_plus = float(np.clip(c.eta_reward * multiplier, 0.001, 0.3))
        self.eta_minus = float(np.clip(c.eta_penalty * multiplier, 0.0005, 0.1))
        if self.name == "lri":
            self.p = la_update(self.p, a, reward, c.eta_reward, 0)
        elif self.name == "lr_ep":
            self.p = la_update(self.p, a, reward, c.eta_reward, c.eta_penalty)
        elif self.name == "u_clada" or self.name in ABLATIONS:
            if c.la_mode != "frozen":
                self.p = la_update(self.p, a, reward, self.eta_plus, self.eta_minus)
        self.counts[a] += 1
        self.values[a] += (reward - self.values[a]) / self.counts[a]
        self.window.append((fb.slot, a, (reward + 1) / 2))
        while len(self.window) > c.window + 1:
            self.window.popleft()
        self.ts_alpha = 1 + c.forget * (self.ts_alpha - 1)
        self.ts_beta = 1 + c.forget * (self.ts_beta - 1)
        bounded = (reward + 1) / 2
        self.ts_alpha[a] += bounded
        self.ts_beta[a] += 1 - bounded
        if self.name == "exp3":
            gain = bounded / max(self.distribution[a], 1e-12)
            self.logweights[a] += c.exp3_gamma * gain / len(self.p)
            self.logweights -= np.max(self.logweights)
        if self.name == "q_learning":
            next_state = a * 2 + int(bool(fb.ack))
            target = reward + c.q_gamma * self.qtable[next_state].max(axis=None)
            self.qtable[self.last_state, a] += c.q_alpha * (target - self.qtable[self.last_state, a])
        if self.deep is not None:
            self.deep.observe(fb, self.belief.predict(fb.slot + 1)[0])

    def diagnostics(self) -> dict[str, Any]:
        a = self.last_action
        return {
            "forecast_busy": float(self.predicted_q[a - 1]) if a else None,
            "forecast_uncertainty": float(self.predicted_u[a - 1]) if a else None,
            "eligible_fraction": float(self.mask[1:].mean()),
            "la_entropy": float(-np.sum(self.p * np.log(np.maximum(self.p, 1e-300)))),
            "eta_reward": self.eta_plus,
            "eta_penalty": self.eta_minus,
            "drift_alarm": bool(self.detector.alarm),
        }

    def begin_evaluation(self, training_slots: int, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.belief.rebase(training_slots)
        self.window.clear()
        self.detector = PageHinkley(self.config.drift_threshold, self.config.drift_tolerance)
        if self.deep is not None:
            self.deep.begin_evaluation()

    def local_state(self) -> dict[str, Any]:
        return {
            "p": self.p.tolist(),
            "belief_busy": self.belief.busy.tolist(),
            "belief_idle": self.belief.idle.tolist(),
            "config": self.config.model_dump(),
        }
