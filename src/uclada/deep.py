"""Independent DQN using only local history, with replay and a target network.

This is an independent common-simulator baseline, not a reproduction of a cited paper.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
from torch import nn

from uclada.common import MethodConfig
from uclada.env import LocalFeedback, LocalObservation


class DQNLearner:
    def __init__(self, channels: int, config: MethodConfig, seed: int) -> None:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.channels, self.config = channels, config
        self.rng = np.random.default_rng(seed)
        features = 2 * (channels + 1) + 4
        self.feature_size = features
        self.history: deque[np.ndarray] = deque(maxlen=config.deep_history)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(features * config.deep_history, config.deep_hidden),
                nn.ReLU(),
                nn.Linear(config.deep_hidden, config.deep_hidden),
                nn.ReLU(),
                nn.Linear(config.deep_hidden, channels + 1),
            )
            self.target = nn.Sequential(
                nn.Linear(features * config.deep_history, config.deep_hidden),
                nn.ReLU(),
                nn.Linear(config.deep_hidden, config.deep_hidden),
                nn.ReLU(),
                nn.Linear(config.deep_hidden, channels + 1),
            )
        self.target.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.deep_lr)
        self.replay: deque[tuple[np.ndarray, int, float, np.ndarray]] = deque(maxlen=10000)
        self.steps = self.updates = 0
        self.training = True
        self.last_x: np.ndarray | None = None
        self.losses: list[float] = []

    def features(self, obs: LocalObservation, q: np.ndarray) -> np.ndarray:
        onehot = np.eye(self.channels + 1)[obs.last_action]
        missing = float(obs.last_ack is None)
        return np.r_[
            onehot,
            0.0,
            q,
            min(obs.queue_length / 100, 1),
            float(bool(obs.last_ack)),
            missing,
            obs.slot / (obs.slot + 100),
        ].astype(np.float32)

    def stack(self, feature: np.ndarray) -> np.ndarray:
        history = list(self.history) + [feature]
        history = history[-self.config.deep_history :]
        padding = [np.zeros(self.feature_size, dtype=np.float32)] * (self.config.deep_history - len(history))
        return np.concatenate(padding + history).astype(np.float32)

    def distribution(
        self, obs: LocalObservation, q: np.ndarray, mask: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        x = self.features(obs, q)
        state = self.stack(x)
        self.history.append(x)
        self.last_x = state
        with torch.no_grad():
            values = self.network(torch.from_numpy(state)[None])[0].numpy()
        values = np.where(mask, values, -np.inf)
        best = mask & np.isclose(values, values.max())
        evaluation_eps = (
            self.config.epsilon if self.config.deep_eval_epsilon is None else self.config.deep_eval_epsilon
        )
        eps = evaluation_eps if not self.training else max(self.config.epsilon, 1 - self.steps / 2000)
        return (1 - eps) * best / best.sum() + eps * mask / mask.sum()

    def observe(self, fb: LocalFeedback, q: np.ndarray) -> None:
        self.steps += 1
        nxt = LocalObservation(fb.slot + 1, fb.queue_length_next, fb.action, fb.ack)
        next_x = self.stack(self.features(nxt, q))
        if self.last_x is not None and (self.training or self.config.deep_updates):
            self.replay.append((self.last_x.copy(), fb.action, fb.reward, next_x))
        if not (self.training or self.config.deep_updates):
            return
        c = self.config
        if len(self.replay) < c.deep_batch or self.steps % c.deep_train_interval:
            return
        indices = self.rng.choice(len(self.replay), c.deep_batch, replace=False)
        batch = [self.replay[int(i)] for i in indices]
        states = torch.from_numpy(np.stack([b[0] for b in batch]))
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch]))
        values = self.network(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            targets = rewards + c.q_gamma * self.target(next_states).max(dim=1).values
        loss = nn.functional.smooth_l1_loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 5)
        self.optimizer.step()
        self.updates += 1
        self.losses.append(float(loss.detach()))
        if self.updates % 50 == 0:
            self.target.load_state_dict(self.network.state_dict())

    def begin_evaluation(self) -> None:
        self.training = False
        self.history.clear()

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.network.state_dict(),
                "target": self.target.state_dict(),
                "config": self.config.model_dump(),
                "updates": self.updates,
                "training_steps": self.steps,
                "losses": self.losses,
            },
            path,
        )

    def profile(self) -> dict[str, Any]:
        return {
            "parameters": sum(p.numel() for p in self.network.parameters()),
            "updates": self.updates,
            "observed_training_steps": self.steps,
        }
