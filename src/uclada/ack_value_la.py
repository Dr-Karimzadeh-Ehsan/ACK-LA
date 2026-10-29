"""Coupled LA with past-ACK opportunity feedback on actually sensed idle defers.

The change addresses positive availability credit on a resource with poor access
success. It estimates an opportunity value; it never fabricates an ACK, reveals
a collision cause, or directly replaces the learned LA action distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from uclada.agents import la_update
from uclada.coupled_la import post_idle_risk
from uclada.env import LocalFeedback
from uclada.evidence_la import EvidenceLA


class AckValueLA(EvidenceLA):
    def update(self, fb: LocalFeedback) -> None:
        previous = self.p.copy()
        super().update(fb)
        c = self.config
        self.deferred_channel_value = None
        if not c.defer_ack_value or c.freeze_channel_la or fb.action == 0 or fb.attempted or fb.sensed_busy:
            return
        k = fb.action - 1
        success = (c.ack_prior_strength + self.ack_success[k]) / (
            c.ack_prior_strength + self.ack_success[k] + self.ack_failure[k]
        )
        risk = post_idle_risk(float(self.predicted_q[k]), float(self.predicted_u[k]), c)
        success = float(np.clip(success / max(1 - risk, 1e-6), 0, 1))
        value = success - c.access_target
        # Replace only this channel update. Belief/evidence/real ACK counting and
        # access-LA reinforcement from super() remain intact and are performed once.
        self.p = la_update(previous, k, value, self.eta_plus, c.defer_channel_penalty)
        self.deferred_channel_value = value

    def diagnostics(self) -> dict[str, Any]:
        return {
            **super().diagnostics(),
            "deferred_channel_value": getattr(self, "deferred_channel_value", None),
            "deferred_channel_penalty_rate": self.config.defer_channel_penalty,
        }
