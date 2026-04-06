"""LLM-augmented adaptive perception for OaK agents.

The perception module:
- Encodes raw observations into a latent subjective state via a
  configurable encoder (MLP, CNN, etc.) selected by the LLM or heuristic.
- Discovers and ranks features based on utility feedback.
- Generates subtasks from high-ranked features.

The encoder architecture is chosen at build time based on discovery
results and (optionally) LLM analysis.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from oak.interfaces import Perception
from oak.types import (
    FeatureId,
    FeatureSpec,
    SubtaskSpec,
    UtilityRecord,
)

from .encoders import Encoder


class AdaptivePerception(Perception[Any, Any, torch.Tensor]):
    """General-purpose perception driven by a pluggable encoder.

    The encoder is selected during the discovery phase (before the OaK
    agent is constructed) based on the observation type detected through
    trial-and-error, optionally refined by an LLM.
    """

    def __init__(
        self,
        encoder: Encoder,
        initial_features: Sequence[dict[str, Any]] | None = None,
        *,
        train_encoder: bool = False,
    ) -> None:
        self._encoder = encoder
        self._train_encoder = train_encoder
        self._state: torch.Tensor | None = None

        # Feature store
        self._features: dict[FeatureId, FeatureSpec] = {}
        self._created_subtask_for: set[FeatureId] = set()

        # Populate initial features (from LLM or generic)
        if initial_features:
            for feat in initial_features:
                fid = str(feat.get("id", f"feature_{len(self._features)}"))
                self._features[fid] = FeatureSpec(
                    feature_id=fid,
                    name=str(feat.get("name", fid)),
                    description=str(feat.get("description", "")),
                )

    # ------------------------------------------------------------------
    # Perception interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._state = None

    def update(
        self,
        observation: Any,
        reward: float,
        last_action: Any | None,
    ) -> torch.Tensor:
        self._state = self._encoder.encode(observation).detach()
        if self._train_encoder:
            self._encoder.train_step(observation)
        return self._state

    def current_subjective_state(self) -> torch.Tensor:
        if self._state is None:
            raise RuntimeError("update() has not been called yet")
        return self._state

    def discover_and_rank_features(
        self,
        subjective_state: torch.Tensor,
        utility_scores: Sequence[UtilityRecord],
        feature_budget: int,
    ) -> Sequence[FeatureId]:
        # Build a utility lookup for ranking
        utility_map: dict[str, float] = {
            rec.component_id: rec.utility for rec in utility_scores
        }

        # Rank features: those with higher utility first, then by insertion order
        ranked = sorted(
            self._features.keys(),
            key=lambda fid: utility_map.get(fid, 0.0),
            reverse=True,
        )
        return tuple(ranked[:feature_budget])

    def generate_subtasks(
        self,
        ranked_feature_ids: Sequence[FeatureId],
    ) -> Sequence[SubtaskSpec]:
        created: list[SubtaskSpec] = []
        for fid in ranked_feature_ids:
            if fid in self._created_subtask_for:
                continue
            if fid not in self._features:
                continue
            self._created_subtask_for.add(fid)
            created.append(
                SubtaskSpec(
                    subtask_id=f"subtask:{fid}",
                    name=f"Track {self._features[fid].name}",
                    feature_id=fid,
                )
            )
        return tuple(created)

    def list_features(self) -> Sequence[FeatureSpec]:
        return tuple(self._features.values())

    def remove_features(self, feature_ids: Sequence[FeatureId]) -> None:
        for fid in feature_ids:
            self._features.pop(fid, None)
            self._created_subtask_for.discard(fid)
