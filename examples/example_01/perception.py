"""LLM-planned adaptive perception for OaK Example 01."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from oak.interfaces import Perception
from oak.types import (
    FeatureId,
    FeatureSpec,
    SubtaskSpec,
    UtilityRecord,
)

from .encoders import Encoder
from .schema import ExampleSubjectiveState, PerceptionPlan
from .startup import (
    extract_semantic_field,
    normalize_observation,
    tensor_input_from_observation,
)


@dataclass
class _RunningVectorStats:
    """Online running moments for one raw-value tensor view."""

    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def normalize(self, value: np.ndarray) -> np.ndarray:
        flat = np.asarray(value, dtype=np.float32).reshape(-1)
        if self.count < 2 or self.mean is None or self.m2 is None:
            return flat
        variance = self.m2 / max(self.count - 1, 1)
        scale = np.sqrt(np.maximum(variance, 1e-4))
        normalized = (flat - self.mean) / scale
        return np.clip(normalized, -5.0, 5.0).astype(np.float32)

    def update(self, value: np.ndarray) -> None:
        flat = np.asarray(value, dtype=np.float32).reshape(-1)
        if self.mean is None or self.m2 is None:
            self.mean = np.zeros_like(flat)
            self.m2 = np.zeros_like(flat)
        self.count += 1
        delta = flat - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = flat - self.mean
        self.m2 = self.m2 + delta * delta2


class AdaptivePerception(Perception[Any, Any, ExampleSubjectiveState]):
    """Perception driven by a startup-time perception plan.

    The world still emits raw observations. This module owns:

    - normalization into `AgentObservation`
    - grouped semantic field extraction
    - encoder-based tensor views for lower learners
    """

    def __init__(
        self,
        perception_plan: PerceptionPlan,
        encoders: Mapping[str, Encoder],
        *,
        train_encoder: bool = False,
        subtask_creation_interval: int = 500,
        normalize_raw_views: bool = False,
    ) -> None:
        self._perception_plan = perception_plan
        self._encoders = dict(encoders)
        self._encoder = self._encoders[perception_plan.default_tensor_view]
        self._train_encoder = train_encoder
        self._subtask_creation_interval = max(subtask_creation_interval, 1)
        self._normalize_raw_views = normalize_raw_views
        self._state: ExampleSubjectiveState | None = None
        self._update_count = 0
        self._last_subtask_creation_update = -self._subtask_creation_interval

        self._features: dict[FeatureId, FeatureSpec] = {}
        self._created_subtask_for: set[FeatureId] = set()
        self._raw_view_stats: dict[str, _RunningVectorStats] = {}
        self._episode_encoder_loss_sum = 0.0
        self._episode_encoder_loss_count = 0

        for field_plan in perception_plan.feature_groups:
            self._features[field_plan.field_id] = FeatureSpec(
                feature_id=field_plan.field_id,
                name=field_plan.name,
                description=field_plan.description,
                metadata={
                    "source_channel": field_plan.source_channel,
                    "selector_names": list(field_plan.selector_names),
                    "selector_indices": list(field_plan.selector_indices),
                },
            )

    def reset(self) -> None:
        self._state = None

    def update(
        self,
        observation: Any,
        reward: float,
        last_action: Any | None,
    ) -> ExampleSubjectiveState:
        self._update_count += 1
        agent_observation = normalize_observation(
            observation,
            self._perception_plan.world_description.observation_channels,
            metadata={
                "raw_observation_type": type(observation).__name__,
            },
        )

        named_fields = {
            field_plan.field_id: extract_semantic_field(agent_observation, field_plan)
            for field_plan in self._perception_plan.feature_groups
        }

        tensor_views = {}
        for view_plan in self._perception_plan.tensor_views:
            encoder = self._encoders[view_plan.view_id]
            view_input = tensor_input_from_observation(
                agent_observation,
                self._perception_plan.world_description,
                view_plan,
            )
            if self._normalize_raw_views and view_plan.source_channel in agent_observation.raw_values:
                view_input = self._normalize_raw_tensor_view(view_plan.view_id, view_input)
            tensor_views[view_plan.view_id] = encoder.encode(view_input).detach()
            if self._train_encoder:
                loss = float(encoder.train_step(view_input))
                if loss > 0.0:
                    self._episode_encoder_loss_sum += loss
                    self._episode_encoder_loss_count += 1

        self._state = ExampleSubjectiveState(
            agent_observation=agent_observation,
            named_fields=named_fields,
            tensor_views=tensor_views,
            default_tensor_view=self._perception_plan.default_tensor_view,
            metadata={
                "reward": reward,
                "last_action": last_action,
                "planner_notes": self._perception_plan.notes,
            },
        )
        return self._state

    def current_subjective_state(self) -> ExampleSubjectiveState:
        if self._state is None:
            raise RuntimeError("update() has not been called yet")
        return self._state

    def discover_and_rank_features(
        self,
        subjective_state: ExampleSubjectiveState,
        utility_scores: Sequence[UtilityRecord],
        feature_budget: int,
    ) -> Sequence[FeatureId]:
        utility_map = {
            rec.component_id: rec.utility for rec in utility_scores
        }
        ranked = sorted(
            self._features.keys(),
            key=lambda fid: (
                fid not in self._created_subtask_for,
                utility_map.get(fid, 0.0),
            ),
            reverse=True,
        )
        return tuple(ranked[:feature_budget])

    def generate_subtasks(
        self,
        ranked_feature_ids: Sequence[FeatureId],
    ) -> Sequence[SubtaskSpec]:
        if (
            self._created_subtask_for
            and self._update_count - self._last_subtask_creation_update
            < self._subtask_creation_interval
        ):
            return ()

        created: list[SubtaskSpec] = []
        creation_limit = len(ranked_feature_ids) if not self._created_subtask_for else 1
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
            if len(created) >= creation_limit:
                break
        if created:
            self._last_subtask_creation_update = self._update_count
        return tuple(created)

    def list_features(self) -> Sequence[FeatureSpec]:
        return tuple(self._features.values())

    def remove_features(self, feature_ids: Sequence[FeatureId]) -> None:
        for fid in feature_ids:
            self._features.pop(fid, None)
            self._created_subtask_for.discard(fid)

    def training_metrics(self) -> Mapping[str, float]:
        if self._episode_encoder_loss_count == 0:
            return {}
        return {
            "perception_encoder_loss": (
                self._episode_encoder_loss_sum / self._episode_encoder_loss_count
            )
        }

    def end_episode(self) -> None:
        self._episode_encoder_loss_sum = 0.0
        self._episode_encoder_loss_count = 0

    def _normalize_raw_tensor_view(
        self,
        view_id: str,
        view_input: Any,
    ) -> np.ndarray:
        raw = np.asarray(view_input, dtype=np.float32).reshape(-1)
        stats = self._raw_view_stats.setdefault(view_id, _RunningVectorStats())
        normalized = stats.normalize(raw)
        stats.update(raw)
        return normalized
