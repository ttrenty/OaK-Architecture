"""LLM-planned adaptive perception for OaK Example 01."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
    ) -> None:
        self._perception_plan = perception_plan
        self._encoders = dict(encoders)
        self._encoder = self._encoders[perception_plan.default_tensor_view]
        self._train_encoder = train_encoder
        self._state: ExampleSubjectiveState | None = None

        self._features: dict[FeatureId, FeatureSpec] = {}
        self._created_subtask_for: set[FeatureId] = set()

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
            tensor_views[view_plan.view_id] = encoder.encode(view_input).detach()
            if self._train_encoder:
                encoder.train_step(view_input)

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
