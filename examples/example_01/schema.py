"""Shared schema and state objects for Example 01.

These types define the common language between:

- raw world discovery
- embedded world descriptions
- startup-time perception planning
- runtime perception updates
- lower RL modules that still consume tensor views
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np
import numpy.typing as npt
import torch


ObservationPathPart: TypeAlias = str | int
ObservationChannelKind: TypeAlias = Literal["raw_values", "image", "text", "sound"]
EncoderKind: TypeAlias = Literal["identity", "mlp", "cnn"]
ScalarValue: TypeAlias = bool | int | float | str
ArrayLike: TypeAlias = npt.NDArray[Any]


@dataclass(slots=True, frozen=True)
class ObservationChannelDescription:
    """One channel in the normalized observation schema."""

    channel_id: str
    kind: ObservationChannelKind
    path: tuple[ObservationPathPart, ...] = ()
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    description: str = ""
    value_names: tuple[str, ...] = ()
    encoder_hint: EncoderKind | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def input_dim(self) -> int | None:
        if self.kind == "raw_values":
            if self.value_names:
                return len(self.value_names)
            if self.shape:
                return int(np.prod(self.shape))
        if self.kind == "text" and self.shape:
            return self.shape[0]
        if self.kind == "sound" and self.shape:
            return int(np.prod(self.shape))
        return None


@dataclass(slots=True, frozen=True)
class ActionDescription:
    """Structured description of the world action space."""

    action_type: str
    action_n: int
    labels: tuple[str, ...] = ()
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SemanticFieldPlan:
    """One grouped semantic field extracted into the subjective state."""

    field_id: str
    name: str
    source_channel: str
    description: str = ""
    selector_names: tuple[str, ...] = ()
    selector_indices: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_feature_dict(self) -> dict[str, Any]:
        return {
            "id": self.field_id,
            "name": self.name,
            "description": self.description,
            "source_channel": self.source_channel,
            "selector_names": list(self.selector_names),
            "selector_indices": list(self.selector_indices),
        }


@dataclass(slots=True, frozen=True)
class TensorViewPlan:
    """How perception should produce a tensor view for lower learners."""

    view_id: str
    source_channel: str
    encoder_type: EncoderKind
    input_shape: tuple[int, ...] | None = None
    input_dim: int | None = None
    input_channels: int = 1
    latent_dim: int | None = None
    description: str = ""
    selector_names: tuple[str, ...] = ()
    selector_indices: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def resolved_latent_dim(self) -> int:
        # Identity encoders are width-preserving: downstream state_dim must
        # match the actual tensor width emitted at runtime, not an LLM hint.
        if self.encoder_type == "identity":
            return self.input_dim or self.latent_dim or 1
        if self.encoder_type == "cnn":
            return self.latent_dim or 128
        if self.latent_dim is not None:
            return self.latent_dim
        return self.input_dim or 1


@dataclass(slots=True, frozen=True)
class WorldDescription:
    """Shared world schema for both discovery and embedded worlds."""

    observation_channels: tuple[ObservationChannelDescription, ...]
    action: ActionDescription
    default_encoder_type: EncoderKind = "identity"
    feature_hints: tuple[SemanticFieldPlan, ...] = ()
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def primary_channel(self) -> ObservationChannelDescription:
        if not self.observation_channels:
            raise ValueError("WorldDescription must define at least one observation channel")
        return self.observation_channels[0]

    @property
    def obs_type(self) -> str:
        channel = self.primary_channel
        if channel.kind == "raw_values":
            if channel.shape is not None and len(channel.shape) == 1:
                return "numeric_vector"
            return "numeric_array"
        if channel.kind == "image":
            return "image"
        if channel.kind == "text":
            return "text"
        if channel.kind == "sound":
            return "sound"
        return "unknown"

    @property
    def obs_shape(self) -> tuple[int, ...] | None:
        return self.primary_channel.shape

    @property
    def obs_dtype(self) -> str:
        return self.primary_channel.dtype or "float32"

    @property
    def action_type(self) -> str:
        return self.action.action_type

    @property
    def action_n(self) -> int:
        return self.action.action_n

    @property
    def encoder_type(self) -> str:
        return self.primary_channel.encoder_hint or self.default_encoder_type

    @property
    def features(self) -> list[dict[str, Any]]:
        return [feature.to_feature_dict() for feature in self.feature_hints]

    def to_config(self) -> dict[str, Any]:
        """Compatibility export for callers that still expect a config dict."""
        return {
            "obs_type": self.obs_type,
            "obs_shape": self.obs_shape,
            "obs_dtype": self.obs_dtype,
            "action_type": self.action_type,
            "action_n": self.action_n,
            "encoder_type": self.encoder_type,
            "features": self.features,
            "observation_channels": [
                {
                    "channel_id": channel.channel_id,
                    "kind": channel.kind,
                    "path": list(channel.path),
                    "shape": channel.shape,
                    "dtype": channel.dtype,
                    "description": channel.description,
                    "value_names": list(channel.value_names),
                    "encoder_hint": channel.encoder_hint,
                }
                for channel in self.observation_channels
            ],
            "action": {
                "action_type": self.action.action_type,
                "action_n": self.action.action_n,
                "labels": list(self.action.labels),
                "description": self.action.description,
            },
            "default_encoder_type": self.default_encoder_type,
            "feature_hints": self.features,
            "notes": self.notes,
        }


@dataclass(slots=True, frozen=True)
class AgentObservation:
    """Normalized multimodal observation consumed by perception."""

    raw_values: Mapping[str, Mapping[str, ScalarValue]] = field(default_factory=dict)
    images: Mapping[str, ArrayLike] = field(default_factory=dict)
    text: Mapping[str, str] = field(default_factory=dict)
    sounds: Mapping[str, ArrayLike] = field(default_factory=dict)
    channel_descriptions: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ExampleSubjectiveState:
    """Structured subjective state for Example 01."""

    agent_observation: AgentObservation
    named_fields: Mapping[str, Any]
    tensor_views: Mapping[str, torch.Tensor]
    default_tensor_view: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def tensor_view(self, view_name: str | None = None) -> torch.Tensor:
        selected = view_name or self.default_tensor_view
        try:
            return self.tensor_views[selected]
        except KeyError as exc:
            raise KeyError(
                f"Tensor view {selected!r} not found. Available views: "
                f"{sorted(self.tensor_views)}"
            ) from exc


@dataclass(slots=True, frozen=True)
class PerceptionPlan:
    """Startup-time plan describing normalization, grouping, and tensor views."""

    world_description: WorldDescription
    feature_groups: tuple[SemanticFieldPlan, ...]
    tensor_views: tuple[TensorViewPlan, ...]
    default_tensor_view: str
    notes: str = ""
    llm_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def view(self, view_name: str | None = None) -> TensorViewPlan:
        selected = view_name or self.default_tensor_view
        for view in self.tensor_views:
            if view.view_id == selected:
                return view
        raise KeyError(
            f"Tensor view {selected!r} not found. Available views: "
            f"{[view.view_id for view in self.tensor_views]}"
        )

    def to_config(self) -> dict[str, Any]:
        config = self.world_description.to_config()
        config["features"] = [feature.to_feature_dict() for feature in self.feature_groups]
        config["encoder_type"] = self.view().encoder_type
        config["default_tensor_view"] = self.default_tensor_view
        config["tensor_views"] = [
            {
                "view_id": view.view_id,
                "source_channel": view.source_channel,
                "encoder_type": view.encoder_type,
                "input_shape": view.input_shape,
                "input_dim": view.input_dim,
                "input_channels": view.input_channels,
                "latent_dim": view.latent_dim,
                "description": view.description,
                "selector_names": list(view.selector_names),
                "selector_indices": list(view.selector_indices),
            }
            for view in self.tensor_views
        ]
        config["notes"] = self.notes
        config["llm_used"] = self.llm_used
        return config


@dataclass(slots=True, frozen=True)
class ExampleAgentSpec:
    """Typed startup artifact used to build the Example 01 agent."""

    perception_plan: PerceptionPlan
    source: str = "legacy"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_config(self) -> dict[str, Any]:
        config = self.perception_plan.to_config()
        config["source"] = self.source
        return config


@dataclass(slots=True, frozen=True)
class StateTensorAdapter:
    """Extract one or more tensor views from the structured subjective state.

    When ``view_names`` is provided, all listed views are concatenated along
    the last dimension.  Otherwise a single ``view_name`` is extracted (the
    original behaviour).
    """

    view_name: str | None = None
    view_names: tuple[str, ...] | None = None

    def tensor(self, state: ExampleSubjectiveState) -> torch.Tensor:
        if self.view_names is not None:
            # For real states every view exists; for synthetic planning
            # states the concatenated tensor lives under a single key.
            if all(n in state.tensor_views for n in self.view_names):
                return torch.cat(
                    [state.tensor_view(n) for n in self.view_names], dim=-1
                )
            return state.tensor_view()
        return state.tensor_view(self.view_name)

    def state_dim(self, plan: PerceptionPlan) -> int:
        if self.view_names is not None:
            return sum(
                plan.view(n).resolved_latent_dim() for n in self.view_names
            )
        return plan.view(self.view_name).resolved_latent_dim()


def subjective_state_from_tensor(
    tensor: torch.Tensor,
    *,
    view_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> ExampleSubjectiveState:
    """Wrap a tensor into a minimal structured subjective state."""
    return ExampleSubjectiveState(
        agent_observation=AgentObservation(),
        named_fields={},
        tensor_views={view_name: tensor},
        default_tensor_view=view_name,
        metadata=dict(metadata or {}),
    )
