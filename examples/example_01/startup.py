"""Startup-time helpers for discovery, normalization, and planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .schema import (
    ActionDescription,
    AgentObservation,
    ExampleAgentSpec,
    ObservationChannelDescription,
    ObservationPathPart,
    PerceptionPlan,
    ScalarValue,
    SemanticFieldPlan,
    TensorViewPlan,
    WorldDescription,
)


def infer_world_description(
    observation_sample: Any,
    action: ActionDescription,
    *,
    notes: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> WorldDescription:
    """Infer a structured world description from one raw observation sample."""
    channels = infer_observation_channels(observation_sample)
    default_encoder = channels[0].encoder_hint or "identity"
    return WorldDescription(
        observation_channels=channels,
        action=action,
        default_encoder_type=default_encoder,
        feature_hints=tuple(_generic_feature_hints(channels)),
        notes=notes,
        metadata=dict(metadata or {}),
    )


def infer_observation_channels(
    observation: Any,
    *,
    channel_id: str = "main",
    path: tuple[ObservationPathPart, ...] = (),
) -> tuple[ObservationChannelDescription, ...]:
    """Mechanically classify a raw observation into typed observation channels."""
    if isinstance(observation, np.ndarray):
        return (_channel_from_array(observation, channel_id=channel_id, path=path),)

    if _is_numeric_scalar(observation):
        return (
            ObservationChannelDescription(
                channel_id=channel_id,
                kind="raw_values",
                path=path,
                shape=(1,),
                dtype=type(observation).__name__,
                description=f"Numeric scalar extracted from {channel_id!r}.",
                value_names=(channel_id,),
                encoder_hint="identity",
            ),
        )

    if isinstance(observation, str):
        return (
            ObservationChannelDescription(
                channel_id=channel_id,
                kind="text",
                path=path,
                shape=(len(observation),),
                dtype="str",
                description=f"Text observation extracted from {channel_id!r}.",
                encoder_hint="mlp",
            ),
        )

    if isinstance(observation, Mapping):
        channels: list[ObservationChannelDescription] = []
        for key, value in observation.items():
            key_str = str(key)
            channels.extend(
                infer_observation_channels(
                    value,
                    channel_id=key_str,
                    path=path + (key_str,),
                )
            )
        return tuple(channels)

    if isinstance(observation, (list, tuple)):
        if observation and all(_is_numeric_scalar(item) for item in observation):
            value_names = tuple(f"value_{index}" for index in range(len(observation)))
            return (
                ObservationChannelDescription(
                    channel_id=channel_id,
                    kind="raw_values",
                    path=path,
                    shape=(len(observation),),
                    dtype="float32",
                    description=f"Numeric vector extracted from {channel_id!r}.",
                    value_names=value_names,
                    encoder_hint="identity" if len(value_names) <= 32 else "mlp",
                ),
            )

        channels = []
        for index, value in enumerate(observation):
            channels.extend(
                infer_observation_channels(
                    value,
                    channel_id=f"item_{index}",
                    path=path + (index,),
                )
            )
        return tuple(channels)

    return (
        ObservationChannelDescription(
            channel_id=channel_id,
            kind="text",
            path=path,
            shape=(len(repr(observation)),),
            dtype=type(observation).__name__,
            description=f"Fallback textual representation for {channel_id!r}.",
            encoder_hint="mlp",
        ),
    )


def build_heuristic_perception_plan(
    description: WorldDescription,
    *,
    notes: str = "",
    llm_used: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> PerceptionPlan:
    """Build a deterministic startup-time perception plan."""
    feature_groups = description.feature_hints or tuple(
        _generic_feature_hints(description.observation_channels)
    )
    tensor_views = tuple(_default_tensor_views(description))
    if not tensor_views:
        raise ValueError("Perception plan requires at least one tensor view")
    return PerceptionPlan(
        world_description=description,
        feature_groups=feature_groups,
        tensor_views=tensor_views,
        default_tensor_view=tensor_views[0].view_id,
        notes=notes or description.notes,
        llm_used=llm_used,
        metadata=dict(metadata or {}),
    )


def build_agent_spec(
    description: WorldDescription,
    *,
    source: str,
    notes: str = "",
    llm_used: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> ExampleAgentSpec:
    """Construct the typed startup artifact consumed by `build_agent`."""
    plan = build_heuristic_perception_plan(
        description,
        notes=notes,
        llm_used=llm_used,
        metadata=metadata,
    )
    return ExampleAgentSpec(
        perception_plan=plan,
        source=source,
        metadata=dict(metadata or {}),
    )


def normalize_observation(
    observation: Any,
    channels: Sequence[ObservationChannelDescription],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> AgentObservation:
    """Turn a raw observation into the normalized `AgentObservation`."""
    raw_values: dict[str, Mapping[str, ScalarValue]] = {}
    images: dict[str, np.ndarray] = {}
    text: dict[str, str] = {}
    sounds: dict[str, np.ndarray] = {}
    descriptions: dict[str, str] = {}

    for channel in channels:
        payload = _extract_by_path(observation, channel.path)
        descriptions[channel.channel_id] = channel.description

        if channel.kind == "raw_values":
            raw_values[channel.channel_id] = _normalize_raw_values(payload, channel)
        elif channel.kind == "image":
            images[channel.channel_id] = np.asarray(payload).copy()
        elif channel.kind == "sound":
            sounds[channel.channel_id] = np.asarray(payload).copy()
        elif channel.kind == "text":
            text[channel.channel_id] = str(payload)

    return AgentObservation(
        raw_values=raw_values,
        images=images,
        text=text,
        sounds=sounds,
        channel_descriptions=descriptions,
        metadata=dict(metadata or {}),
    )


def extract_semantic_field(
    observation: AgentObservation,
    field_plan: SemanticFieldPlan,
) -> Any:
    """Extract one grouped semantic field from the normalized observation."""
    if field_plan.source_channel in observation.raw_values:
        channel_values = dict(observation.raw_values[field_plan.source_channel])
        if field_plan.selector_names:
            return {
                name: channel_values[name]
                for name in field_plan.selector_names
                if name in channel_values
            }
        if field_plan.selector_indices:
            items = list(channel_values.items())
            return {
                name: value
                for index, (name, value) in enumerate(items)
                if index in field_plan.selector_indices
            }
        return channel_values

    if field_plan.source_channel in observation.images:
        return observation.images[field_plan.source_channel]
    if field_plan.source_channel in observation.sounds:
        return observation.sounds[field_plan.source_channel]
    if field_plan.source_channel in observation.text:
        return observation.text[field_plan.source_channel]
    raise KeyError(f"Unknown source channel {field_plan.source_channel!r}")


def tensor_input_from_observation(
    observation: AgentObservation,
    description: WorldDescription,
    view_plan: TensorViewPlan,
) -> np.ndarray:
    """Build the encoder input for one tensor view."""
    channel = _channel_lookup(description, view_plan.source_channel)
    if channel.kind == "raw_values":
        values = observation.raw_values[view_plan.source_channel]
        ordered_names = channel.value_names or tuple(values.keys())
        return np.asarray(
            [_scalar_to_float(values[name]) for name in ordered_names],
            dtype=np.float32,
        )

    if channel.kind == "image":
        return np.asarray(observation.images[view_plan.source_channel], dtype=np.float32)

    if channel.kind == "sound":
        return np.asarray(observation.sounds[view_plan.source_channel], dtype=np.float32)

    text_value = observation.text[view_plan.source_channel]
    encoded = np.frombuffer(text_value.encode("utf-8"), dtype=np.uint8).astype(np.float32)
    encoded = encoded / 255.0
    input_dim = view_plan.input_dim or max(len(encoded), 1)
    if len(encoded) >= input_dim:
        return encoded[:input_dim]
    padded = np.zeros(input_dim, dtype=np.float32)
    padded[: len(encoded)] = encoded
    return padded


def serialize_observation_sample(observation: Any) -> str:
    """Convert an observation to a compact LLM-facing representation."""
    if isinstance(observation, np.ndarray):
        return (
            f"ndarray(shape={observation.shape}, dtype={observation.dtype}, "
            f"values={observation.tolist()})"
        )
    if isinstance(observation, Mapping):
        return repr({key: serialize_observation_sample(value) for key, value in observation.items()})
    return repr(observation)


def _channel_from_array(
    observation: np.ndarray,
    *,
    channel_id: str,
    path: tuple[ObservationPathPart, ...],
) -> ObservationChannelDescription:
    array = np.asarray(observation)
    if array.ndim in (2, 3):
        return ObservationChannelDescription(
            channel_id=channel_id,
            kind="image",
            path=path,
            shape=tuple(int(dim) for dim in array.shape),
            dtype=str(array.dtype),
            description=f"Image-like array extracted from {channel_id!r}.",
            encoder_hint="cnn",
        )

    flat_size = int(array.size)
    value_names = tuple(f"value_{index}" for index in range(flat_size))
    return ObservationChannelDescription(
        channel_id=channel_id,
        kind="raw_values",
        path=path,
        shape=(flat_size,),
        dtype=str(array.dtype),
        description=f"Numeric array extracted from {channel_id!r}.",
        value_names=value_names,
        encoder_hint="identity" if flat_size <= 32 else "mlp",
    )


def _default_tensor_views(
    description: WorldDescription,
) -> list[TensorViewPlan]:
    views: list[TensorViewPlan] = []
    for channel in description.observation_channels:
        if channel.kind == "raw_values":
            input_dim = channel.input_dim() or 1
            encoder_type = channel.encoder_hint or (
                "identity" if input_dim <= 32 else "mlp"
            )
            latent_dim = input_dim if encoder_type == "identity" else max(input_dim * 4, 32)
            views.append(
                TensorViewPlan(
                    view_id=channel.channel_id,
                    source_channel=channel.channel_id,
                    encoder_type=encoder_type,
                    input_shape=channel.shape,
                    input_dim=input_dim,
                    input_channels=1,
                    latent_dim=latent_dim,
                    description=channel.description,
                )
            )
            continue

        if channel.kind == "image":
            input_shape = channel.shape
            input_channels = (
                int(input_shape[-1]) if input_shape is not None and len(input_shape) >= 3 else 1
            )
            views.append(
                TensorViewPlan(
                    view_id=channel.channel_id,
                    source_channel=channel.channel_id,
                    encoder_type=channel.encoder_hint or "cnn",
                    input_shape=input_shape,
                    input_dim=channel.input_dim(),
                    input_channels=input_channels,
                    latent_dim=128,
                    description=channel.description,
                )
            )
            continue

        input_dim = channel.input_dim() or 64
        encoder_type = channel.encoder_hint or (
            "identity" if input_dim <= 32 else "mlp"
        )
        latent_dim = input_dim if encoder_type == "identity" else max(input_dim * 2, 32)
        views.append(
            TensorViewPlan(
                view_id=channel.channel_id,
                source_channel=channel.channel_id,
                encoder_type=encoder_type,
                input_shape=channel.shape,
                input_dim=input_dim,
                input_channels=1,
                latent_dim=latent_dim,
                description=channel.description,
            )
        )
    return views


def _generic_feature_hints(
    channels: Sequence[ObservationChannelDescription],
) -> list[SemanticFieldPlan]:
    hints: list[SemanticFieldPlan] = []
    for channel in channels:
        if channel.kind == "raw_values":
            hints.extend(_raw_value_groups(channel))
        elif channel.kind == "image":
            hints.append(
                SemanticFieldPlan(
                    field_id=f"{channel.channel_id}_visual_scene",
                    name=f"{channel.channel_id.replace('_', ' ').title()} visual scene",
                    source_channel=channel.channel_id,
                    description=channel.description or "Image-based observation channel.",
                )
            )
        elif channel.kind == "sound":
            hints.append(
                SemanticFieldPlan(
                    field_id=f"{channel.channel_id}_audio_pattern",
                    name=f"{channel.channel_id.replace('_', ' ').title()} audio pattern",
                    source_channel=channel.channel_id,
                    description=channel.description or "Audio observation channel.",
                )
            )
        elif channel.kind == "text":
            hints.append(
                SemanticFieldPlan(
                    field_id=f"{channel.channel_id}_text_context",
                    name=f"{channel.channel_id.replace('_', ' ').title()} text context",
                    source_channel=channel.channel_id,
                    description=channel.description or "Text observation channel.",
                )
            )
    return hints


def _raw_value_groups(channel: ObservationChannelDescription) -> list[SemanticFieldPlan]:
    names = channel.value_names or tuple(
        f"value_{index}" for index in range(channel.input_dim() or 1)
    )
    grouped_by_prefix: dict[str, list[str]] = {}
    for name in names:
        prefix = name.split("_", 1)[0]
        grouped_by_prefix.setdefault(prefix, []).append(name)

    if len(grouped_by_prefix) >= 2 and all(
        len(grouped_by_prefix[prefix]) < len(names) for prefix in grouped_by_prefix
    ):
        groups: list[SemanticFieldPlan] = []
        for prefix, group_names in grouped_by_prefix.items():
            groups.append(
                SemanticFieldPlan(
                    field_id=f"{channel.channel_id}_{prefix}",
                    name=f"{prefix.replace('_', ' ').title()} state",
                    source_channel=channel.channel_id,
                    description=f"Grouped raw values sharing the prefix {prefix!r}.",
                    selector_names=tuple(group_names),
                )
            )
        return groups

    if len(names) >= 4:
        midpoint = len(names) // 2
        return [
            SemanticFieldPlan(
                field_id=f"{channel.channel_id}_group_0",
                name=f"{channel.channel_id.replace('_', ' ').title()} group 1",
                source_channel=channel.channel_id,
                description="First grouped slice of the raw-value channel.",
                selector_indices=tuple(range(0, midpoint)),
            ),
            SemanticFieldPlan(
                field_id=f"{channel.channel_id}_group_1",
                name=f"{channel.channel_id.replace('_', ' ').title()} group 2",
                source_channel=channel.channel_id,
                description="Second grouped slice of the raw-value channel.",
                selector_indices=tuple(range(midpoint, len(names))),
            ),
        ]

    return [
        SemanticFieldPlan(
            field_id=f"{channel.channel_id}_values",
            name=f"{channel.channel_id.replace('_', ' ').title()} values",
            source_channel=channel.channel_id,
            description=channel.description or "Grouped raw values.",
        )
    ]


def _normalize_raw_values(
    payload: Any,
    channel: ObservationChannelDescription,
) -> Mapping[str, ScalarValue]:
    if isinstance(payload, np.ndarray):
        values = np.asarray(payload).reshape(-1).tolist()
    elif isinstance(payload, (list, tuple)):
        values = list(payload)
    else:
        values = [payload]

    names = channel.value_names or tuple(f"value_{index}" for index in range(len(values)))
    normalized: dict[str, ScalarValue] = {}
    for name, value in zip(names, values, strict=False):
        normalized[name] = _scalar_value(value)
    return normalized


def _extract_by_path(observation: Any, path: Sequence[ObservationPathPart]) -> Any:
    current = observation
    for part in path:
        if isinstance(part, int):
            current = current[part]
        else:
            current = current[part]
    return current


def _channel_lookup(
    description: WorldDescription,
    channel_id: str,
) -> ObservationChannelDescription:
    for channel in description.observation_channels:
        if channel.channel_id == channel_id:
            return channel
    raise KeyError(f"Unknown channel {channel_id!r}")


def _scalar_value(value: Any) -> ScalarValue:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _scalar_to_float(value: ScalarValue) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return float(len(value))


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, np.generic)) and not isinstance(value, str)
