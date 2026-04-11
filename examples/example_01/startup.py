"""Startup-time helpers for discovery, normalization, and planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
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


_KNOWN_RAW_VALUE_SCALES: dict[str, dict[str, dict[str, float]]] = {
    "CartPole-v1": {
        "main": {
            "cart_position": 2.4,
            "cart_velocity": 3.0,
            "pole_angle": 0.2095,
            "pole_angular_velocity": 3.5,
        }
    },
    "Acrobot-v1": {
        "main": {
            "cos_theta1": 1.0,
            "sin_theta1": 1.0,
            "cos_theta2": 1.0,
            "sin_theta2": 1.0,
            "angular_velocity_1": float(4.0 * np.pi),
            "angular_velocity_2": float(9.0 * np.pi),
        }
    },
}


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
    return stabilize_perception_plan(
        PerceptionPlan(
            world_description=description,
            feature_groups=feature_groups,
            tensor_views=tensor_views,
            default_tensor_view=tensor_views[0].view_id,
            notes=notes or description.notes,
            llm_used=llm_used,
            metadata=dict(metadata or {}),
        )
    )


def stabilize_perception_plan(plan: PerceptionPlan) -> PerceptionPlan:
    """Normalize a startup plan into a safe control-state configuration.

    For raw-value worlds, downstream control should always have access to a
    full-state tensor view, even if an LLM proposes only partial slices such as
    ``cart_motion_view``.
    """
    tensor_views = list(plan.tensor_views)
    changed = False
    for channel in plan.world_description.observation_channels:
        if channel.kind != "raw_values":
            continue
        if _has_full_raw_tensor_view(channel, tensor_views):
            continue
        tensor_views.append(
            _full_raw_tensor_view(
                plan.world_description,
                channel,
                existing_view_ids={view.view_id for view in tensor_views},
            )
        )
        changed = True

    feature_groups = plan.feature_groups
    if _is_small_raw_control_world(plan.world_description):
        preferred_feature_groups = plan.world_description.feature_hints or tuple(
            _generic_feature_hints(plan.world_description.observation_channels)
        )
        if preferred_feature_groups and preferred_feature_groups != feature_groups:
            feature_groups = preferred_feature_groups
            changed = True

    default_tensor_view = _preferred_default_tensor_view(
        plan.world_description,
        tensor_views,
        requested_default=plan.default_tensor_view,
    )
    if (
        not changed
        and default_tensor_view == plan.default_tensor_view
        and feature_groups == plan.feature_groups
    ):
        return plan
    return replace(
        plan,
        feature_groups=tuple(feature_groups),
        tensor_views=tuple(tensor_views),
        default_tensor_view=default_tensor_view,
    )


def _has_full_raw_tensor_view(
    channel: ObservationChannelDescription,
    tensor_views: Sequence[TensorViewPlan],
) -> bool:
    channel_names = channel.value_names or tuple(
        f"value_{index}" for index in range(channel.input_dim() or 0)
    )
    channel_dim = len(channel_names) or (channel.input_dim() or 0)
    for view in tensor_views:
        if view.source_channel != channel.channel_id:
            continue
        if (
            view.encoder_type == "identity"
            and _resolved_raw_selector_names(view, channel_names) == channel_names
        ):
            return True
        if (
            view.encoder_type == "identity"
            and not view.selector_names
            and not view.selector_indices
            and int(view.input_dim or 0) >= channel_dim
        ):
            return True
    return False


def _full_raw_tensor_view(
    description: WorldDescription,
    channel: ObservationChannelDescription,
    *,
    existing_view_ids: set[str],
) -> TensorViewPlan:
    input_dim = channel.input_dim() or 1
    encoder_type = channel.encoder_hint or ("identity" if input_dim <= 32 else "mlp")
    latent_dim = input_dim if encoder_type == "identity" else max(input_dim * 4, 32)
    preferred_view_id = (
        "full_state_view"
        if description.primary_channel.channel_id == channel.channel_id
        else f"{channel.channel_id}_full_state_view"
    )
    view_id = preferred_view_id
    suffix = 1
    while view_id in existing_view_ids:
        view_id = f"{preferred_view_id}_{suffix}"
        suffix += 1
    return TensorViewPlan(
        view_id=view_id,
        source_channel=channel.channel_id,
        encoder_type=encoder_type,
        input_shape=channel.shape,
        input_dim=input_dim,
        input_channels=1,
        latent_dim=latent_dim,
        description=channel.description or f"Full raw-value state for {channel.channel_id!r}.",
        selector_names=channel.value_names,
    )


def _preferred_default_tensor_view(
    description: WorldDescription,
    tensor_views: Sequence[TensorViewPlan],
    *,
    requested_default: str,
) -> str:
    if not tensor_views:
        raise ValueError("Perception plan requires at least one tensor view")

    views_by_id = {view.view_id: view for view in tensor_views}
    primary_channel = description.primary_channel
    if primary_channel.kind == "raw_values":
        primary_views = [
            view for view in tensor_views if view.source_channel == primary_channel.channel_id
        ]
        if primary_views:
            return _richest_raw_tensor_view(primary_channel, primary_views).view_id

    # For image channels, prefer a CNN (or MLP) view over an identity view.
    # An identity encoder on raw pixels produces an unusably large flat state.
    if primary_channel.kind == "image":
        cnn_views = [
            view for view in tensor_views
            if view.source_channel == primary_channel.channel_id
            and view.encoder_type in ("cnn", "mlp")
        ]
        if cnn_views:
            return cnn_views[0].view_id

    if requested_default in views_by_id:
        return requested_default
    return tensor_views[0].view_id


def _richest_raw_tensor_view(
    channel: ObservationChannelDescription,
    tensor_views: Sequence[TensorViewPlan],
) -> TensorViewPlan:
    channel_names = channel.value_names or tuple(
        f"value_{index}" for index in range(channel.input_dim() or 0)
    )
    return max(
        tensor_views,
        key=lambda view: (
            int(_resolved_raw_selector_names(view, channel_names) == channel_names),
            int(view.encoder_type == "identity"),
            int(view.input_dim or 0),
            int(not view.selector_names and not view.selector_indices),
            len(view.selector_names),
            len(view.selector_indices),
        ),
    )


def _resolved_raw_selector_names(
    view: TensorViewPlan,
    channel_names: tuple[str, ...],
) -> tuple[str, ...]:
    if view.selector_names:
        return tuple(name for name in view.selector_names if name in channel_names)
    if view.selector_indices:
        return tuple(
            channel_names[index]
            for index in view.selector_indices
            if 0 <= index < len(channel_names)
        )
    return channel_names


def _is_small_raw_control_world(description: WorldDescription) -> bool:
    channels = description.observation_channels
    if not channels or any(channel.kind != "raw_values" for channel in channels):
        return False
    total_dim = sum(channel.input_dim() or 0 for channel in channels)
    return total_dim <= 32


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
        if view_plan.selector_names:
            ordered_names = tuple(
                name for name in view_plan.selector_names if name in values
            )
        elif view_plan.selector_indices:
            ordered_names = tuple(
                name
                for index, name in enumerate(ordered_names)
                if index in view_plan.selector_indices
            )
        if not ordered_names:
            ordered_names = channel.value_names or tuple(values.keys())
        raw_array = np.asarray(
            [_scalar_to_float(values[name]) for name in ordered_names],
            dtype=np.float32,
        )
        return _normalize_raw_value_array(
            raw_array,
            description,
            channel,
            ordered_names,
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


def _normalize_raw_value_array(
    values: np.ndarray,
    description: WorldDescription,
    channel: ObservationChannelDescription,
    ordered_names: Sequence[str],
) -> np.ndarray:
    scale_map = _raw_value_scale_map(description, channel)
    if not scale_map:
        return values
    scales = np.asarray(
        [
            abs(float(scale_map.get(name, 1.0)))
            for name in ordered_names
        ],
        dtype=np.float32,
    )
    scales = np.where(scales > 1e-6, scales, 1.0).astype(np.float32)
    normalized = values / scales
    return np.clip(normalized, -5.0, 5.0).astype(np.float32)


def _raw_value_scale_map(
    description: WorldDescription,
    channel: ObservationChannelDescription,
) -> dict[str, float]:
    channel_scale_raw = channel.metadata.get("normalization_scales")
    if isinstance(channel_scale_raw, Mapping):
        return {
            str(name): float(scale)
            for name, scale in channel_scale_raw.items()
            if isinstance(scale, (int, float))
        }

    env_id = str(description.metadata.get("env_id", "")).strip()
    if not env_id:
        return {}
    return dict(
        _KNOWN_RAW_VALUE_SCALES.get(env_id, {}).get(channel.channel_id, {})
    )


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, np.generic)) and not isinstance(value, str)
