"""Typed startup-spec -> build -> train flow for Example 01."""

from __future__ import annotations

from dataclasses import replace
import os
from typing import Any, Callable, Protocol, cast

import torch

from oak.agent import OaKAgent
from oak.interfaces import World
from oak.types import EpisodeTrace

from .discovery import DiscoveryManager
from .encoders import create_encoder
from .llm import analyze_world
from .perception import AdaptivePerception
from .reactive_policy import OptionCriticPolicy
from .schema import (
    ActionDescription,
    ExampleAgentSpec,
    ObservationChannelDescription,
    PerceptionPlan,
    SemanticFieldPlan,
    StateTensorAdapter,
    TensorViewPlan,
    WorldDescription,
)
from .startup import build_heuristic_perception_plan, stabilize_perception_plan
from .transition_model import DynaTransitionModel
from .value_function import OptionValueFunction


class WorldWithDescription(World[Any, Any, Any], Protocol):
    """World protocol with an embedded structured description."""

    description: WorldDescription


def _recommended_policy_hyperparameters(
    perception_plan: PerceptionPlan,
) -> tuple[float, int, int | None]:
    """Return environment-aware exploration defaults for Example 01."""
    env_id = str(perception_plan.world_description.metadata.get("env_id", ""))
    channel_kinds = {
        channel.kind for channel in perception_plan.world_description.observation_channels
    }

    if env_id == "CartPole-v1":
        return 0.02, 5_000, 80
    if env_id == "Acrobot-v1":
        return 0.05, 5_000, 450
    if channel_kinds == {"raw_values"}:
        return 0.02, 5_000, 250
    return 0.01, 20_000, None


def _recommended_training_hyperparameters(
    perception_plan: PerceptionPlan,
) -> tuple[int, int, int]:
    """Return environment-aware training defaults for Example 01."""
    env_id = str(perception_plan.world_description.metadata.get("env_id", ""))
    channel_kinds = {
        channel.kind for channel in perception_plan.world_description.observation_channels
    }

    if env_id == "CartPole-v1":
        return 1, 25_000, 128
    if env_id == "Acrobot-v1":
        return 1, 50_000, 128
    if channel_kinds == {"raw_values"}:
        return 1, 30_000, 128
    return 1, 10_000, 500


def _prefers_cpu_training_device(perception_plan: PerceptionPlan) -> bool:
    """Prefer CPU for tiny raw-state control tasks.

    Small identity-view classic-control runs spend much more time on
    host<->device transfer and CUDA launch overhead than on actual math.
    For those cases CPU is materially faster in practice.
    """

    channel_kinds = {
        channel.kind for channel in perception_plan.world_description.observation_channels
    }
    if channel_kinds != {"raw_values"}:
        return False
    if not perception_plan.tensor_views:
        return False
    if any(view.encoder_type != "identity" for view in perception_plan.tensor_views):
        return False
    default_view = perception_plan.view()
    return default_view.resolved_latent_dim() <= 16


def _resolve_training_device(
    perception_plan: PerceptionPlan,
    explicit_device: torch.device | None = None,
) -> tuple[torch.device, str]:
    """Resolve the execution device for Example 01 training."""

    if explicit_device is not None:
        return explicit_device, "explicit"

    env_override = os.environ.get("OAK_EXAMPLE_DEVICE", "").strip()
    if env_override and env_override.lower() != "auto":
        requested = torch.device(env_override)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu"), f"env override {env_override!r} unavailable"
        return requested, f"env override {env_override!r}"

    if _prefers_cpu_training_device(perception_plan):
        return torch.device("cpu"), "auto-selected for small raw-value control world"

    if torch.cuda.is_available():
        return torch.device("cuda"), "auto-selected CUDA"
    return torch.device("cpu"), "auto-selected CPU"


def build_agent(
    config: dict[str, Any] | ExampleAgentSpec,
    *,
    train_encoder: bool = False,
    planning_budget: int = 1,
    planning_warmup_steps: int | None = None,
    feature_budget: int | None = None,
    device: torch.device | None = None,
) -> OaKAgent:
    """Build a fully wired OaK agent from a typed startup spec or legacy config."""
    spec = _coerce_agent_spec(config)
    perception_plan = stabilize_perception_plan(spec.perception_plan)
    # If the legacy config explicitly overrode the encoder, restore its default
    # view after stabilization (stabilize may have switched to identity).
    explicit_default = perception_plan.metadata.get("explicit_default_tensor_view")
    if explicit_default and explicit_default != perception_plan.default_tensor_view:
        view_ids = {v.view_id for v in perception_plan.tensor_views}
        if explicit_default in view_ids:
            perception_plan = replace(perception_plan, default_tensor_view=explicit_default)
    if perception_plan is not spec.perception_plan:
        spec = replace(spec, perception_plan=perception_plan)
    device, _ = _resolve_training_device(perception_plan, device)

    channel_kinds = {
        ch.kind for ch in perception_plan.world_description.observation_channels
    }
    (
        recommended_feature_budget,
        recommended_planning_warmup_steps,
        subtask_creation_interval,
    ) = _recommended_training_hyperparameters(perception_plan)
    effective_feature_budget = feature_budget or recommended_feature_budget
    effective_planning_warmup_steps = (
        planning_warmup_steps
        if planning_warmup_steps is not None
        else recommended_planning_warmup_steps
    )

    if len(channel_kinds) > 1 and len(perception_plan.tensor_views) > 1:
        state_adapter = StateTensorAdapter(
            view_names=tuple(v.view_id for v in perception_plan.tensor_views)
        )
    else:
        state_adapter = StateTensorAdapter(perception_plan.default_tensor_view)

    encoders = {}
    for view in perception_plan.tensor_views:
        input_dim = view.input_dim
        if input_dim is None and view.input_shape is not None:
            input_dim = int(torch.tensor(view.input_shape).prod().item())
        input_dim = input_dim or 1
        encoders[view.view_id] = create_encoder(
            view.encoder_type,
            input_dim,
            latent_dim=view.resolved_latent_dim(),
            trainable=train_encoder,
            input_channels=view.input_channels,
        )

    state_dim = state_adapter.state_dim(perception_plan)
    action_n = perception_plan.world_description.action_n

    value_function = OptionValueFunction(
        state_dim=state_dim,
        max_options=8,
        device=device,
        state_adapter=state_adapter,
    )

    perception = AdaptivePerception(
        perception_plan=perception_plan,
        encoders=encoders,
        train_encoder=train_encoder,
        subtask_creation_interval=subtask_creation_interval,
    )

    transition_model = DynaTransitionModel(
        state_dim=state_dim,
        num_actions=action_n,
        planning_warmup_steps=effective_planning_warmup_steps,
        device=device,
        state_adapter=state_adapter,
    )

    epsilon_end, epsilon_decay_steps, epsilon_decay_episodes = (
        _recommended_policy_hyperparameters(perception_plan)
    )

    reactive_policy = OptionCriticPolicy(
        value_function=value_function,
        num_actions=action_n,
        state_dim=state_dim,
        device=device,
        state_adapter=state_adapter,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        epsilon_decay_episodes=epsilon_decay_episodes,
    )

    return OaKAgent(
        perception=perception,
        transition_model=transition_model,
        value_function=value_function,
        reactive_policy=reactive_policy,
        planning_budget=planning_budget,
        feature_budget=effective_feature_budget,
        option_stop_threshold=0.5,
    )


def _spec_from_discovery(
    world: World[Any, Any, Any],
    *,
    ollama_model: str,
    verbose: bool,
) -> ExampleAgentSpec:
    """Run trial-and-error discovery, then build a startup perception plan."""
    if verbose:
        print("=" * 60)
        print("Phase 1: STARTUP, probing raw world and planning perception")
        print("=" * 60)

    discovery = DiscoveryManager()
    while not discovery.is_complete():
        discovery.probe_step(world)

    description = discovery.get_world_description()
    perception_plan = _startup_plan(
        description,
        discovery.observation_samples,
        ollama_model=ollama_model,
    )

    if verbose:
        _print_startup_summary("discovery", description, perception_plan)

    return ExampleAgentSpec(
        perception_plan=perception_plan,
        source="discovery",
        metadata={"observation_samples": len(discovery.observation_samples)},
    )


def _spec_from_description(
    world: WorldWithDescription,
    *,
    ollama_model: str,
    verbose: bool,
) -> ExampleAgentSpec:
    """Read the embedded description, then build a startup perception plan."""
    perception_plan = _startup_plan(
        world.description,
        [],
        ollama_model=ollama_model,
    )

    if verbose:
        print("=" * 60)
        print("Phase 1: STARTUP, reading world description and planning perception")
        print("=" * 60)
        _print_startup_summary("embedded", world.description, perception_plan)

    return ExampleAgentSpec(
        perception_plan=perception_plan,
        source="embedded",
        metadata={"embedded": True},
    )


def run_training(
    world: World[Any, Any, Any],
    *,
    num_episodes: int = 500,
    average_window: int = 100,
    solved_threshold: float | None = None,
    ollama_model: str = "qwen3.5:9b",
    train_encoder: bool = False,
    planning_budget: int = 1,
    planning_warmup_steps: int | None = None,
    feature_budget: int | None = None,
    episode_logger: Callable[[int, float, float, OaKAgent], None] | None = None,
    episode_trace_logger: Callable[[EpisodeTrace[Any, Any, Any, Any]], None] | None = None,
    trace_selector: Callable[[int, int], bool] | None = None,
    capture_rendered_frames: bool = False,
    verbose: bool = True,
    device: torch.device | None = None,
) -> list[float]:
    """Run the full startup-spec, build, train pipeline on the given world."""
    embedded = hasattr(world, "description")

    if embedded:
        spec = _spec_from_description(
            cast(WorldWithDescription, world),
            ollama_model=ollama_model,
            verbose=verbose,
        )
    else:
        spec = _spec_from_discovery(
            world,
            ollama_model=ollama_model,
            verbose=verbose,
        )

    device, device_reason = _resolve_training_device(spec.perception_plan, device)

    if verbose:
        print(f"  Device: {device}")
        if device_reason != "explicit":
            print(f"  Device reason: {device_reason}")
        print(f"  Mode:   {'embedded' if embedded else 'discovery'}\n")

    if verbose:
        print("\n" + "=" * 60)
        print("Phase 2: BUILD, constructing OaK agent")
        print("=" * 60)

    agent = build_agent(
        spec,
        train_encoder=train_encoder,
        planning_budget=planning_budget,
        planning_warmup_steps=planning_warmup_steps,
        feature_budget=feature_budget,
        device=device,
    )

    if verbose:
        default_view = spec.perception_plan.view()
        print(f"  Default tensor view: {default_view.view_id}")
        print(f"  Encoder: {default_view.encoder_type}")
        print(f"  Feature groups: {len(spec.perception_plan.feature_groups)}")
        print(f"  Max options: 8")

    if verbose:
        print("\n" + "=" * 60)
        print("Phase 3: TRAIN, running OaK episode loop")
        print("=" * 60)

    reward_history = agent.train(
        world,
        num_episodes=num_episodes,
        average_window=average_window,
        solved_threshold=solved_threshold,
        episode_logger=episode_logger,
        episode_trace_logger=episode_trace_logger,
        trace_selector=trace_selector,
        capture_rendered_frames=capture_rendered_frames,
    )

    world.close()

    if verbose:
        final_window = (
            reward_history[-average_window:]
            if len(reward_history) >= average_window
            else reward_history
        )
        final_avg = sum(final_window) / max(len(final_window), 1)
        print(
            f"\n  Training complete. Final {average_window}-ep avg: {final_avg:.1f}"
        )

    return reward_history


def _coerce_agent_spec(config: dict[str, Any] | ExampleAgentSpec) -> ExampleAgentSpec:
    if isinstance(config, ExampleAgentSpec):
        return config

    description = _description_from_legacy_config(config)
    if "tensor_views" in config:
        perception_plan = _perception_plan_from_config(description, config)
    else:
        perception_plan = _perception_plan_from_legacy_config(description, config)
    return ExampleAgentSpec(
        perception_plan=perception_plan,
        source=str(config.get("source", "legacy")),
        metadata={"legacy": True},
    )


def _description_from_legacy_config(config: dict[str, Any]) -> WorldDescription:
    if "observation_channels" in config and "action" in config:
        return WorldDescription(
            observation_channels=tuple(
                ObservationChannelDescription(
                    channel_id=str(channel.get("channel_id", "main")),
                    kind=cast(Any, str(channel.get("kind", "raw_values"))),
                    path=tuple(channel.get("path", [])),
                    shape=tuple(channel["shape"]) if channel.get("shape") is not None else None,
                    dtype=channel.get("dtype"),
                    description=str(channel.get("description", "")),
                    value_names=tuple(channel.get("value_names", [])),
                    encoder_hint=channel.get("encoder_hint"),
                )
                for channel in cast(list[dict[str, Any]], config["observation_channels"])
            ),
            action=ActionDescription(
                action_type=str(config["action"].get("action_type", "discrete")),
                action_n=int(config["action"].get("action_n", config.get("action_n", 2))),
                labels=tuple(config["action"].get("labels", [])),
                description=str(config["action"].get("description", "")),
            ),
            default_encoder_type=cast(
                Any,
                str(
                    config.get(
                        "default_encoder_type",
                        config.get("encoder_type", "identity"),
                    )
                ),
            ),
            feature_hints=_feature_hints_from_legacy(config, default_channel="main"),
            notes=str(config.get("notes", "")),
        )

    obs_type = str(config.get("obs_type", "numeric_vector"))
    obs_shape_raw = config.get("obs_shape", (4,))
    obs_shape = tuple(obs_shape_raw) if obs_shape_raw is not None else None
    obs_dtype = str(config.get("obs_dtype", "float32"))
    default_encoder_type = str(config.get("encoder_type", "identity"))
    channel_kind = _channel_kind_from_obs_type(
        obs_type,
        encoder_type=default_encoder_type,
        obs_shape=obs_shape,
    )
    value_names = _legacy_value_names(config, obs_shape, channel_kind)

    return WorldDescription(
        observation_channels=(
            ObservationChannelDescription(
                channel_id="main",
                kind=cast(Any, channel_kind),
                shape=obs_shape,
                dtype=obs_dtype,
                description="Legacy primary observation channel.",
                value_names=value_names,
                encoder_hint=cast(Any, default_encoder_type),
            ),
        ),
        action=ActionDescription(
            action_type=str(config.get("action_type", "discrete")),
            action_n=int(config.get("action_n", 2)),
        ),
        default_encoder_type=cast(Any, default_encoder_type),
        feature_hints=_feature_hints_from_legacy(config, default_channel="main"),
        notes=str(config.get("notes", "")),
    )


def _perception_plan_from_config(
    description: WorldDescription,
    config: dict[str, Any],
) -> PerceptionPlan:
    tensor_views = tuple(
        TensorViewPlan(
            view_id=str(view.get("view_id", view.get("source_channel", "main"))),
            source_channel=str(view.get("source_channel", "main")),
            encoder_type=cast(
                Any,
                str(view.get("encoder_type", description.encoder_type)),
            ),
            input_shape=tuple(view["input_shape"]) if view.get("input_shape") is not None else None,
            input_dim=int(view["input_dim"]) if view.get("input_dim") is not None else None,
            input_channels=int(view.get("input_channels", 1)),
            latent_dim=int(view["latent_dim"]) if view.get("latent_dim") is not None else None,
            description=str(view.get("description", "")),
            selector_names=tuple(str(name) for name in view.get("selector_names", [])),
            selector_indices=tuple(int(index) for index in view.get("selector_indices", [])),
        )
        for view in cast(list[dict[str, Any]], config["tensor_views"])
    )
    if not tensor_views:
        return build_heuristic_perception_plan(description)
    default_view = str(config.get("default_tensor_view", tensor_views[0].view_id))
    if default_view not in {view.view_id for view in tensor_views}:
        default_view = tensor_views[0].view_id
    return PerceptionPlan(
        world_description=description,
        feature_groups=_feature_hints_from_legacy(config, default_channel=tensor_views[0].source_channel),
        tensor_views=tensor_views,
        default_tensor_view=default_view,
        notes=str(config.get("notes", "")),
        llm_used=bool(config.get("llm_used", False)),
    )


def _perception_plan_from_legacy_config(
    description: WorldDescription,
    config: dict[str, Any],
) -> PerceptionPlan:
    plan = build_heuristic_perception_plan(
        description,
        notes=str(config.get("notes", "Legacy config normalized to structured spec.")),
        llm_used=bool(config.get("llm_used", False)),
        metadata={"source": "legacy"},
    )
    if not plan.tensor_views:
        return plan

    default_view = plan.view()
    encoder_override = cast(
        Any,
        str(config.get("encoder_type", default_view.encoder_type)),
    )
    latent_override = (
        int(config["latent_dim"]) if config.get("latent_dim") is not None else default_view.latent_dim
    )
    if encoder_override == default_view.encoder_type and latent_override == default_view.latent_dim:
        return plan

    overridden_views = [replace(default_view, encoder_type=encoder_override, latent_dim=latent_override)]
    overridden_views.extend(
        view for view in plan.tensor_views if view.view_id != default_view.view_id
    )
    overridden_plan = PerceptionPlan(
        world_description=description,
        feature_groups=plan.feature_groups,
        tensor_views=tuple(overridden_views),
        default_tensor_view=plan.default_tensor_view,
        notes=plan.notes,
        llm_used=plan.llm_used,
        metadata={**plan.metadata, "explicit_default_tensor_view": plan.default_tensor_view},
    )
    return overridden_plan


def _feature_hints_from_legacy(
    config: dict[str, Any],
    *,
    default_channel: str,
) -> tuple[SemanticFieldPlan, ...]:
    features_raw = cast(list[dict[str, Any]], config.get("feature_hints") or config.get("features") or [])
    hints: list[SemanticFieldPlan] = []
    for index, feature in enumerate(features_raw):
        field_id = str(feature.get("id", f"feature_{index}"))
        hints.append(
            SemanticFieldPlan(
                field_id=field_id,
                name=str(feature.get("name", field_id)),
                source_channel=str(feature.get("source_channel", default_channel)),
                description=str(feature.get("description", "")),
                selector_names=tuple(feature.get("selector_names", [])),
                selector_indices=tuple(int(value) for value in feature.get("selector_indices", [])),
            )
        )
    return tuple(hints)


def _legacy_value_names(
    config: dict[str, Any],
    obs_shape: tuple[int, ...] | None,
    channel_kind: str,
) -> tuple[str, ...]:
    if channel_kind != "raw_values":
        return ()
    feature_names = [
        str(feature.get("id"))
        for feature in cast(list[dict[str, Any]], config.get("features", []))
        if feature.get("id")
    ]
    if feature_names and obs_shape and len(feature_names) == obs_shape[0]:
        return tuple(feature_names)
    if obs_shape and len(obs_shape) == 1:
        return tuple(f"value_{index}" for index in range(obs_shape[0]))
    return ()


def _channel_kind_from_obs_type(
    obs_type: str,
    *,
    encoder_type: str = "identity",
    obs_shape: tuple[int, ...] | None = None,
) -> str:
    if encoder_type == "cnn" and obs_shape is not None and len(obs_shape) >= 3:
        return "image"
    if obs_type in {"image", "grid"}:
        return "image"
    if obs_type == "text":
        return "text"
    if obs_type == "sound":
        return "sound"
    return "raw_values"


def _print_startup_summary(
    source: str,
    description: WorldDescription,
    perception_plan: PerceptionPlan,
) -> None:
    print(f"  Startup source: {source}")
    print(
        "  Observation channels: "
        + ", ".join(
            f"{channel.channel_id}:{channel.kind}"
            for channel in description.observation_channels
        )
    )
    print(f"  Action: {description.action_type}, n={description.action_n}")
    print(
        "  Tensor views: "
        + ", ".join(
            f"{view.view_id}:{view.encoder_type}"
            for view in perception_plan.tensor_views
        )
    )
    print(
        "  Feature groups: "
        + ", ".join(field.field_id for field in perception_plan.feature_groups)
    )
    print(f"  Planner: {'LLM' if perception_plan.llm_used else 'heuristic'}")


def _startup_plan(
    description: WorldDescription,
    observation_samples: list[Any],
    *,
    ollama_model: str,
) -> PerceptionPlan:
    llm_plan = analyze_world(
        description,
        observation_samples,
        model=ollama_model,
    )
    if llm_plan is not None:
        return stabilize_perception_plan(llm_plan)
    return stabilize_perception_plan(
        build_heuristic_perception_plan(
            description,
            notes="Heuristic startup perception plan (LLM unavailable).",
            llm_used=False,
            metadata={"planner": "heuristic"},
        )
    )
