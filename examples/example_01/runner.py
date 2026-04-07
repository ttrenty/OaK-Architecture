"""Config → Build → Train flow for the OaK agent.

This module orchestrates three phases:

1. **Config**: obtain the observation/action space configuration from the world.
   If the world exposes a `description` attribute, the config is read directly.
   Otherwise, trial-and-error discovery is used (optionally refined by an LLM).
2. **Build**: construct the OaK agent with all four modules using the config.
3. **Train**: run the standard OaK episode loop until solved or max episodes.

The orchestration is environment-agnostic. To switch to a different RL problem,
pass a different `World` implementation to `run_training()`.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, cast

import torch

from oak.agent import OaKAgent
from oak.interfaces import World

from .encoders import create_encoder
from .perception import AdaptivePerception
from .reactive_policy import OptionCriticPolicy
from .transition_model import DynaTransitionModel
from .value_function import OptionValueFunction
from .discovery import DiscoveryManager
from .llm import analyze_world
from .world_embedded import WorldDescription


class WorldWithDescription(World[Any, Any, Any], Protocol):
    """World protocol with an embedded config description."""

    description: WorldDescription


def build_agent(
    config: dict[str, Any],
    *,
    train_encoder: bool = False,
    planning_budget: int = 5,
    planning_warmup_steps: int = 500,
    feature_budget: int = 2,
    device: torch.device | None = None,
) -> OaKAgent:
    """Build a fully wired OaK agent from a config dict.

    The config dict can come from either discovery or an embedded world
    description; `build_agent` does not care which.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_shape = config.get("obs_shape", (4,))
    obs_dim = obs_shape[0] if obs_shape else 4
    action_n: int = config.get("action_n", 2)
    encoder_type: str = config.get("encoder_type", "mlp")
    features: list[dict[str, Any]] = config.get("features", [])

    # Create encoder: latent_dim depends on encoder type
    if encoder_type == "identity":
        latent_dim = obs_dim
    elif encoder_type == "cnn":
        latent_dim = config.get("latent_dim", 128)
    else:
        latent_dim = config.get("latent_dim", max(obs_dim * 4, 32))

    # For CNN, determine input channels from observation shape.
    input_channels = 3
    if encoder_type == "cnn" and len(obs_shape) >= 3:
        input_channels = obs_shape[-1]  # (H, W, C) convention

    encoder = create_encoder(
        encoder_type,
        obs_dim,
        latent_dim=latent_dim,
        trainable=train_encoder,
        input_channels=input_channels,
    )

    # If no features provided, create generic per-dimension features
    if not features:
        features = [
            {
                "id": f"dim_{i}",
                "name": f"Dimension {i}",
                "description": f"Observation dimension {i}",
            }
            for i in range(obs_dim)
        ]

    # Build modules
    value_function = OptionValueFunction(
        state_dim=latent_dim,
        max_options=8,
        device=device,
    )

    perception = AdaptivePerception(
        encoder=encoder,
        initial_features=features,
        train_encoder=train_encoder,
    )

    transition_model = DynaTransitionModel(
        state_dim=latent_dim,
        num_actions=action_n,
        planning_warmup_steps=planning_warmup_steps,
        device=device,
    )

    reactive_policy = OptionCriticPolicy(
        value_function=value_function,
        num_actions=action_n,
        state_dim=latent_dim,
        device=device,
    )

    return OaKAgent(
        perception=perception,
        transition_model=transition_model,
        value_function=value_function,
        reactive_policy=reactive_policy,
        planning_budget=planning_budget,
        feature_budget=feature_budget,
        option_stop_threshold=0.5,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Config acquisition helpers
# ──────────────────────────────────────────────────────────────────────────────


def _config_from_discovery(
    world: World[Any, Any, Any],
    *,
    ollama_model: str,
    verbose: bool,
) -> dict[str, Any]:
    """Run trial-and-error discovery + optional LLM analysis."""
    if verbose:
        print("=" * 60)
        print("Phase 1: DISCOVERY, probing observation and action spaces")
        print("=" * 60)

    discovery = DiscoveryManager()

    while not discovery.is_complete():
        discovery.probe_step(world)

    config = discovery.get_config()
    if verbose:
        print(f"  Observation: type={config['obs_type']}, shape={config['obs_shape']}")
        print(f"  Action:      type={config['action_type']}, n={config['action_n']}")

    # ── LLM analysis (optional) ──────────────────────────────────────
    if verbose:
        print("\n  Consulting LLM for feature analysis...")

    llm_result = analyze_world(
        discovery.observation_samples,
        {
            "action_type": config["action_type"],
            "action_n": config["action_n"],
        },
        model=ollama_model,
    )

    if llm_result:
        if verbose:
            print(
                f"  LLM encoder recommendation: {llm_result.get('encoder_type', 'N/A')}"
            )
            llm_features = llm_result.get("features", [])
            print(f"  LLM proposed {len(llm_features)} features")
            if llm_result.get("notes"):
                print(f"  LLM notes: {llm_result['notes']}")
        # Merge LLM results into config (features only).
        # Keep the discovery heuristic's encoder choice: it knows
        # whether identity is sufficient for small vector observations,
        # while the LLM lacks context about encoder training state.
        if "features" in llm_result and llm_result["features"]:
            config["features"] = llm_result["features"]
    else:
        if verbose:
            print("  LLM not available, using heuristic defaults")

    return config


def _config_from_description(
    world: WorldWithDescription, *, verbose: bool
) -> dict[str, Any]:
    """Read the config directly from the world's embedded description."""
    description = world.description
    config = description.to_config()

    if verbose:
        print("=" * 60)
        print("Phase 1: CONFIG, reading embedded world description")
        print("=" * 60)
        print(f"  Observation: type={config['obs_type']}, shape={config['obs_shape']}")
        print(f"  Action:      type={config['action_type']}, n={config['action_n']}")
        n_feat = len(config.get("features", []))
        print(f"  Features:    {n_feat} provided by world description")

    return config


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_training(
    world: World[Any, Any, Any],
    *,
    num_episodes: int = 500,
    average_window: int = 100,
    solved_threshold: float | None = None,
    ollama_model: str = "qwen3.5:9b",
    train_encoder: bool = False,
    planning_budget: int = 5,
    planning_warmup_steps: int = 500,
    episode_logger: Callable[[int, float, float, OaKAgent], None] | None = None,
    verbose: bool = True,
    device: torch.device | None = None,
) -> list[float]:
    """Run the full config, build, train pipeline on the given world.

    The config-acquisition mode is chosen automatically:

    - If `world` has a `description` attribute, the config is read from it
      (embedded mode, no discovery, no LLM).
    - Otherwise, the agent discovers observation/action spaces through
      trial-and-error, optionally refined by an LLM (discovery mode).

    Parameters
    ----------
    world:
        An environment implementing the `World` protocol.  If it exposes
        a `description` attribute with a `to_config()` method, the
        embedded config path is used.
    num_episodes:
        Maximum number of training episodes.
    average_window:
        Number of recent episodes to average for performance tracking.
    solved_threshold:
        If set, stop early when the `average_window`-episode average reward
        reaches this value.
    ollama_model:
        Ollama model name (only used in discovery mode).
    train_encoder:
        Whether to train the encoder via reconstruction loss.
    planning_budget:
        Number of Dyna-Q imagined rollouts per step.
    planning_warmup_steps:
        Number of real transitions to observe before planning is enabled.
    episode_logger:
        Optional callback `(episode, episode_reward, avg_reward, agent)` used
        for per-episode logging or other user-controlled training hooks.
    verbose:
        Print setup and final-summary messages to stdout.
    device:
        Torch device.  Defaults to CUDA if available.

    Returns
    -------
    list[float]
        Per-episode reward history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedded = hasattr(world, "description")

    if verbose:
        print(f"  Device: {device}")
        print(f"  Mode:   {'embedded' if embedded else 'discovery'}\n")

    # ── Phase 1: CONFIG ───────────────────────────────────────────────
    if embedded:
        config = _config_from_description(
            cast(WorldWithDescription, world), verbose=verbose
        )
    else:
        config = _config_from_discovery(
            world, ollama_model=ollama_model, verbose=verbose
        )

    # ── Phase 2: BUILD ────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("Phase 2: BUILD, constructing OaK agent")
        print("=" * 60)

    agent = build_agent(
        config,
        train_encoder=train_encoder,
        planning_budget=planning_budget,
        planning_warmup_steps=planning_warmup_steps,
        device=device,
    )

    n_features = len(agent.perception.list_features())
    if verbose:
        print(f"  Encoder: {config.get('encoder_type', 'mlp')}")
        print(f"  Features: {n_features}")
        print(f"  Max options: 8")

    # ── Phase 3: TRAIN ────────────────────────────────────────────────
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
