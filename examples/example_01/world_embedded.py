"""Gymnasium world wrappers with embedded structured world descriptions.

Unlike `world.py` (which exposes only raw observations), this wrapper
embeds a `WorldDescription` that gives the startup planner a typed
observation-channel schema and action metadata up front.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Generic, TypeVar, cast

import gymnasium as gym

from oak.interfaces import World
from oak.types import TimeStep

from .schema import (
    ActionDescription,
    ObservationChannelDescription,
    SemanticFieldPlan,
    WorldDescription,
)


GymObsT = TypeVar("GymObsT")
GymActionT = TypeVar("GymActionT")


CARTPOLE_WORLD_DESCRIPTION = WorldDescription(
    observation_channels=(
        ObservationChannelDescription(
            channel_id="main",
            kind="raw_values",
            shape=(4,),
            dtype="float32",
            description="CartPole state vector from Gymnasium.",
            value_names=(
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_angular_velocity",
            ),
            encoder_hint="identity",
        ),
    ),
    action=ActionDescription(
        action_type="discrete",
        action_n=2,
        labels=("push_left", "push_right"),
        description="Push the cart left or right.",
    ),
    default_encoder_type="identity",
    feature_hints=(
        SemanticFieldPlan(
            field_id="cart_motion",
            name="Cart motion",
            source_channel="main",
            description="Horizontal cart position and velocity.",
            selector_names=("cart_position", "cart_velocity"),
        ),
        SemanticFieldPlan(
            field_id="pole_balance",
            name="Pole balance",
            source_channel="main",
            description="Pole angle and angular velocity.",
            selector_names=("pole_angle", "pole_angular_velocity"),
        ),
    ),
    notes="Bundled CartPole description with grouped raw-value semantics.",
    metadata={"env_id": "CartPole-v1"},
)

_KNOWN_WORLD_DESCRIPTIONS: dict[str, WorldDescription] = {
    "CartPole-v1": CARTPOLE_WORLD_DESCRIPTION,
}


class DescribedGymWorld(
    World[GymObsT, GymActionT, dict[str, Any]],
    Generic[GymObsT, GymActionT],
):
    """Gymnasium wrapper with an embedded `WorldDescription`.

    Functionally identical to `GymWorld` for `reset()`/`step()`, but also
    exposes a `.description` attribute that the runner can read instead of
    inferring the observation/action schema from raw samples.
    """

    def __init__(
        self,
        env_id: str,
        *,
        description: WorldDescription | None = None,
        make_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        resolved_description = description or _KNOWN_WORLD_DESCRIPTIONS.get(env_id)
        if resolved_description is None:
            raise ValueError(
                "DescribedGymWorld requires a WorldDescription for unknown env_id "
                f"{env_id!r}"
            )

        self.env_id = env_id
        self.description = resolved_description
        self.env: gym.Env[GymObsT, GymActionT] = cast(
            gym.Env[GymObsT, GymActionT],
            gym.make(env_id, **dict(make_kwargs or {})),
        )

    def reset(self) -> TimeStep[GymObsT, dict[str, Any]]:
        obs, info = self.env.reset()
        return TimeStep(observation=obs, reward=0.0, info=info)

    def step(self, action: GymActionT) -> TimeStep[GymObsT, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return TimeStep(
            observation=obs,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self) -> None:
        self.env.close()
