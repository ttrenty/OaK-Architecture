"""Raw Gymnasium wrapper conforming to the OaK World protocol.

This wrapper intentionally returns the original environment observations.
Discovery and perception are responsible for turning those raw values into
the normalized `AgentObservation` and structured subjective state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Generic, TypeVar, cast

import gymnasium as gym

from oak.interfaces import World
from oak.types import TimeStep


GymObsT = TypeVar("GymObsT")
GymActionT = TypeVar("GymActionT")


class GymWorld(World[GymObsT, GymActionT, dict[str, Any]], Generic[GymObsT, GymActionT]):
    """Thin Gymnasium wrapper with no embedded observation semantics."""

    def __init__(
        self,
        env_id: str,
        *,
        make_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.env_id = env_id
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

    def render_frame(self) -> Any | None:
        """Return the current rendered frame when the env was created with rendering."""
        try:
            return self.env.render()
        except Exception:
            return None

    def close(self) -> None:
        self.env.close()
