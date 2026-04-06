"""Gymnasium wrapper conforming to the OaK World protocol.

The wrapper exposes no metadata about observation or action spaces.
The agent must discover everything through interaction.
"""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym

from oak.types import TimeStep


class GymWorld:
    """Thin gymnasium wrapper for any environment, with no metadata exposure."""

    def __init__(
        self,
        env_id: str,
        *,
        make_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.env_id = env_id
        self.env = gym.make(env_id, **dict(make_kwargs or {}))

    def reset(self) -> TimeStep:
        obs, info = self.env.reset()
        return TimeStep(observation=obs, reward=0.0, info=info)

    def step(self, action: object) -> TimeStep:
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
