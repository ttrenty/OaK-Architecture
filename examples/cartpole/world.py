"""CartPole gymnasium wrapper conforming to the OaK World protocol.

The wrapper exposes NO metadata about observation or action spaces.
The agent must discover everything through interaction.
"""

from __future__ import annotations

import gymnasium as gym

from oak_architecture.types import TimeStep


class CartPoleWorld:
    """Thin gymnasium wrapper, no metadata exposure."""

    def __init__(self) -> None:
        self.env = gym.make("CartPole-v1")

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
