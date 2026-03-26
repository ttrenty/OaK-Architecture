"""CartPole gymnasium wrapper with embedded observation/action space descriptions.

Unlike `world.py` (which exposes NO metadata), this wrapper provides a
`WorldDescription` that tells the agent everything it needs to know about
the observation and action spaces up front, so no trial-and-error discovery is
needed.

Any world that exposes a `description` attribute (a `WorldDescription`)
will automatically use the embedded config path in `run_training()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym

from oak_architecture.types import TimeStep


@dataclass(frozen=True)
class WorldDescription:
    """Metadata that describes the observation and action spaces.

    This is the information the discovery approach learns through
    trial-and-error.  The embedded approach provides it directly.
    """

    obs_type: str
    obs_shape: tuple[int, ...]
    obs_dtype: str
    action_type: str
    action_n: int
    encoder_type: str
    features: list[dict[str, Any]] = field(default_factory=list)

    def to_config(self) -> dict[str, Any]:
        """Convert to the config dict expected by `build_agent`."""
        return {
            "obs_type": self.obs_type,
            "obs_shape": self.obs_shape,
            "obs_dtype": self.obs_dtype,
            "action_type": self.action_type,
            "action_n": self.action_n,
            "encoder_type": self.encoder_type,
            "features": list(self.features),
        }


class DescribedCartPoleWorld:
    """Gymnasium CartPole wrapper with embedded space descriptions.

    Functionally identical to `CartPoleWorld` for `reset()`/`step()`,
    but also exposes a `.description` attribute that the runner can read
    instead of running the discovery phase.
    """

    description = WorldDescription(
        obs_type="numeric_vector",
        obs_shape=(4,),
        obs_dtype="float32",
        action_type="discrete",
        action_n=2,
        encoder_type="identity",
        features=[
            {
                "id": "cart_position",
                "name": "Cart position",
                "description": "Horizontal position of the cart on the track",
            },
            {
                "id": "cart_velocity",
                "name": "Cart velocity",
                "description": "Horizontal velocity of the cart",
            },
            {
                "id": "pole_angle",
                "name": "Pole angle",
                "description": "Angle of the pole from vertical (radians)",
            },
            {
                "id": "pole_angular_velocity",
                "name": "Pole angular velocity",
                "description": "Angular velocity of the pole tip",
            },
        ],
    )

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
