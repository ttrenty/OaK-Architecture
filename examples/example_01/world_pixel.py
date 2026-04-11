"""Pixel-observation Gymnasium environments for CNN-based perception.

These environments wrap CartPole-v1 and expose rendered pixel observations
(84x84 RGB) instead of (or alongside) the raw state vector.

Two variants are provided:

- ``PixelCartPole-v1``: pure pixel observations (CNN only).
- ``MultiModalCartPole-v1``: dict observation with both pixel and state
  vector channels (CNN + identity encoders).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .schema import (
    ActionDescription,
    ObservationChannelDescription,
    SemanticFieldPlan,
    WorldDescription,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TARGET_H = 84
_TARGET_W = 84


def _resize_image(
    img: np.ndarray, target_h: int = _TARGET_H, target_w: int = _TARGET_W
) -> np.ndarray:
    """Nearest-neighbour resize using numpy fancy indexing."""
    h, w = img.shape[:2]
    row_idx = np.linspace(0, h - 1, target_h).astype(int)
    col_idx = np.linspace(0, w - 1, target_w).astype(int)
    return img[np.ix_(row_idx, col_idx)]


# ---------------------------------------------------------------------------
# Gymnasium environments
# ---------------------------------------------------------------------------


class PixelCartPoleEnv(gym.Env[np.ndarray, int]):
    """CartPole that returns 84x84 RGB pixel observations."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self._inner = gym.make("CartPole-v1", render_mode="rgb_array")
        self.action_space = self._inner.action_space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(_TARGET_H, _TARGET_W, 3), dtype=np.uint8
        )
        self.render_mode = render_mode

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _obs, info = self._inner.reset(seed=seed, options=options)
        pixels = self._capture()
        return pixels, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _obs, reward, terminated, truncated, info = self._inner.step(action)
        pixels = self._capture()
        return pixels, float(reward), terminated, truncated, info

    def render(self) -> Any | None:
        return self._capture()

    def close(self) -> None:
        self._inner.close()

    def _capture(self) -> np.ndarray:
        frame: Any = self._inner.render()
        if not isinstance(frame, np.ndarray):
            return np.zeros((_TARGET_H, _TARGET_W, 3), dtype=np.uint8)
        return _resize_image(frame)


class MultiModalCartPoleEnv(gym.Env[dict[str, np.ndarray], int]):
    """CartPole that returns both 84x84 pixels and the raw state vector."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self._inner = gym.make("CartPole-v1", render_mode="rgb_array")
        self.action_space = self._inner.action_space
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(
                    low=0,
                    high=255,
                    shape=(_TARGET_H, _TARGET_W, 3),
                    dtype=np.uint8,
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )
        self.render_mode = render_mode

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        state, info = self._inner.reset(seed=seed, options=options)
        pixels = self._capture()
        return {"pixels": pixels, "state": np.asarray(state, dtype=np.float32)}, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        state, reward, terminated, truncated, info = self._inner.step(action)
        pixels = self._capture()
        return (
            {"pixels": pixels, "state": np.asarray(state, dtype=np.float32)},
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self) -> Any | None:
        return self._capture()

    def close(self) -> None:
        self._inner.close()

    def _capture(self) -> np.ndarray:
        frame: Any = self._inner.render()
        if not isinstance(frame, np.ndarray):
            return np.zeros((_TARGET_H, _TARGET_W, 3), dtype=np.uint8)
        return _resize_image(frame)


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------

gym.register(
    id="PixelCartPole-v1",
    entry_point="examples.example_01.world_pixel:PixelCartPoleEnv",
    max_episode_steps=500,
)

gym.register(
    id="MultiModalCartPole-v1",
    entry_point="examples.example_01.world_pixel:MultiModalCartPoleEnv",
    max_episode_steps=500,
)


# ---------------------------------------------------------------------------
# World descriptions
# ---------------------------------------------------------------------------

PIXEL_CARTPOLE_WORLD_DESCRIPTION = WorldDescription(
    observation_channels=(
        ObservationChannelDescription(
            channel_id="pixels",
            kind="image",
            shape=(_TARGET_H, _TARGET_W, 3),
            dtype="uint8",
            description="RGB pixel rendering of CartPole state.",
            encoder_hint="cnn",
        ),
    ),
    action=ActionDescription(
        action_type="discrete",
        action_n=2,
        labels=("push_left", "push_right"),
        description="Push the cart left or right.",
    ),
    default_encoder_type="cnn",
    feature_hints=(
        SemanticFieldPlan(
            field_id="visual_scene",
            name="Visual scene",
            source_channel="pixels",
            description="RGB pixel rendering of the CartPole environment.",
        ),
    ),
    notes="CartPole rendered as 84x84 RGB pixels for CNN encoding.",
    metadata={"env_id": "PixelCartPole-v1"},
)


MULTIMODAL_CARTPOLE_WORLD_DESCRIPTION = WorldDescription(
    observation_channels=(
        ObservationChannelDescription(
            channel_id="pixels",
            kind="image",
            path=("pixels",),
            shape=(_TARGET_H, _TARGET_W, 3),
            dtype="uint8",
            description="RGB pixel rendering of CartPole state.",
            encoder_hint="cnn",
        ),
        ObservationChannelDescription(
            channel_id="state",
            kind="raw_values",
            path=("state",),
            shape=(4,),
            dtype="float32",
            description="CartPole state vector.",
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
    default_encoder_type="cnn",
    feature_hints=(
        SemanticFieldPlan(
            field_id="visual_scene",
            name="Visual scene",
            source_channel="pixels",
            description="RGB pixel rendering of the CartPole environment.",
        ),
        SemanticFieldPlan(
            field_id="cart_motion",
            name="Cart motion",
            source_channel="state",
            description="Horizontal cart position and velocity.",
            selector_names=("cart_position", "cart_velocity"),
        ),
        SemanticFieldPlan(
            field_id="pole_balance",
            name="Pole balance",
            source_channel="state",
            description="Pole angle and angular velocity.",
            selector_names=("pole_angle", "pole_angular_velocity"),
        ),
    ),
    notes="CartPole with both 84x84 RGB pixels (CNN) and state vector (identity).",
    metadata={"env_id": "MultiModalCartPole-v1"},
)
