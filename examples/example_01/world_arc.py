"""ARC-AGI-3 environment wrapper conforming to the OaK World protocol.

Wraps interactive ARC-AGI-3 game environments so that the OaK agent can
interact with them step-by-step.  Observations are one-hot encoded 64x64
grids (16 color channels), and discrete keyboard actions are mapped to
``GameAction`` enums.

ARC-AGI-3 games are *continuous*: the agent plays through multiple levels
in sequence without restarting.  The ``World`` protocol's ``reset()``
initialises the game on the first call and is a no-op thereafter (the
game state carries forward).

Only keyboard-action environments are supported for now (no coordinate
click ACTION6).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from oak.interfaces import World
from oak.types import TimeStep

import arc_agi
from arcengine.enums import GameAction, GameState, FrameDataRaw

from .world_embedded import WorldDescription


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 64
NUM_COLORS = 16

# Reward shaping: ARC-AGI-3 only provides WIN/GAME_OVER states, so we
# synthesise scalar rewards for the RL agent.
REWARD_WIN = 10.0
REWARD_GAME_OVER = -1.0
REWARD_STEP = -0.01  # small penalty to encourage efficiency (RHAE-aligned)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_hot_grid(frame: np.ndarray) -> np.ndarray:
    """Convert a (64, 64) int8 grid to a (64, 64, 16) float32 one-hot tensor.

    Colors are categorical (not ordinal), so one-hot encoding preserves the
    right inductive bias for convolutional encoders.
    """
    grid = np.clip(frame.astype(np.int32), 0, NUM_COLORS - 1)
    one_hot = np.zeros((GRID_SIZE, GRID_SIZE, NUM_COLORS), dtype=np.float32)
    rows, cols = np.meshgrid(
        np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing="ij"
    )
    one_hot[rows, cols, grid] = 1.0
    return one_hot


def _grid_from_obs(
    obs: FrameDataRaw | None, fallback: np.ndarray | None = None
) -> np.ndarray:
    """Extract a one-hot grid from an observation, with fallback."""
    if obs is not None and obs.frame and len(obs.frame) > 0:
        return _one_hot_grid(obs.frame[0])
    if fallback is not None:
        return fallback
    return np.zeros((GRID_SIZE, GRID_SIZE, NUM_COLORS), dtype=np.float32)


# ---------------------------------------------------------------------------
# ARC World Description
# ---------------------------------------------------------------------------


def arc_world_description(action_n: int) -> WorldDescription:
    """Build a ``WorldDescription`` for an ARC-AGI-3 keyboard environment."""
    return WorldDescription(
        obs_type="grid",
        obs_shape=(GRID_SIZE, GRID_SIZE, NUM_COLORS),
        obs_dtype="float32",
        action_type="discrete",
        action_n=action_n,
        encoder_type="cnn",
        features=[
            {
                "id": "grid_pattern",
                "name": "Grid pattern",
                "description": "Spatial color pattern on the 64x64 grid",
            },
            {
                "id": "grid_change",
                "name": "Grid change",
                "description": "Difference in grid state between steps",
            },
        ],
    )


# ---------------------------------------------------------------------------
# ARC World
# ---------------------------------------------------------------------------


class ArcWorld(World[np.ndarray, int, dict[str, Any]]):
    """OaK ``World`` wrapper for a single ARC-AGI-3 environment.

    The game is played *continuously* across levels.  ``reset()`` starts
    the game on the first call; subsequent calls return the current frame
    so the agent can resume.

    ``step()`` takes a discrete keyboard action and returns a ``TimeStep``.
    The ``terminated`` flag is set on WIN (level done) or GAME_OVER (game
    done).  ``truncated`` is set when the total action budget is exceeded.

    Level progression is managed by the ARC engine: WIN automatically
    advances to the next level.  RHAE scoring should be read from the
    ``arc_agi`` scorecard after the run completes.

    Parameters
    ----------
    env:
        An ``arc_agi.EnvironmentWrapper`` created via ``arcade.make()``.
    env_info:
        The ``EnvironmentInfo`` for this environment (provides baseline
        action counts and metadata).
    max_actions_multiplier:
        Total action budget as a multiple of the sum of human baselines.
        Default is 5 (matching the ARC-AGI-3 scoring rules).
    """

    def __init__(
        self,
        env: arc_agi.EnvironmentWrapper,
        env_info: arc_agi.EnvironmentInfo,
        *,
        max_actions_multiplier: int = 5,
    ) -> None:
        self.env = env
        self.env_info = env_info

        # Map integer actions (0..n-1) to GameAction enums.
        # Filter out ACTION6 (coordinate click) and ACTION7 (undo) for now.
        keyboard_actions = [a for a in env.action_space if a.value <= 5]
        if not keyboard_actions:
            raise ValueError(
                f"Environment {env_info.title} has no keyboard actions "
                f"(action_space={env.action_space}). "
                "Only keyboard environments are supported."
            )
        self._actions: list[GameAction] = keyboard_actions
        self._action_n = len(keyboard_actions)

        # Build the embedded description.
        self.description = arc_world_description(self._action_n)

        # Total action budget across all levels.
        total_baseline = sum(env_info.baseline_actions)
        self._max_total_actions = max_actions_multiplier * total_baseline
        self._total_actions = 0

        # State tracking.
        self._done = False
        self._last_obs: FrameDataRaw | None = None
        self._last_grid: np.ndarray | None = None
        self._levels_completed = 0

    @property
    def action_n(self) -> int:
        return self._action_n

    @property
    def win_levels(self) -> int:
        if self._last_obs is not None:
            return self._last_obs.win_levels
        return len(self.env_info.baseline_actions)

    @property
    def baseline_actions(self) -> list[int]:
        return list(self.env_info.baseline_actions)

    @property
    def is_game_over(self) -> bool:
        return self._done

    @property
    def total_actions(self) -> int:
        return self._total_actions

    @property
    def levels_completed(self) -> int:
        return self._levels_completed

    # -- World protocol -----------------------------------------------------

    def reset(self) -> TimeStep[np.ndarray, dict[str, Any]]:
        """Initialise the game (first call) or return current state."""
        if self._last_obs is None:
            obs = self.env.reset()
            self._last_obs = obs
            self._last_grid = _grid_from_obs(obs)
        grid = self._last_grid if self._last_grid is not None else _grid_from_obs(None)
        return TimeStep(observation=grid, reward=0.0)

    def step(self, action: int) -> TimeStep[np.ndarray, dict[str, Any]]:
        """Take one discrete action and return the resulting ``TimeStep``."""
        if self._done:
            grid = self._last_grid if self._last_grid is not None else _grid_from_obs(None)
            return TimeStep(observation=grid, reward=0.0, terminated=True)

        if action < 0 or action >= self._action_n:
            raise ValueError(
                f"Action {action} out of range [0, {self._action_n})"
            )

        game_action = self._actions[action]
        obs = self.env.step(game_action)
        self._total_actions += 1

        if obs is None:
            self._done = True
            grid = self._last_grid if self._last_grid is not None else _grid_from_obs(None)
            return TimeStep(
                observation=grid, reward=REWARD_GAME_OVER, terminated=True
            )

        self._last_obs = obs
        grid = _grid_from_obs(obs, self._last_grid)
        self._last_grid = grid

        # -- WIN: level completed, engine auto-advances --
        if obs.state == GameState.WIN:
            # Some environments report the incremented level count one frame
            # later, so count the WIN transition itself as forward progress.
            self._levels_completed = max(
                self._levels_completed + 1,
                obs.levels_completed,
            )
            if self._levels_completed >= obs.win_levels:
                self._done = True
            return TimeStep(observation=grid, reward=REWARD_WIN, terminated=True)

        # -- GAME_OVER: game ended --
        if obs.state == GameState.GAME_OVER:
            self._done = True
            return TimeStep(
                observation=grid, reward=REWARD_GAME_OVER, terminated=True
            )

        # -- Total budget exceeded --
        if self._total_actions >= self._max_total_actions:
            self._done = True
            return TimeStep(
                observation=grid, reward=REWARD_GAME_OVER, truncated=True
            )

        return TimeStep(observation=grid, reward=REWARD_STEP)

    def close(self) -> None:
        """No-op: the Arcade manages environment lifecycle."""
