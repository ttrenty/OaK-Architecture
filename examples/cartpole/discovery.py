"""Trial-and-error discovery of observation and action spaces.

The `DiscoveryManager` interacts with a World to determine:
- The observation type, shape, and dtype
- The action type and range (by trying action prototypes until one fails)

No environment metadata is read; everything is discovered through
interaction.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from oak_architecture.interfaces import World
from oak_architecture.types import TimeStep

logger = logging.getLogger(__name__)


class DiscoveryManager:
    """Discovers observation and action spaces through trial-and-error."""

    # Action prototypes to try, in order.
    _INTEGER_PROBES: list[int] = list(range(20))
    _FLOAT_PROBES: list[float] = [0.0, 0.5, 1.0, -1.0, -0.5]

    def __init__(self, *, max_probes: int = 50) -> None:
        self.observation_samples: list[Any] = []

        self._max_probes = max_probes
        self._probes_done = 0

        # Action discovery state
        self._valid_int_actions: list[int] = []
        self._int_upper_found = False  # True once an int fails
        self._float_idx = 0
        self._valid_float_actions: list[float] = []
        self._phase: str = "integers"  # "integers" → "floats" → "done"

        # Observation discovery state
        self._obs_type: str | None = None
        self._obs_shape: tuple[int, ...] | None = None
        self._obs_dtype: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe_step(
        self, world: World[Any, Any, Any]
    ) -> tuple[bool, TimeStep[Any, Any] | None]:
        """Try the next action prototype.

        Returns `(success, time_step)` where *success* is True when the
        world accepted the action.  On the very first call the world is
        reset and the initial observation is recorded.
        """
        if self._probes_done >= self._max_probes:
            self._phase = "done"
            return False, None

        # Always reset before probing so we have a fresh state
        ts = world.reset()
        self._record_observation(ts.observation)

        action = self._next_action()
        if action is None:
            self._phase = "done"
            return False, None

        try:
            ts = world.step(action)
            self._record_success(action)
            self._record_observation(ts.observation)
            self._probes_done += 1
            return True, ts
        except Exception:
            self._record_failure(action)
            self._probes_done += 1
            return False, None

    def is_complete(self) -> bool:
        return self._phase == "done"

    def get_config(self) -> dict[str, Any]:
        """Return the discovered configuration."""
        self._infer_obs_type()

        # Determine action config
        action_type, action_n = self._infer_action_space()

        return {
            "obs_type": self._obs_type or "unknown",
            "obs_shape": self._obs_shape,
            "obs_dtype": self._obs_dtype or "float32",
            "action_type": action_type,
            "action_n": action_n,
            "encoder_type": self._recommend_encoder(),
        }

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _record_observation(self, obs: Any) -> None:
        if len(self.observation_samples) < 20:
            self.observation_samples.append(obs)

        if self._obs_shape is None:
            self._infer_obs_type_from(obs)

    def _infer_obs_type_from(self, obs: Any) -> None:
        if isinstance(obs, np.ndarray):
            self._obs_shape = obs.shape
            self._obs_dtype = str(obs.dtype)
            if obs.ndim == 1:
                self._obs_type = "numeric_vector"
            elif obs.ndim == 3:
                self._obs_type = "image"
            else:
                self._obs_type = "numeric_array"
        elif isinstance(obs, (list, tuple)):
            self._obs_shape = (len(obs),)
            self._obs_type = "numeric_vector"
            self._obs_dtype = "float32"
        elif isinstance(obs, str):
            self._obs_type = "text"
            self._obs_shape = None
            self._obs_dtype = "str"
        else:
            self._obs_type = "unknown"
            self._obs_shape = None

    def _infer_obs_type(self) -> None:
        if self._obs_type is not None:
            return
        if self.observation_samples:
            self._infer_obs_type_from(self.observation_samples[0])

    def _recommend_encoder(self) -> str:
        if self._obs_type == "image":
            return "cnn"
        if self._obs_type in ("numeric_vector", "numeric_array"):
            # For small vector observations, identity encoding is more
            # stable (avoids random projection issues).  The downstream
            # Q-networks have their own hidden layers for representation.
            if self._obs_shape and self._obs_shape[0] <= 32:
                return "identity"
            return "mlp"
        return "identity"

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _next_action(self) -> Any | None:
        """Return the next action prototype to try."""
        if self._phase == "integers":
            if not self._int_upper_found:
                idx = len(self._valid_int_actions) + (
                    0 if not self._valid_int_actions else 0
                )
                if idx < len(self._INTEGER_PROBES):
                    return self._INTEGER_PROBES[idx]
            # All integer probes exhausted or upper bound found
            self._phase = "done"
            return None

        if self._phase == "floats":
            if self._float_idx < len(self._FLOAT_PROBES):
                action = self._FLOAT_PROBES[self._float_idx]
                self._float_idx += 1
                return action
            self._phase = "done"
            return None

        return None

    def _record_success(self, action: Any) -> None:
        if isinstance(action, int):
            self._valid_int_actions.append(action)
            logger.debug(f"Action {action} accepted")
        elif isinstance(action, float):
            self._valid_float_actions.append(action)

    def _record_failure(self, action: Any) -> None:
        logger.debug(f"Action {action} rejected")
        if isinstance(action, int) and self._phase == "integers":
            self._int_upper_found = True
            # If at least one int worked, we have a discrete space
            if self._valid_int_actions:
                self._phase = "done"
            else:
                # No ints worked at all, try floats
                self._phase = "floats"

    def _infer_action_space(self) -> tuple[str, int]:
        if self._valid_int_actions:
            n = max(self._valid_int_actions) + 1
            return "discrete", n
        if self._valid_float_actions:
            return "continuous", 1
        # Fallback
        return "discrete", 2
