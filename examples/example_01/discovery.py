"""Trial-and-error discovery of observation and action structure."""

from __future__ import annotations

import logging
from typing import Any

from oak.interfaces import World
from oak.types import TimeStep

from .schema import ActionDescription, WorldDescription
from .startup import infer_world_description

logger = logging.getLogger(__name__)


class DiscoveryManager:
    """Discovers observation and action spaces through trial-and-error."""

    _INTEGER_PROBES: list[int] = list(range(20))
    _FLOAT_PROBES: list[float] = [0.0, 0.5, 1.0, -1.0, -0.5]

    def __init__(self, *, max_probes: int = 50) -> None:
        self.observation_samples: list[Any] = []

        self._max_probes = max_probes
        self._probes_done = 0

        self._valid_int_actions: list[int] = []
        self._int_upper_found = False
        self._float_idx = 0
        self._valid_float_actions: list[float] = []
        self._phase: str = "integers"

        self._world_description: WorldDescription | None = None

    def probe_step(
        self, world: World[Any, Any, Any]
    ) -> tuple[bool, TimeStep[Any, Any] | None]:
        """Try the next action prototype and record observation samples."""
        if self._probes_done >= self._max_probes:
            self._phase = "done"
            return False, None

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

    def get_world_description(self) -> WorldDescription:
        """Return the discovered structured world description."""
        if not self.observation_samples:
            raise RuntimeError("No observation samples were collected during discovery")
        self._world_description = infer_world_description(
            self.observation_samples[0],
            self.action_description(),
            notes="World description inferred from raw discovery samples.",
            metadata={"source": "discovery"},
        )
        return self._world_description

    def get_config(self) -> dict[str, Any]:
        """Compatibility export for older callers."""
        return self.get_world_description().to_config()

    def action_description(self) -> ActionDescription:
        action_type, action_n = self._infer_action_space()
        return ActionDescription(
            action_type=action_type,
            action_n=action_n,
            description="Action space inferred through probe actions.",
        )

    def _record_observation(self, obs: Any) -> None:
        if len(self.observation_samples) < 20:
            self.observation_samples.append(obs)

    def _next_action(self) -> Any | None:
        if self._phase == "integers":
            if not self._int_upper_found:
                idx = len(self._valid_int_actions)
                if idx < len(self._INTEGER_PROBES):
                    return self._INTEGER_PROBES[idx]
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
            if self._valid_int_actions:
                self._phase = "done"
            else:
                self._phase = "floats"

    def _infer_action_space(self) -> tuple[str, int]:
        if self._valid_int_actions:
            return "discrete", max(self._valid_int_actions) + 1
        if self._valid_float_actions:
            return "continuous", 1
        return "discrete", 2
