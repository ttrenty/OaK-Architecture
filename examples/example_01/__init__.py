"""Example 01: a full OaK RL agent with discovery and described-world support.

The learning modules are environment-agnostic: pass any compatible `World` to
`run_training()` and the same agent pipeline can be reused.

CartPole is bundled as one sample gymnasium world description, but it is not
part of the example's identity.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_agent",
    "run_training",
    "GymWorld",
    "DescribedGymWorld",
    "CARTPOLE_WORLD_DESCRIPTION",
    "WorldDescription",
    "ArcWorld",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the public example API without importing torch eagerly."""
    exports: dict[str, object]
    if name in {"build_agent", "run_training"}:
        from .runner import build_agent, run_training

        exports = {
            "build_agent": build_agent,
            "run_training": run_training,
        }
    elif name == "GymWorld":
        from .world import GymWorld

        exports = {"GymWorld": GymWorld}
    elif name in {
        "DescribedGymWorld",
        "CARTPOLE_WORLD_DESCRIPTION",
        "WorldDescription",
    }:
        from .world_embedded import (
            CARTPOLE_WORLD_DESCRIPTION,
            DescribedGymWorld,
            WorldDescription,
        )

        exports = {
            "DescribedGymWorld": DescribedGymWorld,
            "CARTPOLE_WORLD_DESCRIPTION": CARTPOLE_WORLD_DESCRIPTION,
            "WorldDescription": WorldDescription,
        }
    elif name == "ArcWorld":
        from .world_arc import ArcWorld

        exports = {"ArcWorld": ArcWorld}
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals().update(exports)
    return globals()[name]
