"""Repository-level example implementations built on top of `oak_architecture`."""

from .smoke.minimal_oak import (
    build_minimal_agent,
    run_minimal_episode,
    run_minimal_training,
)
from .smoke.minimal_oak_fine_grained import (
    build_minimal_agent as build_minimal_fine_grained_agent,
)
from .smoke.minimal_oak_fine_grained import (
    run_minimal_episode as run_minimal_fine_grained_episode,
)

__all__ = [
    "build_minimal_agent",
    "run_minimal_episode",
    "run_minimal_training",
    "build_minimal_fine_grained_agent",
    "run_minimal_fine_grained_episode",
]
