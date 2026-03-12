"""Runnable reference implementations shipped with the package.

These implementations are intentionally small. They are provided so the package
can be executed, inspected, and used as a starting point for real
implementations.

The current `minimal_oak` module is best treated as a tutorial or smoke test,
not as a competitive baseline.
"""

from .minimal_oak import build_minimal_agent, run_minimal_episode

__all__ = ["build_minimal_agent", "run_minimal_episode"]
