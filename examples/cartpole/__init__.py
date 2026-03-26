"""OaK agent with LLM-augmented perception, Option-Critic, and Dyna-Q planning.

This example demonstrates a full OaK agent using Option-Critic temporal
abstraction with Dyna-Q planning.  The default world is CartPole-v1, but the
agent modules are environment-agnostic: swap the `World` implementation to
apply the same agent to a different RL problem.

Two config modes are supported (chosen automatically by `run_training`):

- **Embedded**: the world exposes a `description` attribute with
  observation/action space metadata, skipping the discovery phase.
- **Discovery**: the agent discovers observation/action spaces through
  trial-and-error, optionally refined by an LLM via ollama.

Both modes share the same learning modules (perception, value function,
transition model, reactive policy).  Only the config-acquisition step differs.
"""

from .runner import build_agent, run_training
from .world import CartPoleWorld
from .world_embedded import DescribedCartPoleWorld, WorldDescription

__all__ = [
    "build_agent",
    "run_training",
    "CartPoleWorld",
    "DescribedCartPoleWorld",
    "WorldDescription",
]
