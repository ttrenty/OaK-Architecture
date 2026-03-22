# OaK Architecture

`oak-architecture` is an interface-first Python package for experimenting with
the OaK architecture vision associated with Richard Sutton.

The repository focuses on two things:

- a small, typed core package that defines the shared data structures,
  component interfaces, and the canonical `OaKAgent` step loop
- external example implementations that show how a separate project can build
  on top of those interfaces

The goal is to make comparative implementation work **possible**. The package
provides the contracts and runtime wiring; concrete learning systems can live
outside the package and evolve independently.

## Current scope

The published package is intentionally interface-only. Concrete examples in
this repository live under `examples/` so they reflect how downstream users can
implement the architecture in practice.

The package exposes **two abstraction levels**:

- the default **four-interface layer:** `OaKAgent` plus the four main OaK interfaces:
  `Perception`, `TransitionModel`, `ValueFunction`, and `ReactivePolicy`.
  This is the simplest way to use the package and the main conceptual surface.
- the optional **fine-grained layer:** `oak_architecture.fine_grained`, which breaks those four slots into smaller
  building blocks and provides `Composite*` implementations for wiring them
  back into the main agent.

In other words, you can either:

- implement the four main interfaces directly
- or work one level lower and assemble those interfaces from finer-grained
  parts

This repository currently provides:

- shared types such as `TimeStep`, `Transition`, `PlanningUpdate`, and
  `AgentStepResult`
- abstract interfaces for the four main OaK components
- a `World` protocol that environments must implement for use with
  `OaKAgent.train()`
- an optional `oak_architecture.fine_grained` submodule with lower-level
  building blocks and `Composite*` implementations
- the package's official `OaKAgent` coordinator that wires those components
  together, including a built-in `train()` method for running the standard
  episode loop on any `World`
- two minimal external example implementations used as smoke tests:
  one direct and one fine-grained
- a full learning agent applied to CartPole that exercises all OaK machinery
  (see below)

## CartPole example

`examples/cartpole/` contains a full OaK agent that learns to solve
CartPole-v1 using all four interfaces. It demonstrates the entire OaK
lifecycle: discovery, LLM-augmented perception, Option-Critic temporal
abstraction, Dyna-Q model-based planning, GVF auxiliary predictions, and
utility-based curation.

The agent modules are environment-agnostic. To apply the same agent to a
different RL problem, implement a new `World` and pass it to `run_training()`.

### Two config modes

The config mode is chosen automatically based on the world you pass in:

| World class | Config source | Discovery? | LLM? |
|---|---|---|---|
| `CartPoleWorld` (no `description`) | Trial-and-error probing | Yes | Optional |
| `DescribedCartPoleWorld` (has `description`) | `WorldDescription` attribute | No | No |

```python
from examples.cartpole import CartPoleWorld, DescribedCartPoleWorld, run_training

# Discovery mode: agent discovers everything through trial-and-error
run_training(CartPoleWorld(), num_episodes=1000, solved_threshold=475.0)

# Embedded mode: world description provides obs/action metadata directly
run_training(DescribedCartPoleWorld(), num_episodes=1000, solved_threshold=475.0)
```

**Discovery mode** (world without `description`): the agent probes the
world with trial-and-error actions to discover observation type/shape and the
action space, then optionally consults an LLM for feature analysis.

**Embedded mode** (world with `description`): observation shape, action count,
encoder type, and feature descriptions are read directly from the world's
`WorldDescription`.  This skips discovery and LLM calls entirely, making
startup instant and training deterministic from step one.

### How it works

1. **Config**: obtain observation/action space info from the world (either
   auto-discovered or read from its `description` attribute).
2. **Build**: the agent is assembled from four modules:
   - `AdaptivePerception`: encodes observations, manages features/subtasks
   - `OptionValueFunction`: DQN-style Q_Omega over option slots + GVF heads
   - `DynaTransitionModel`: learned world model with imagined rollouts
   - `OptionCriticPolicy`: per-option DQN Q-networks + learned termination
3. **Train**: call ``agent.train(world)`` which runs the standard OaK 6-phase
   step loop (perceive, learn, grow, plan, act, maintain) for the configured
   number of episodes. The world must implement the ``World`` protocol from
   ``oak_architecture.interfaces``.

### Running

```bash
# Discovery mode (1000 episodes)
pixi run python tst/run_cartpole.py

# Embedded mode (no discovery, no LLM)
pixi run python tst/run_cartpole_embedded.py

# Custom training
pixi run python -c "
from examples.cartpole import DescribedCartPoleWorld, run_training
import torch

def log_episode(episode, reward, avg_reward, agent):
    if episode % 10 == 0:
        print(f'episode={episode} reward={reward:.1f} avg={avg_reward:.1f}')

run_training(
    DescribedCartPoleWorld(),
    num_episodes=1000,
    solved_threshold=475.0,
    planning_budget=5,
    episode_logger=log_episode,
    device=torch.device('cuda'),
)
"

# Fast component tests only (seconds, no full training)
pixi run python -c "
import sys; sys.path.insert(0, '.')
from tst.debug_cartpole import test_sanity, test_value_function, test_q_equivalence
test_sanity()
test_value_function()
test_q_equivalence()
"
```

### Ollama setup

The LLM analysis step calls ollama at `http://172.26.64.1:11434` (WSL2
host gateway). To use a different host, edit `_get_ollama_url()` in
`examples/cartpole/llm.py`. If ollama is unreachable, the agent falls
back to heuristic feature/encoder selection and still trains normally.

To run the dedicated live connectivity check:

```bash
pixi run test_llm_connection
# optional overrides
OLLAMA_HOST=http://localhost:11434 OAK_LLM_MODEL=qwen3.5:9b pixi run test_llm_connection
```

### Hyperparameters

The main knobs to tune, organized by module:

**`run_training()` in `runner.py`**

| Parameter | Default | Description |
|---|---|---|
| `world` | (required) | A `World` implementation to train on |
| `num_episodes` | 500 | Total training episodes |
| `average_window` | 100 | Window size used for rolling-average tracking |
| `solved_threshold` | `None` | Early-stop when the `average_window` average reaches this |
| `planning_budget` | 5 | Dyna-Q rollouts per step (0 = disable planning) |
| `ollama_model` | `"qwen3.5:9b"` | Ollama model for feature analysis (discovery mode only) |
| `train_encoder` | `False` | Whether to train the encoder (identity encoder has no params) |
| `episode_logger` | `None` | Optional callback `(episode, reward, avg_reward, agent)` for user-owned per-episode logging |

**`build_agent()` in `runner.py`**

| Parameter | Default | Description |
|---|---|---|
| `feature_budget` | 2 | Features processed per step (= number of options created) |

**`OptionCriticPolicy` in `reactive_policy.py`**

| Parameter | Default | Description |
|---|---|---|
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Minimum exploration rate |
| `epsilon_decay_steps` | 5000 | Steps for linear epsilon decay |
| `lr` | 1e-3 | Learning rate for option Q-networks |
| `gamma` | 0.99 | Discount factor |
| `buffer_capacity` | 5000 | Replay buffer size for option Q-learning |
| `batch_size` | 64 | Mini-batch size for DQN updates |

**`OptionValueFunction` in `value_function.py`**

| Parameter | Default | Description |
|---|---|---|
| `lr` | 1e-3 | Learning rate for Q_Omega |
| `buffer_capacity` | 5000 | Replay buffer size for Q_Omega |
| `target_sync_freq` | 200 | Hard target network sync interval |
| `max_options` | 8 | Maximum number of option slots |

**`DynaTransitionModel` in `transition_model.py`**

| Parameter | Default | Description |
|---|---|---|
| `lr` | 1e-3 | Learning rate for world model |
| `buffer_capacity` | 5000 | World model training buffer |
| `model_train_batch` | 32 | Batch size for world model training |

### Module layout

```
examples/cartpole/
  __init__.py            # public API exports
  runner.py              # build_agent() + run_training() orchestration

  # ── Agent modules (environment-agnostic, reusable with any World) ──
  encoders.py            # Identity, MLP, CNN encoder architectures
  perception.py          # Adaptive perception (pluggable encoder)
  value_function.py      # Q_Omega + GVFs + utility/curation
  transition_model.py    # Dyna-Q world model + planning
  reactive_policy.py     # Option-Critic (per-option DQN + termination)
  discovery.py           # Trial-and-error observation/action space discovery
  llm.py                 # Ollama REST API for feature analysis

  # ── CartPole-specific World implementations ──
  world.py               # Opaque CartPole wrapper (triggers discovery mode)
  world_embedded.py      # CartPole wrapper with WorldDescription metadata

tst/
  debug_cartpole.py      # Targeted component tests
  run_cartpole.py        # Training with discovery (CartPoleWorld)
  run_cartpole_embedded.py  # Training with embedded info (DescribedCartPoleWorld)
```

### Known limitations

DQN on CartPole exhibits inherent instability: the agent typically peaks
at avg 340-380 reward then experiences periodic performance drops due to
catastrophic forgetting in the replay buffer. The agent recovers from
crashes given enough episodes. This is a well-known DQN property, not
specific to the OaK architecture. Possible mitigations to experiment with:
increasing `epsilon_end` to 0.05 (more stable but lower peak),
Polyak averaging for target networks (code exists in
`_OptionNetworks.soft_update_target()`), or Double DQN (select action
with online network, evaluate with target network).

## Documentation

Project documentation is published at:

- [GitHub Pages documentation](https://ttrenty.github.io/OaK-Architecture/)

The docs include:

- the API reference for `oak_architecture`
- the architecture guide embedded directly into that API page
- rendered diagrams for the default four-interface view, the fine-grained
  slot map, and the runtime call paths

## Development

### Environment setup

This project uses `pixi` for dependency management and task execution.

Install `pixi` by following the official instructions:

- [Pixi installation guide](https://pixi.sh/latest/)

On Unix-like systems, one common installation method is:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
# or with wget instead
wget -qO- https://pixi.sh/install.sh | sh
```

Then install the project environment from the repository root:

```bash
pixi install
```

### Common tasks

- `pixi run tests`
  Install the package in editable mode and run every Python test script in
  `tst/`.
- `pixi run test_llm_connection`
  Run the live Ollama smoke test that verifies the CartPole LLM helper can
  reach the configured model and parse a structured response.
- `pixi run docs`
  Generate the API documentation site in `docs/api/`.
- `pixi run render_diagrams`
  Regenerate the rendered PlantUML diagrams used by the docs.
- `pixi run build_package`
  Build the source distribution and wheel in `dist/`.

A `Makefile` is also provided for convenience, but it only forwards to
`pixi run` commands.

## Repository layout

- `src/oak_architecture/`
  Core package with shared types, interface definitions, and the canonical
  `OaKAgent` execution loop.
- `src/oak_architecture/fine_grained/`
  Optional lower-level interfaces and `Composite*` implementations for
  projects that want to swap internal building blocks independently.
- `examples/`
  Repository-level example implementations that use the package as an external
  consumer would, including `minimal_oak.py`,
  `minimal_oak_fine_grained.py`, and the full `cartpole/` agent.
- `tst/`
  Runnable test scripts. `pixi run tests` executes every `*.py` file in this
  directory tree. `pixi run test_llm_connection` runs the dedicated live
  Ollama connectivity check.
- `docs/`
  Documentation sources, diagrams, API-doc templates, and generated API docs.

## Working in this repository

If you want to prototype a concrete implementation in this repository, place it
under `examples/` and add checks under `tst/`. Running `pixi run tests` will
pick up any new Python test file automatically.
