# OaK Architecture

`oakagent` is an interface-first Python package for experimenting with
the OaK architecture vision of Richard Sutton ([http://incompleteideas.net/](http://incompleteideas.net/)).

Install the published distribution with `pip install oakagent`, then import it
in code as `import oak`.

The repository focuses on two things:

- a small, typed core package that defines the shared data structures,
  component interfaces, and the concrete class `OaKAgent` that uses those interfaces.
- external example implementations that show how a separate project can build
  on top of those interfaces

The goal is to make comparative implementation work **possible**. The package
provides the contracts and runtime wiring; concrete learning systems can live
outside the package and evolve independently.

## Documentation

Project documentation is published at:

- [GitHub Pages documentation](https://ttrenty.github.io/OaK-Architecture/)

The docs include:

- the API reference for `oakagent` (Use `import oak` in your code)
- the architecture guide embedded directly into that API page
- rendered diagrams for the default four-interface view, the fine-grained
  slot map, and the runtime call paths

## Current scope

The published package is intentionally interface-focused. Concrete examples in
this repository live under `examples/` so they reflect how downstream users can
implement the architecture in practice.

The package exposes **two abstraction levels**:

- the default **four-interface layer:** `OaKAgent` plus the four main OaK interfaces:
  `Perception`, `TransitionModel`, `ValueFunction`, and `ReactivePolicy`.
  This is the simplest way to use the package and the main conceptual surface.
- the optional **fine-grained layer:** `oak.fine_grained`, which breaks those four slots into smaller
  building blocks and provides `Composite*` implementations for wiring them
  back into the main agent.

In other words, you can either:

- implement the four main interfaces directly
- or work one level lower and assemble those interfaces from finer-grained
  parts

This repository currently provides:

- abstract interfaces for the four main OaK components
- a `World` protocol that environments must implement for use with
  `OaKAgent.train()`
- the package's official `OaKAgent` coordinator that wires those components
  together, including a built-in `train()` method for running the standard
  episode loop on any `World`
- two minimal external example implementations used as smoke tests:
  one direct and one fine-grained
- a full learning agent example for RL worlds, demonstrated on Gymnasium Environment and ARC-AGI-3 (see below)

## Example 01

`examples/example_01/` contains the first full OaK agent example for reinforcement-learning
worlds. CartPole-v1 is the bundled sample world, but the implementation is
not named after any specific environment. It demonstrates the entire OaK
lifecycle: discovery, LLM-augmented perception, Option-Critic temporal
abstraction, Dyna-Q model-based planning, GVF auxiliary predictions, and
utility-based curation.

The agent modules are environment-agnostic. To apply the same agent to a
different RL problem, implement a new `World` and pass it to `run_training()`.

### Two config modes

The config mode is chosen automatically based on the world you pass in:

| World class | Config source | Discovery? | LLM? |
|---|---|---|---|
| `GymWorld("CartPole-v1")` (no `description`) | Trial-and-error probing | Yes | Optional |
| `DescribedGymWorld("CartPole-v1")` (has `description`) | `WorldDescription` attribute | No | No |

```python
from examples.example_01 import DescribedGymWorld, GymWorld, run_training

# Discovery mode: agent discovers everything through trial-and-error
run_training(GymWorld("CartPole-v1"), num_episodes=1000, solved_threshold=475.0)

# Described mode: world description provides obs/action metadata directly
run_training(
    DescribedGymWorld("CartPole-v1"),
    num_episodes=1000,
    solved_threshold=475.0,
)
```

**Discovery mode** (world without `description`): the agent probes the
world with trial-and-error actions to discover observation type/shape and the
action space, then optionally consults an LLM for feature analysis.

**Described mode** (world with `description`): observation shape, action count,
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
   ``oak.interfaces``.

### Ollama setup

The LLM analysis step calls ollama at `http://172.26.64.1:11434` (WSL2
host gateway). To use a different host, edit `_get_ollama_url()` in
`examples/example_01/llm.py`. If ollama is unreachable, the agent falls
back to heuristic feature/encoder selection and still trains normally.

To run the dedicated live connectivity check:

```bash
pixi run test_llm_connection
# optional overrides
OLLAMA_HOST=http://localhost:11434 OAK_LLM_MODEL=qwen3.5:9b pixi run test_llm_connection
```

### Running

```bash
# Smoke-only checks in the default environment
pixi run tests

# Example 01 CartPole runners require a torch-enabled environment.
# Use `linux-gpu` on Linux GPU systems or `macos` on Apple Silicon.
pixi run -e linux-gpu test_example_01_cartpole
pixi run -e linux-gpu test_example_01_cartpole_described
pixi run -e linux-gpu test_example_01_integration

# Fast component tests only (seconds, no full training)
pixi run -e linux-gpu test_debug_example_01_cartpole
```

### ARC-AGI-3 benchmark

The ARC benchmark now supports explicit ARC Prize API configuration and a
development-oriented local pretraining pass.

Important distinction:

- `ARC_API_KEY` is for the ARC Prize environment API / scorecards ([https://docs.arcprize.org/api-keys](https://docs.arcprize.org/api-keys)).
- It does **not** provide LLM inference. The current ARC benchmark does not use
  the `example_01` Ollama feature-analysis path for action selection.

Recommended local benchmark run:

```bash
OPERATION_MODE=offline \
OAK_ARC_PRETRAIN_EPISODES=12 \
pixi run -e linux-gpu benchmark_arc_agi
```

Run against the hosted ARC service with your API key:

```bash
export ARC_API_KEY="your-api-key-here"
OAK_ARC_OPERATION_MODE=online \
OAK_ARC_PRETRAIN_EPISODES=12 \
pixi run -e linux-gpu benchmark_arc_agi
```

Useful ARC-specific knobs:

- `OAK_ARC_OPERATION_MODE=offline|normal|online`
- `ARC_API_KEY` or `OAK_ARC_API_KEY`
- `OAK_ARC_PRETRAIN_EPISODES=12` to warm up on local copies before the scored run
- `OAK_ARC_TRAIN_ENCODER=1` to train the CNN encoder instead of using frozen random features
- `OAK_ARC_PLANNING_WARMUP=32` so Dyna planning can activate within short ARC episodes
- `OAK_ARC_GREEDY_EVAL=1` to disable epsilon exploration for the scored pass

### Hyperparameters

The main knobs to tune, organized by module:

---

**`run_training()` in `runner.py`**

| Parameter | Default | Description |
|---|---|---|
| `world` | (required) | A `World` implementation to train on |
| `num_episodes` | 500 | Total training episodes |
| `average_window` | 100 | Window size used for rolling-average tracking |
| `solved_threshold` | `None` | Early-stop when the `average_window` average reaches this |
| `planning_budget` | 5 | Dyna-Q rollouts per step (0 = disable planning) |
| `planning_warmup_steps` | 500 | Number of real transitions before planning activates |
| `ollama_model` | `"qwen3.5:9b"` | Ollama model for feature analysis (discovery mode only) |
| `train_encoder` | `False` | Whether to train the encoder (identity encoder has no params) |
| `episode_logger` | `None` | Optional callback `(episode, reward, avg_reward, agent)` for user-owned per-episode logging |

---

**`build_agent()` in `runner.py`**

| Parameter | Default | Description |
|---|---|---|
| `feature_budget` | 2 | Features processed per step (= number of options created) |

--- 

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

---

**`OptionValueFunction` in `value_function.py`**

| Parameter | Default | Description |
|---|---|---|
| `lr` | 1e-3 | Learning rate for Q_Omega |
| `buffer_capacity` | 5000 | Replay buffer size for Q_Omega |
| `target_sync_freq` | 200 | Hard target network sync interval |
| `max_options` | 8 | Maximum number of option slots |

---

**`DynaTransitionModel` in `transition_model.py`**

| Parameter | Default | Description |
|---|---|---|
| `lr` | 1e-3 | Learning rate for world model |
| `buffer_capacity` | 5000 | World model training buffer |
| `model_train_batch` | 32 | Batch size for world model training |

### Module layout

```
examples/example_01/
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

  # ── Gym World wrappers ──
  world.py               # Opaque gym wrapper (triggers discovery mode)
  world_embedded.py      # Described gym wrapper + bundled CartPole metadata

tests/
  debug_example_01_cartpole.py         # Targeted Example 01 component tests
  run_example_01_cartpole.py           # Training with discovery on CartPole-v1
  run_example_01_cartpole_described.py # Training with described CartPole-v1 metadata
  test_example_01_integration.py       # Example 01 integration checks
```

### Known limitations

When trained on CartPole, DQN exhibits inherent instability: the agent typically peaks
at avg 340-380 reward then experiences periodic performance drops due to
catastrophic forgetting in the replay buffer. The agent recovers from
crashes given enough episodes. This is a well-known DQN property, not
specific to the OaK architecture. Possible mitigations to experiment with:
increasing `epsilon_end` to 0.05 (more stable but lower peak),
Polyak averaging for target networks (code exists in
`_OptionNetworks.soft_update_target()`), or Double DQN (select action
with online network, evaluate with target network).

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
  Run the default smoke checks without installing the package in editable mode.
- `pixi run test_llm_connection`
  Run the live Ollama smoke test that verifies the Example 01 LLM helper can
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

- `src/oak/`
  Core package with shared types, interface definitions, and the canonical
  `OaKAgent` execution loop.
- `src/oak/fine_grained/`
  Optional lower-level interfaces and `Composite*` implementations for
  projects that want to swap internal building blocks independently.
- `examples/`
  Repository-level example implementations that use the package as an external
  consumer would, including `minimal_oak.py`,
  `minimal_oak_fine_grained.py`, and the full `example_01/` example.
- `tests/`
  Runnable test scripts and example entrypoints. `pixi run tests` covers the
  default smoke path, while the `test_example_01_*` tasks exercise the
  torch-backed example under a torch-enabled Pixi environment.
- `docs/`
  Documentation sources, diagrams, API-doc templates, and generated API docs.

## Working in this repository

If you want to prototype a concrete implementation in this repository, place it
under `examples/` and add checks under `tests/`.
