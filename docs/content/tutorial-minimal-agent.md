# Tutorial: Minimal Agent

This tutorial shows how to use the smallest runnable implementation in the
repository.

## Run the smoke example

```bash
pixi run test
```

That command installs the package in editable mode and runs the small smoke
check from `tst/run_minimal_oak.py`.

## Import the minimal implementation

```python
from oak_architecture.implementations.minimal_oak import (
    build_minimal_agent,
    run_minimal_episode,
)

agent = build_minimal_agent()
trace = run_minimal_episode(horizon=5)
```

`run_minimal_episode()` returns a compact trace containing:

- the current `State`
- the chosen primitive action
- the active option id, if any
- the subtasks created at that step
- the planning budget reported by the planner

## What the minimal example is doing

The minimal implementation provides:

- a toy `World` with integer observations
- a `Perception` module that maps observations directly to `State`
- one feature and one generated subtask
- one trivial option
- no-op model learning and curation
- a tiny policy and planner

This is enough to prove that the package interfaces can be instantiated and run
through a full OaK step loop.

## How to turn it into your own implementation

Start by replacing these parts:

1. Replace `MinimalWorld` with your environment.
2. Replace `MinimalPerception` with a domain-specific state builder.
3. Replace `MinimalValueFunction` with real predictive learning.
4. Replace `MinimalTransitionModel` and `MinimalPlanner` with a model and
   search procedure that actually matter in your domain.

The minimal implementation is best used as a structural example, not as an
algorithmic baseline.
