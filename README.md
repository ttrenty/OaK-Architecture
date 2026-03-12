# OaK Architecture

Python interfaces, diagrams, and a minimal runnable reference for experimenting
with Richard Sutton's OaK architecture.

This repository is organized around three goals:

- make the main OaK components explicit and easy to implement
- provide a package that can be imported, extended, and tested
- keep a very small smoke implementation that proves the interfaces fit
  together end to end

## What is in this repository

- `src/oak_architecture/`
  Core package with shared types, interface definitions, the reference
  `OaKAgent` execution loop, and a minimal implementation under
  `oak_architecture.implementations`.
- `docs/`
  Documentation sources, diagrams, API-doc templates, and generated API docs.
- `ressources/`
  Background material and source notes.
- `tst/`
  Runnable smoke checks for the package.

## Architecture at a glance

The project keeps the four main OaK blocks visible:

- `Perception`
- `ReactivePolicy`
- `ValueFunction`
- `TransitionModel`

`Perception` produces the agent `State`, and the other three blocks operate on
that state. The package also includes interfaces for feature construction,
subtasks, options, planning, utility assessment, and curation so a fuller OaK
agent can be built incrementally.

## Quick start

Install the project environment with `pixi`, then use:

```bash
pixi run test
pixi run docs_api
pixi run render_diagrams
pixi run build_package
```

These tasks do the following:

- `pixi run test`
  Install the package in editable mode and run the minimal smoke example in
  `tst/run_minimal_oak.py`.
- `pixi run docs_api`
  Generate the documentation site into `docs/api/`. This task depends on
  `render_diagrams`, so the authored guide pages and diagram assets are
  created together.
- `pixi run render_diagrams`
  Regenerate the PlantUML SVG diagrams in `docs/api/img/`.
- `pixi run build_package`
  Build the source distribution and wheel in `dist/`.

## Using the package

Minimal example:

```python
from oak_architecture.implementations.minimal_oak import run_minimal_episode

trace = run_minimal_episode(horizon=5)
for step in trace:
    print(step["action"], step["state"])
```

The minimal implementation is intentionally small. It is meant to show how the
interfaces connect, not to serve as a finished learning system.

## Documentation

Start with these files:

- [Source Markdown guides](./docs/content/)
- [Generated API docs](./docs/api/index.html)
- [Hosted API docs](https://ttrenty.github.io/OaK-Architecture/)
- [Hosted overview](https://ttrenty.github.io/OaK-Architecture/overview.html)
- [Hosted implementation guide](https://ttrenty.github.io/OaK-Architecture/implementation-guide.html)
- [Hosted minimal agent tutorial](https://ttrenty.github.io/OaK-Architecture/tutorial-minimal-agent.html)

Key diagrams:

- [High-level PlantUML source](./docs/diagrams/oak_core.puml)
- [Detailed PlantUML source](./docs/diagrams/oak_architecture.puml)
- [Runtime sequence source](./docs/diagrams/oak_runtime_sequence.puml)
- [Hosted rendered core diagram](https://ttrenty.github.io/OaK-Architecture/img/oak_core.svg)
- [Hosted rendered architecture diagram](https://ttrenty.github.io/OaK-Architecture/img/oak_architecture.svg)
- [Hosted rendered runtime sequence](https://ttrenty.github.io/OaK-Architecture/img/oak_runtime_sequence.svg)

The repository also includes a GitHub Pages deployment workflow for the API
docs in [.github/workflows/deploy-docs.yml](./.github/workflows/deploy-docs.yml).
After enabling Pages with the `GitHub Actions` source in the repository
settings, pushes to `main` will run `pixi run docs_api`, generate the API docs
and rendered diagrams, and publish them to the hosted URL above.

## Project status

The repository is currently interface-first:

- the package structure and typing are in place
- the `OaKAgent` loop is wired end to end
- the minimal implementation runs as a smoke test
- major learning components still need real implementations

The main missing work is in the concrete algorithms: perception learning, value
learning, transition modeling, option learning, planning control, utility
assessment, and curation policy.
