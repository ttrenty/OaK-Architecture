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

This repository currently provides:

- shared types such as `TimeStep`, `Transition`, `PlanningUpdate`, and
  `AgentStepResult`
- abstract interfaces for the main OaK components
- the package's official `OaKAgent` coordinator that wires those components
  together
- a minimal external example implementation used as a smoke test

## Documentation

Project documentation is published at:

- [GitHub Pages documentation](https://ttrenty.github.io/OaK-Architecture/)

The docs include:

- the API reference for `oak_architecture`
- the architecture guide embedded directly into that API page
- rendered diagrams for the OaK control flow and component layout

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

- `pixi run test`
  Install the package in editable mode and run every Python test script in
  `tst/`.
- `pixi run docs`
  Generate the API documentation site in `docs/api/`.
- `pixi run docs_api`
  Compatibility alias for `pixi run docs`.
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
- `examples/`
  Repository-level example implementations that use the package as an external
  consumer would.
- `tst/`
  Runnable test scripts. `pixi run test` executes every `*.py` file in this
  directory tree.
- `docs/`
  Documentation sources, diagrams, API-doc templates, and generated API docs.

## Working in this repository

If you want to prototype a concrete implementation in this repository, place it
under `examples/` and add checks under `tst/`. Running `pixi run test` will
pick up any new Python test file automatically.
