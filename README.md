# OaK Architecture

This repository now contains a source-grounded interface proposal for Rich Sutton's OaK architecture, based on the materials in [`ressources/`](./ressources) and a comparison of the two AI-generated deep research reports.

The convergence decision is simple:

- Keep the ChatGPT report's source discipline, modular decomposition, FC-STOMP pipeline, GVF framing, and utility-driven curation loop.
- Keep the Gemini report's practical focus on prototype milestones, evaluation, online normalization, and the need for pragmatic shortcuts.
- Do not bake deep-network choices, average-reward-only assumptions, or long-range intelligence-amplification claims into the core API.

Key files:

- [Comparison and convergence note](./docs/oak-convergence.md)
- [Mermaid architecture diagram](./docs/diagrams/oak_architecture.mmd)
- [PlantUML component diagram](./docs/diagrams/oak_architecture.puml)
- [PlantUML runtime sequence](./docs/diagrams/oak_runtime_sequence.puml)
- [Core interfaces](./src/oak_architecture/interfaces.py)
- [Reference agent wiring](./src/oak_architecture/agent.py)
- [Shared datatypes](./src/oak_architecture/types.py)

The resulting package is intentionally interface-first. It captures the contracts for continual learning, FC-STOMP abstraction building, option modeling, planning, per-weight meta step sizes, and utility-based pruning, while keeping the actual learning algorithms replaceable.

## Generate the diagrams

Use the `pixi` tasks if you want all of the SVGs regenerated:

```bash
pixi run render_diagrams
```

You can also run the diagram generators individually:

```bash
pixi run render_mermaid
pixi run render_plantuml
```

The Mermaid command uses [`docs/diagrams/puppeteer-config.json`](./docs/diagrams/puppeteer-config.json) so it also works in headless Linux environments and CI.

## Rendered diagrams

### Mermaid architecture overview

![Mermaid architecture overview](./docs/img/oak_architecture_marmaid.svg)

### PlantUML component view

![PlantUML component view](./docs/img/oak_architecture.svg)

### PlantUML runtime sequence

![PlantUML runtime sequence](./docs/img/oak_runtime_sequence.svg)
