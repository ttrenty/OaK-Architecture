# Implementation Guide

This project is designed to make OaK implementable in small increments. The
interfaces are intentionally split so you can start from a simple agent and add
more machinery as your implementation matures.

## Core agent view

At the highest level, an OaK agent is organized around four blocks:

- `Perception`
- `ReactivePolicy`
- `ValueFunction`
- `TransitionModel`

`Perception` converts the observation stream into the agent `State`. The other
three blocks operate on that state.

## Additional OaK mechanisms

The package also defines interfaces for mechanisms that support a richer OaK
agent:

- feature construction and ranking
- subtask generation
- option learning and option models
- planning updates
- utility assessment
- curation and pruning
- meta step-size adaptation

These parts let you grow an implementation from a minimal continual-learning
agent into one that can manage its own representational and behavioral
machinery.

## What a real implementation still needs

The minimal implementation proves the interface wiring, but a serious OaK
system still needs:

- a meaningful `State` representation inside `Perception`
- actual GVFs or other predictive targets inside `ValueFunction`
- an action and option model inside `TransitionModel`
- a planner that uses model predictions under a bounded compute budget
- a utility accounting scheme that measures whether learned structures are
  worth keeping
- a curation policy that can add, retain, or remove structures safely

## Recommended implementation order

1. Implement `World` and `Perception`.
2. Implement a small `ReactivePolicy` and one `ValueFunction`.
3. Add a simple one-step `TransitionModel`.
4. Plug them into `OaKAgent` and verify the smoke path.
5. Add feature construction, subtasks, and options.
6. Add planning updates.
7. Add utility assessment and curation.

## Design advice

- Keep `State` concrete and domain-specific.
- Keep `ReactivePolicy` simple at first. It should choose actions from state,
  not solve every learning problem itself.
- Start with one useful prediction in `ValueFunction` before adding a bank of
  auxiliary signals.
- Make `TransitionModel` honest about uncertainty. A poor model is often worse
  than no model if planning uses it aggressively.
- Add curation only after you can measure usage or utility in a stable way.

## Where to look in the code

- `oak_architecture.interfaces`
- `oak_architecture.types`
- `oak_architecture.agent`
- `oak_architecture.implementations.minimal_oak`
