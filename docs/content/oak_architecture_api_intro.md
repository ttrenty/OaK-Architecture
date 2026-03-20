---

# Architecture Guide

## Diagram Gallery

<div class="oak-diagram-grid">
  <figure class="oak-diagram-card">
    <img src="img/oak_core.svg" alt="Main four-interface OaK view showing Perception, ValueFunction, TransitionModel, and ReactivePolicy." loading="lazy">
    <figcaption><strong>`oak_core`.</strong> The default conceptual slot map: OaKAgent coordinating the four main interfaces and their main data flow.</figcaption>
  </figure>
  <figure class="oak-diagram-card">
    <img src="img/oak_architecture.svg" alt="Fine-grained slot map showing Composite modules, their delegated interfaces, and associated optional interfaces." loading="lazy">
    <figcaption><strong>`oak_architecture`.</strong> The fine-grained slot map: Composite modules plus the lower-level interfaces available inside each slot.</figcaption>
  </figure>
  <figure class="oak-diagram-card">
    <img src="img/oak_runtime_overview.svg" alt="Simplified runtime sequence for the six phases of OaKAgent.step(...)." loading="lazy">
    <figcaption><strong>`oak_runtime_overview`.</strong> The top-level step path through the four main interfaces for the six phases: Perceive, Learn, Grow, Plan, Act, Maintain.</figcaption>
  </figure>
  <figure class="oak-diagram-card">
    <img src="img/oak_runtime_sequence.svg" alt="Detailed runtime sequence showing the fine-grained interfaces actually touched during one OaKAgent step through Composite modules." loading="lazy">
    <figcaption><strong>`oak_runtime_sequence`.</strong> The detailed composite-wired step path: <code>OaKAgent -&gt; Composite* -&gt; fine_grained interface used during that step</code>.</figcaption>
  </figure>
</div>

## What You Must Implement

`OaKAgent` is the canonical coordinator.  It is composed of exactly four
objects, one per Sutton module:

- `perception`
  Implements `Perception`.  It receives raw environment data and must return
  the current `subjective_state`.  It also manages feature discovery, ranking,
  and subtask generation.
- `transition_model`
  Implements `TransitionModel`.  It learns from transitions, maintains option
  models, and runs bounded planning using the world model and value function.
- `value_function`
  Implements `ValueFunction`.  It learns from `Transition` objects, predicts
  values, tracks utility of learned structures, and produces curation decisions.
- `reactive_policy`
  Implements `ReactivePolicy`.  It selects actions (primitive or options),
  manages the option library and option learning, and integrates planning
  updates.

You also configure scalar controls:

- `planning_budget`
- `feature_budget`
- `option_stop_threshold`

`OaKAgent` manages these runtime fields itself:

- `last_action`
- `last_subjective_state`

Your environment loop does not have to implement `World`, but it must supply
`TimeStep` objects to `OaKAgent.step(...)`.  `World` is just the package's
small protocol for one way to do that.

## Two Ways to Implement

**Direct approach** — implement the four main interfaces directly.  Each of
your classes is a self-contained module.  This is the simplest path and what
the `examples/minimal_oak.py` example demonstrates.

**Composite approach** — use the fine-grained component interfaces from
`oak_architecture.fine_grained.components` and wire them together using the
composites from `oak_architecture.fine_grained.composites`.  This is for
projects that need to independently swap building blocks inside a module
(e.g. replace the planner without touching the world model).  The
`examples/minimal_oak_fine_grained.py` example demonstrates this path with
the same toy behavior as the direct example.

| Main interface    | Composite class              | Fine-grained building blocks                                                      |
|-------------------|------------------------------|-----------------------------------------------------------------------------------|
| `Perception`      | `CompositePerception`        | `StateBuilder`, `FeatureBank`, `FeatureConstructor`, `FeatureRanker`, `SubtaskGenerator` |
| `TransitionModel` | `CompositeTransitionModel`   | `WorldModel`, `OptionModelLearner`, `OptionModel`, `Planner`                      |
| `ValueFunction`   | `CompositeValueFunction`     | `ValueEstimator`, `GeneralValueFunctionLearner`, `UtilityAssessor`, `Curator`, `MetaStepSizeLearner` |
| `ReactivePolicy`  | `CompositeReactivePolicy`    | `ActionSelector`, `Option`, `OptionLibrary`, `OptionLearner`, `OptionKeyboard` (optional) |

## Diagram-to-Code Mapping

The diagrams have different jobs, but they all describe the same
implementation:

- `oak_core`
  The default conceptual slot map: OaKAgent plus the four main interfaces
  and the main data flow between them.
- `oak_architecture`
  The fine-grained slot map: Composite modules, their delegated interfaces,
  and associated optional interfaces from `oak_architecture.fine_grained.components`.
- `oak_runtime_overview`
  The top-level phase-by-phase sequence at the four-interface layer.
- `oak_runtime_sequence`
  The composite-wired per-step call order, showing only the fine-grained
  interfaces actually touched during one `step(...)`.

Recommended reading order for the diagrams:

1. Read `oak_core` to understand the default four-interface surface.
2. Read `oak_runtime_overview` for the six phases of `step(...)`.
3. Read `oak_architecture` to see how the optional fine-grained layer is assembled.
4. Read `oak_runtime_sequence` to trace one composite-wired execution path.

`oak_runtime_overview` and `oak_runtime_sequence` describe the same six
phases.  The difference is only the level of expansion: `oak_runtime_overview`
stays at the four-interface layer, while `oak_runtime_sequence` shows what
happens when those slots are filled by the Composite* implementations from
`oak_architecture.fine_grained.composites`.  If either diagram and the code
ever disagree, the documentation should be fixed.

The diagrams are intentionally runtime-oriented.  They are not exhaustive
method inventories for the interfaces.  For the full surface area
(`reset`, `predict`, `current_subjective_state`, `OptionKeyboard`, and so on),
use the API reference below.  `oak_architecture` is the broadest inventory
view; `oak_runtime_overview` and `oak_runtime_sequence` are narrower and only
show what matters for one `OaKAgent.step(...)`.

## Step Walkthrough

Read the method as a pipeline.  Each block below corresponds to the next
block of code in `OaKAgent.step(...)`.

### 1. Perceive

```python
subjective_state = self.perception.update(...)
```

`time_step` is the input.  It carries `observation`, `reward`, `terminated`,
`truncated`, and optional `info`.  `perception` must turn these into the
current `subjective_state`.  Every later call in the step uses this
`subjective_state`, so your `Perception` implementation defines what the
agent actually reasons over.

### 2. Learn

```python
td_errors = self.value_function.update(transition)
self.reactive_policy.update(transition, td_errors)
self.transition_model.update(transition)
```

Learning starts only once the agent has both a previous `subjective_state`
and a previous `action`.  The first call to `step(...)` therefore sets up
memory but cannot yet build a full transition.

The `Transition` packages the previous/next subjective states, the action,
reward, the termination outcome, and optional `info`.  All three modules
receive it.
`value_function.update` returns TD errors that `reactive_policy.update`
uses for policy improvement.

### 3. Grow

```python
ranked_feature_ids = self.perception.discover_and_rank_features(...)
created_subtasks = self.perception.generate_subtasks(ranked_feature_ids)
self.reactive_policy.ingest_subtasks(created_subtasks)
self.reactive_policy.integrate_options()
self.transition_model.integrate_option_models()
```

`perception` proposes new features, ranks them by utility, and generates
subtasks from the most useful ones.  `reactive_policy` turns subtasks into
options.  `transition_model` integrates the latest option models so planning
can reason about them.  In the overview diagram this appears as top-level
module calls; in the detailed sequence diagram the same phase is expanded into
`Composite* -> fine_grained interface` calls.

### 4. Plan

```python
planning_update = self.transition_model.plan(
    subjective_state, self.value_function, self.planning_budget
)
self.reactive_policy.apply_planning_update(planning_update)
```

`transition_model.plan(...)` receives the current `subjective_state`, the
`value_function` (for state evaluation during search), and a budget.  It
returns a `PlanningUpdate`.  `reactive_policy` is informed about the
planner's output before action selection.

### 5. Act

```python
action, active_option_id = self.reactive_policy.select_action(
    subjective_state, self.option_stop_threshold
)
```

The reactive policy either continues an active option or makes a fresh
decision.  The output is always a primitive `action`, because that is what
the caller receives in `AgentStepResult`.

### 6. Maintain

```python
self.value_function.observe_usage(usage_records)
curation_decision = self.value_function.curate()
self._apply_curation(curation_decision)
```

Usage records for ranked features and the active option are sent to the
value function for utility tracking.  The value function then decides what
to prune.  `_apply_curation(...)` dispatches the decision to the relevant
modules: `perception.remove_features(...)`, `reactive_policy.remove_options(...)`,
`reactive_policy.remove_subtasks(...)`, `transition_model.remove_option_models(...)`,
`value_function.remove(...)`.

## Implementation Order

If your goal is to get a working agent quickly, implement in this order:

1. Make `Perception` produce a useful `subjective_state` from `TimeStep`.
   Have `discover_and_rank_features` return a fixed list and
   `generate_subtasks` return empty.
2. Make `ReactivePolicy` return valid actions from `select_action`.
   Have the other methods be no-ops.
3. Make `ValueFunction` accept `update` and return `predict` values.
   Have `curate` return an empty `CurationDecision`.
4. Make `TransitionModel` accept `update` and return a valid
   `PlanningUpdate` from `plan`, even if trivial.

That is enough to satisfy the exact call sequence of `OaKAgent.step(...)`.
After that, you can improve learning quality without changing the basic
wiring.

## Repository Examples

The concrete implementations live outside `oak_architecture` on purpose.
That shows the intended usage pattern: the package provides the canonical
`OaKAgent` coordinator and interfaces, while downstream code provides the
implementations.

- `examples/minimal_oak.py`
  A full smoke-path implementation using the **direct approach**.  Each of
  the four interfaces is implemented as a single class with intentionally
  small behavior.

To run all repository smoke tests, including the minimal example:

```bash
pixi run tests
```

To inspect the smallest runnable example directly:

```python
from examples.minimal_oak import build_minimal_agent, run_minimal_episode

agent = build_minimal_agent()
trace = run_minimal_episode(horizon=5)
```

`run_minimal_episode(...)` returns a compact trace with the
`subjective_state`, primitive `action`, `active_option_id`,
`created_subtasks`, and planner output at each step.

## Design Constraints

Keep these constraints in mind when you replace the minimal pieces with
real ones:

- `Perception` should define a useful `subjective_state` for the domain.
  The rest of the agent only sees that representation.
- `ReactivePolicy` should stay focused on choosing between primitive
  actions and options.  It should not absorb the work of planning or
  prediction.
- `ValueFunction` should start with one meaningful predictive target
  before you expand to many General Value Functions.
- `TransitionModel` should make honest predictions.  Bounded planning
  becomes misleading quickly if the model invents certainty it does not
  have.
- `ValueFunction.curate()` should stay conservative until you have stable
  evidence that a learned structure is safely removable.

---

# API Documentation
