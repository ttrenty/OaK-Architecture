---

# Architecture Guide

## Diagram Gallery

<div class="oak-diagram-grid">
  <figure class="oak-diagram-card">
    <img src="img/oak_core.svg" alt="Compressed view of the core control path inside OaKAgent.step(...)." loading="lazy">
    <figcaption><strong>`oak_core`.</strong> A compressed view of the decision path inside <code>OaKAgent.step(...)</code>.</figcaption>
  </figure>
  <figure class="oak-diagram-card">
    <img src="img/oak_architecture.svg" alt="Grouped component map for OaKAgent.step(...), organized by Sutton's four families." loading="lazy">
    <figcaption><strong>`oak_architecture`.</strong> A grouped component map: all concrete OaK components stay visible, but they are organized under Sutton's four main families <code>Perception</code>, <code>TransitionModel</code>, <code>ValueFunction</code>, and <code>ReactivePolicy</code>.</figcaption>
  </figure>
  <figure class="oak-diagram-card">
    <img src="img/oak_runtime_overview.svg" alt="Simplified runtime sequence for the main phases of OaKAgent.step(...)." loading="lazy">
    <figcaption><strong>`oak_runtime_overview`.</strong> A simplified runtime sequence that shows the main phases of <code>OaKAgent.step(...)</code> without every individual call.</figcaption>
  </figure>
  <figure class="oak-diagram-card">
    <img src="img/oak_runtime_sequence.svg" alt="Exact call order for one OaKAgent.step(...) execution." loading="lazy">
    <figcaption><strong>`oak_runtime_sequence`.</strong> The exact call order for one <code>OaKAgent.step(...)</code> execution.</figcaption>
  </figure>
</div>

## What You Must Implement

`OaKAgent` is the canonical coordinator in this package. The usual workflow is
not to subclass it, but to instantiate it with concrete objects for its fields.

To run an agent, you must provide implementations for these `OaKAgent` fields:

- `perception`
  Implements `Perception`. It receives raw environment data and must return the
  current `subjective_state`.
- `feature_bank`
  Implements `FeatureBank`. It stores active `FeatureSpec` objects and can add
  or remove them.
- `feature_constructor`
  Implements `FeatureConstructor`. It proposes new `FeatureCandidate` objects
  from the current `subjective_state`.
- `feature_ranker`
  Implements `FeatureRanker`. It orders current features for downstream use.
- `subtask_generator`
  Implements `SubtaskGenerator`. It turns ranked feature ids into
  `SubtaskSpec` objects.
- `value_function`
  Implements `ValueFunction`. It learns from `Transition` objects and returns
  value-error signals for the policy.
- `reactive_policy`
  Implements `ReactivePolicy`. It accepts the current `subjective_state`,
  planning updates, and value updates, then chooses either a primitive action
  or an option id.
- `option_library`
  Implements `OptionLibrary`. It stores executable options and returns them by
  id.
- `option_learner`
  Implements `OptionLearner`. It ingests subtasks, learns from transitions, and
  exports options.
- `option_model_learner`
  Implements `OptionModelLearner`. It learns predictive models for options and
  exports them.
- `transition_model`
  Implements `TransitionModel`. It learns from transitions and answers the
  planner's predictive queries.
- `planner`
  Implements `Planner`. It consumes `subjective_state`, `transition_model`, and
  `value_function`, and must return a `PlanningUpdate`.
- `utility_assessor`
  Implements `UtilityAssessor`. It converts usage evidence into
  `UtilityRecord` scores.
- `curator`
  Implements `Curator`. It turns utility scores into a `CurationDecision`.
- `meta_step_sizes`
  Optionally implements `MetaStepSizeLearner`. If present, it receives error
  signals during transition updates.

You also configure scalar controls:

- `planning_budget`
- `feature_budget`
- `option_stop_threshold`

`OaKAgent` manages these runtime fields itself:

- `active_option`
- `last_action`
- `last_subjective_state`
- `last_observation`

Your environment loop does not have to implement `World`, but it must supply
`TimeStep` objects to `OaKAgent.step(...)`. `World` is just the package's small
protocol for one way to do that.

The diagrams on this page use the exact Python field names from
`OaKAgent`: `perception`, `feature_bank`, `feature_constructor`,
`feature_ranker`, `subtask_generator`, `value_function`, `reactive_policy`,
`option_library`, `option_learner`, `option_model_learner`,
`transition_model`, `planner`, `utility_assessor`, `curator`, and
`meta_step_sizes`.

## Diagram-to-Code Mapping

The diagrams have different jobs, but they all describe the same
implementation:

- `oak_core`
  A compressed view of the main control path: update `subjective_state`, plan,
  decide, and return an `action`.
- `oak_architecture`
  A grouped component map of `OaKAgent.step(...)`. It keeps the concrete OaK
  component names visible, but arranges them under Sutton's four main
  families.
- `oak_runtime_overview`
  A simplified phase-by-phase sequence for understanding the logic before
  reading the exact call order.
- `oak_runtime_sequence`
  The authoritative order of calls made by `OaKAgent.step(...)`.

Recommended reading order for the diagrams:

1. Read `oak_core` for the shortest control path.
2. Read `oak_runtime_overview` for the main phases of `step(...)`.
3. Read `oak_runtime_sequence` for the exact call order.
4. Read `oak_architecture` when you want the concrete components grouped by the
   four main families.

`oak_runtime_sequence` should be read as the visual form of the code in
`oak_architecture.agent.OaKAgent.step(...)`. If that diagram and the code ever
disagree, the documentation should be fixed.

## Step Walkthrough

Read the method as a pipeline. Each block below corresponds to the next block
of code in `OaKAgent.step(...)`.

1. `subjective_state = self.perception.update(...)`
   `time_step` is the input for this step. It carries `observation`, `reward`,
   `terminated`, `truncated`, and optional `info`. `perception` must turn the
   raw `observation`, the scalar `reward`, and `self.last_action` into the
   current `subjective_state`. Every later component in the step uses this
   `subjective_state`, so your `Perception` implementation defines what the
   agent actually reasons over. This is the package's place for Richard
   Sutton's idea that the agent acts from its own learned internal
   representation, not directly from an assumed external world state.
2. `created_subtasks = ()`, `ranked_feature_ids = ()`,
   `planning_update = None`, `curation_decision = None`
   These locals are initialized up front so the method can always return a
   complete `AgentStepResult`, even when nothing new is created or pruned.
3. `if self.last_subjective_state is not None and self.last_action is not None:`
   Learning from experience starts only once the agent has both a previous
   `subjective_state` and a previous `action`. The first call to `step(...)`
   therefore sets up memory but cannot yet build a full transition.
4. `transition = Transition(...)`
   `Transition` is the agent-centric learning record. It packages the previous
   `subjective_state`, previous `action`, current `reward`,
   `next_subjective_state`, previous observation, current observation,
   termination flags, and `info`. In code, those exact field names are
   `subjective_state`, `action`, `reward`, `next_subjective_state`,
   `observation`, `next_observation`, `terminated`, and `info`. This is the
   common object that all learning subsystems share.
5. `self._update_from_transition(transition)`
   This helper applies the observed transition to the main learners. It is
   expanded in the next section, but at a high level it updates
   `value_function`, `reactive_policy`, `option_learner`,
   `option_model_learner`, `transition_model`, and optionally
   `meta_step_sizes`.
6. `candidates = self.feature_constructor.propose(...)`
   `feature_constructor` looks at the current `subjective_state` together with
   the current `feature_bank.list_features()` and proposes new
   `FeatureCandidate` objects. It is allowed to return an empty sequence.
7. `if candidates: self.feature_bank.add_candidates(candidates)`
   `feature_bank` is the current store of active features. If new candidates are
   admitted, they become part of the bank immediately and can participate in the
   rest of the same step.
8. `ranked_feature_ids = self.feature_ranker.rank(...)`
   `feature_ranker` chooses which existing features matter most right now. It
   sees the current features, the current utility scores from
   `utility_assessor.scores()`, and the `feature_budget`. This is the point
   where the agent decides which representational structures deserve downstream
   attention.
9. `created_subtasks = self.subtask_generator.generate(...)`
   `subtask_generator` converts the ranked feature ids into `SubtaskSpec`
   objects. If new subtasks appear, `self.option_learner.ingest_subtasks(...)`
   is called immediately so option learning can start from those subtasks.
10. `for option in self.option_learner.export_options(): ...`
    `option_learner` exports executable options, and `option_library` becomes
    the live store that `reactive_policy` can query by id.
11. `self.transition_model.add_or_replace_option_models(...)`
    `option_model_learner` exports option models, and `transition_model` stores
    them so planning can reason about options as well as primitive actions.
12. `planning_update = self.planner.plan_step(...)`
    `planner` receives the current `subjective_state`, the current
    `transition_model`, the current `value_function`, and `planning_budget`. It
    must return a `PlanningUpdate`. This is where search, rollouts, or any
    other bounded planning computation happens.
13. `self.reactive_policy.apply_planning_update(planning_update)`
    `reactive_policy` is informed about the planner's output before action
    selection happens. This is the bridge from deliberative computation back to
    immediate action choice.
14. `action, active_option_id = self._select_action(subjective_state)`
    This helper either continues `self.active_option` or asks
    `reactive_policy.decide(...)` for a new primitive action or option id. It
    is expanded in the next section. The key point is that the final output of
    the helper is always a primitive `action`, because that is what the caller
    receives in `AgentStepResult`.
15. `usage_records = self._build_usage_records(...)`
    The current ranked features and any active option are converted into
    `UsageRecord` objects. These are the minimal observations that feed utility
    assessment.
16. `self.utility_assessor.observe(usage_records)`
    `utility_assessor` accumulates evidence about what is being used. It is
    allowed to ignore some signals, aggregate them, or turn them into a richer
    internal accounting system.
17. `utility_scores = self.utility_assessor.scores()`
    This produces the current `UtilityRecord` objects. If there are any scores,
    then `curator.curate(utility_scores)` is called next.
18. `curation_decision = self.curator.curate(utility_scores)`
    `curator` decides what should be dropped. `self._apply_curation(...)`
    applies that decision by removing features, subtasks, options, option
    models, and GVFs from the live agent fields.
19. `self.last_subjective_state = subjective_state`,
    `self.last_observation = ...`, `self.last_action = action`
    These assignments update the memory that will be needed to build the next
    `Transition`.
20. `if time_step.terminated or time_step.truncated: self.active_option = None`
    Episode boundaries clear the currently active option so the next episode
    starts cleanly.
21. `return AgentStepResult(...)`
    The caller gets the primitive `action`, current `subjective_state`,
    optional `active_option_id`, optional `planning_update`, any
    `created_subtasks`, and any `curation_decision`. In code, the exact result
    fields are `action`, `subjective_state`, `active_option_id`,
    `planning_update`, `created_subtasks`, and `curation_decision`.

## Helper Methods

`step(...)` delegates important work to four helpers. You need to understand
their contracts because your concrete implementations are called through them.

- `_update_from_transition(transition)`
  Calls `value_function.update(transition)` and expects a mapping of error
  signals back. It then calls `reactive_policy.update_from_values(...)`,
  `option_learner.update(...)`, `option_model_learner.update(...)`, and
  `transition_model.update(...)`. If `meta_step_sizes` exists, it receives the
  returned error signals too.
- `_select_action(subjective_state)`
  If `active_option` exists and its `stop_probability(subjective_state)` is
  below
  `option_stop_threshold`, the option continues and produces the primitive
  `action` through `act(subjective_state)`. Otherwise the agent calls
  `reactive_policy.decide(...)`. If the policy returns an `option_id`, that
  option is fetched from `option_library` and immediately executed with
  `act(subjective_state)`. If it returns an `action`, that primitive action is
  used directly.
- `_build_usage_records(ranked_feature_ids, active_option_id)`
  Builds minimal `UsageRecord` objects for the current step. The default logic
  records each ranked feature and the currently active option, if one exists.
- `_apply_curation(curation_decision)`
  Applies curator outputs to the live agent: `feature_bank.remove(...)`,
  `option_learner.remove_subtasks(...)`, `option_library.remove(...)`,
  `transition_model.remove_option_models(...)`, and
  `value_function.remove(...)`.

## Implementation Order

If your goal is to get a working agent quickly, implement in this order:

1. Make `perception` produce a useful `subjective_state` from `TimeStep`.
2. Make `reactive_policy` return valid `PolicyDecision` objects.
3. Make `value_function` and `transition_model` accept `Transition` updates
   without failing.
4. Make `planner` return a valid `PlanningUpdate`, even if it is initially very
   small.
5. Make `feature_bank`, `feature_constructor`, `feature_ranker`,
   `subtask_generator`, `option_learner`, `option_library`, and
   `option_model_learner` return empty or minimal valid structures.
6. Make `utility_assessor` and `curator` return conservative outputs first.

That is enough to satisfy the exact call sequence of `OaKAgent.step(...)`.
After that, you can improve learning quality without changing the basic wiring.

## Repository Examples

The concrete implementations live outside `oak_architecture` on purpose. That
shows the intended usage pattern: the package provides the canonical
`OaKAgent` coordinator and interfaces, while downstream code provides the
implementations.

The repository currently includes two external examples:

- `examples/minimal_oak.py`
  A full smoke-path implementation. It wires every required component with
  intentionally small behavior so you can see the full contract surface.
- `examples/sutton_planning.py`
  An incomplete planning-focused example. It shows where Sutton-style planning
  formulas belong: predictive terms in the model side and backup calculations
  in the planner.

To run all repository smoke tests, including the minimal example:

```bash
pixi run test
```

To inspect the smallest runnable example directly:

```python
from examples.minimal_oak import build_minimal_agent, run_minimal_episode

agent = build_minimal_agent()
trace = run_minimal_episode(horizon=5)
```

`run_minimal_episode(...)` returns a compact trace with the
`subjective_state`, primitive `action`, `active_option_id`,
`created_subtasks`, and planner output at each step. Use it as a structural
example, not as an algorithmic baseline.

## Design Constraints

Keep these constraints in mind when you replace the minimal pieces with real
ones:

- `Perception` should define a useful `subjective_state` for the domain. The
  rest of the agent only sees that representation.
- `ReactivePolicy` should stay focused on choosing between primitive actions
  and options. It should not absorb the work of planning or prediction.
- `ValueFunction` should start with one meaningful predictive target before you
  expand to many GVFs or auxiliaries.
- `TransitionModel` should make honest predictions. Bounded planning becomes
  misleading quickly if the model invents certainty it does not have.
- `UtilityAssessor` and `Curator` should stay conservative until you have
  stable evidence that a learned structure is useful or safely removable.

---

# API Documentation
