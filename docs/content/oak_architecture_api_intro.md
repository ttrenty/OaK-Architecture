## Diagram Gallery

<div class="oak-diagram-grid">
  <figure class="oak-diagram-card">
    <img src="img/oak_core.svg" alt="High-level PlantUML view of the OaK core loop." loading="lazy">
    <figcaption><strong>High-level PlantUML view.</strong> The smallest useful control loop for the package.</figcaption>
  </figure>
  <figure class="oak-diagram-card">
    <img src="img/oak_architecture.svg" alt="Detailed PlantUML view of the OaK architecture." loading="lazy">
    <figcaption><strong>Detailed PlantUML view.</strong> The full package map, regrouped to separate core runtime, abstraction growth, and maintenance.</figcaption>
  </figure>
  <figure class="oak-diagram-card">
    <img src="img/oak_runtime_sequence.svg" alt="Runtime sequence diagram for one OaK agent step." loading="lazy">
    <figcaption><strong>Runtime sequence.</strong> The order in which `OaKAgent.step(...)` calls the subsystems.</figcaption>
  </figure>
</div>

## How the diagrams map to the Python package

The diagrams are a visual index into three source modules:

- `oak_architecture.interfaces` defines the protocols and abstract base classes
  shown as boxes in the diagrams, including `Perception`, `ReactivePolicy`,
  `ValueFunction`, `TransitionModel`, `FeatureConstructor`, `FeatureBank`,
  `FeatureRanker`, `SubtaskGenerator`, `OptionLearner`, `OptionLibrary`,
  `OptionModelLearner`, `Planner`, `UtilityAssessor`, `Curator`, and
  `MetaStepSizeLearner`.
- `oak_architecture.types` defines the objects flowing along the arrows:
  `TimeStep`, `Transition`, `FeatureCandidate`, `FeatureSpec`, `SubtaskSpec`,
  `OptionDescriptor`, `ModelPrediction`, `PlanningUpdate`, `UsageRecord`,
  `UtilityRecord`, `CurationDecision`, `PolicyDecision`, and `AgentStepResult`.
- `oak_architecture.agent` defines `OaKAgent`, the coordinator whose
  `step(...)` method performs the interactions shown in the runtime diagram.

The docs keep a single PlantUML-based diagram set so the visual vocabulary
stays consistent across the repository. The high-level and detailed views now
differ by level of abstraction, not by diagram tool.

### High-level core views

The `oak_core_*` diagrams are the best entry point for a minimal
implementation. They correspond to the shortest path through
`OaKAgent.step(...)`:

1. A caller or `World` produces a `TimeStep`.
2. `Perception.update(...)` converts the observation, reward, and previous
   action into the current `StateT`.
3. `ValueFunction` and `TransitionModel` supply predictive structure around
   that state.
4. `ReactivePolicy.decide(...)` selects the next primitive action or
   option-backed action.

If you only want a small continual-learning baseline, these are the first
interfaces to make concrete.

### Detailed architecture views

The `oak_architecture_*` diagrams expand the core loop into the additional OaK
machinery:

- feature growth through `FeatureConstructor`, `FeatureBank`, and
  `FeatureRanker`
- abstraction growth through `SubtaskGenerator`, `OptionLearner`, and
  `OptionLibrary`
- option-model learning through `OptionModelLearner` and the option-model
  branch of `TransitionModel`
- planning through `Planner.plan_step(...)` and
  `ReactivePolicy.apply_planning_update(...)`
- self-maintenance through `UtilityAssessor.observe(...)`,
  `UtilityAssessor.scores()`, and `Curator.curate(...)`
- adaptive update control through `MetaStepSizeLearner.update(...)`

These blocks are visible as separate attributes on `OaKAgent`, so the dataclass
in `oak_architecture.agent` is effectively the code counterpart of the detailed
architecture diagrams.

### Runtime sequence

The runtime sequence diagram is the closest visual match to the implementation
of `OaKAgent.step(...)`. The method performs the interactions in this order:

1. Update perception from the incoming `TimeStep`.
2. If a previous state and action exist, build a `Transition` and update
   `ValueFunction`, `ReactivePolicy`, `OptionLearner`,
   `OptionModelLearner`, and `TransitionModel`.
3. Propose and rank features, then generate `SubtaskSpec` objects from the
   top-ranked features.
4. Export learned options and option models into `OptionLibrary` and
   `TransitionModel`.
5. Run `Planner.plan_step(...)` to produce a `PlanningUpdate`, then apply it to
   `ReactivePolicy`.
6. Select a primitive action or continue or start an `Option`.
7. Record `UsageRecord` evidence, score utilities, and let `Curator` prune
   low-value components.
8. Return an `AgentStepResult` containing the chosen action, current state, any
   planning update, any new subtasks, and any curation decision.
