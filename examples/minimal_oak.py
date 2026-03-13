from __future__ import annotations

"""Bare-minimum external implementation used to smoke-test the interface.

This module answers a single question: can the current package interfaces be
instantiated and run through a complete OaK step loop?

The answer should remain "yes" even while the architecture is still under
active development. That makes this module useful both as a tutorial and as a
regression check for interface changes.

Unlike the core package, this module lives outside `oak_architecture` on
purpose. It demonstrates what it looks like for a downstream project to wire
the published interfaces into a concrete agent.

What this module is:

- a tiny integer world
- a direct observation-to-subjective_state perception module
- one simple feature, one subtask, and one trivial option
- a no-op utility/curation setup

What this module is not:

- a trained agent
- a realistic planner
- a serious option-learning system
- a benchmark implementation
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, TypeAlias

from oak_architecture.agent import OaKAgent
from oak_architecture.interfaces import (
    Curator,
    FeatureBank,
    FeatureConstructor,
    FeatureRanker,
    GeneralValueFunctionLearner,
    MetaStepSizeLearner,
    Option,
    OptionLearner,
    OptionLibrary,
    OptionModel,
    OptionModelLearner,
    Perception,
    Planner,
    ReactivePolicy,
    SubtaskGenerator,
    TransitionModel,
    UtilityAssessor,
    ValueFunction,
    World,
)
from oak_architecture.types import (
    CurationDecision,
    FeatureCandidate,
    FeatureId,
    FeatureSpec,
    GeneralValueFunctionId,
    GeneralValueFunctionSpec,
    ModelPrediction,
    OptionDescriptor,
    OptionId,
    PlanningUpdate,
    PolicyDecision,
    SubtaskId,
    SubtaskSpec,
    TimeStep,
    Transition,
    UsageRecord,
    UtilityRecord,
)

Observation = int
Action = int
MinimalInfo: TypeAlias = dict[str, Any]


@dataclass(slots=True, frozen=True)
class MinimalSubjectiveState:
    """Small concrete subjective state used by the smoke implementation."""

    step_index: int
    observation: Observation
    reward: float
    last_action: Optional[Action]


class MinimalWorld(World[Observation, Action, MinimalInfo]):
    """A toy world that increments an integer observation every step."""

    def __init__(self, horizon: int = 5) -> None:
        self.horizon = horizon
        self.current_step = 0

    def reset(self) -> TimeStep[Observation, MinimalInfo]:
        self.current_step = 0
        return TimeStep(observation=0, reward=0.0, info={"reset": True})

    def step(self, action: Action) -> TimeStep[Observation, MinimalInfo]:
        self.current_step += 1
        terminated = self.current_step >= self.horizon
        reward = 1.0 if action == 1 else 0.0
        return TimeStep(
            observation=self.current_step,
            reward=reward,
            terminated=terminated,
            info={"echo_action": action},
        )


class MinimalPerception(Perception[Observation, Action, MinimalSubjectiveState]):
    """Converts each observation directly into a minimal subjective state object."""

    def __init__(self) -> None:
        self._subjective_state = MinimalSubjectiveState(0, 0, 0.0, None)

    def reset(self) -> None:
        self._subjective_state = MinimalSubjectiveState(0, 0, 0.0, None)

    def update(
        self,
        observation: Observation,
        reward: float,
        last_action: Optional[Action],
    ) -> MinimalSubjectiveState:
        self._subjective_state = MinimalSubjectiveState(
            step_index=observation,
            observation=observation,
            reward=reward,
            last_action=last_action,
        )
        return self._subjective_state

    def current_subjective_state(self) -> MinimalSubjectiveState:
        return self._subjective_state


class MinimalFeatureBank(FeatureBank[MinimalSubjectiveState]):
    """Holds one identity feature over the integer observation."""

    def __init__(self) -> None:
        self._features: dict[FeatureId, FeatureSpec] = {
            "observation": FeatureSpec(
                feature_id="observation",
                name="Observation value",
                description="Identity feature for the integer observation.",
            )
        }

    def list_features(self) -> Sequence[FeatureSpec]:
        return tuple(self._features.values())

    def activations(
        self, subjective_state: MinimalSubjectiveState
    ) -> Mapping[FeatureId, float]:
        return {"observation": float(subjective_state.observation)}

    def add_candidates(
        self,
        candidates: Sequence[FeatureCandidate],
    ) -> Sequence[FeatureSpec]:
        added: list[FeatureSpec] = []
        for candidate in candidates:
            spec = FeatureSpec(
                feature_id=candidate.feature_id,
                name=candidate.name,
                description=candidate.description,
                metadata=candidate.metadata,
            )
            self._features[candidate.feature_id] = spec
            added.append(spec)
        return tuple(added)

    def remove(self, feature_ids: Sequence[FeatureId]) -> None:
        for feature_id in feature_ids:
            self._features.pop(feature_id, None)


class MinimalFeatureConstructor(FeatureConstructor[MinimalSubjectiveState]):
    """Returns no new features, keeping the example intentionally minimal."""

    def propose(
        self,
        subjective_state: MinimalSubjectiveState,
        active_features: Sequence[FeatureSpec],
    ) -> Sequence[FeatureCandidate]:
        return ()


class MinimalFeatureRanker(FeatureRanker):
    """Keeps feature ordering stable and deterministic."""

    def rank(
        self,
        features: Sequence[FeatureSpec],
        utilities: Sequence[UtilityRecord],
        limit: Optional[int] = None,
    ) -> Sequence[FeatureId]:
        feature_ids = [feature.feature_id for feature in features]
        return tuple(feature_ids if limit is None else feature_ids[:limit])


class MinimalSubtaskGenerator(SubtaskGenerator[MinimalSubjectiveState]):
    """Creates at most one subtask per feature."""

    def __init__(self) -> None:
        self._created_for_feature: set[FeatureId] = set()

    def generate(
        self,
        ranked_feature_ids: Sequence[FeatureId],
        feature_bank: FeatureBank[MinimalSubjectiveState],
    ) -> Sequence[SubtaskSpec]:
        created: list[SubtaskSpec] = []
        for feature_id in ranked_feature_ids:
            if feature_id in self._created_for_feature:
                continue
            self._created_for_feature.add(feature_id)
            created.append(
                SubtaskSpec(
                    subtask_id=f"subtask:{feature_id}",
                    name=f"Track {feature_id}",
                    feature_id=feature_id,
                )
            )
        return tuple(created)


class MinimalGeneralValueFunctionLearner(
    GeneralValueFunctionLearner[MinimalSubjectiveState, Action, MinimalInfo]
):
    """Trivial GeneralValueFunction learner that stores only the latest reward."""

    def __init__(
        self, general_value_function_id: GeneralValueFunctionId, name: str
    ) -> None:
        self._spec = GeneralValueFunctionSpec(
            general_value_function_id=general_value_function_id,
            name=name,
            cumulant=lambda transition: transition.reward,
            continuation=lambda transition: 0.0 if transition.terminated else 1.0,
            termination_value=lambda transition: 0.0,
        )
        self._value = 0.0

    @property
    def spec(self) -> GeneralValueFunctionSpec:
        return self._spec

    def predict(
        self,
        subjective_state: MinimalSubjectiveState,
        action: Optional[Action] = None,
    ) -> float:
        return self._value

    def update(
        self,
        transition: Transition[Any, Action, MinimalSubjectiveState, MinimalInfo],
    ) -> float:
        self._value = transition.reward
        return 0.0


class MinimalValueFunction(ValueFunction[MinimalSubjectiveState, Action, MinimalInfo]):
    """Container for the main trivial GeneralValueFunction learner."""

    def __init__(self) -> None:
        self._learners: dict[
            GeneralValueFunctionId, MinimalGeneralValueFunctionLearner
        ] = {"main": MinimalGeneralValueFunctionLearner("main", "Main reward")}

    def list_general_value_functions(
        self,
    ) -> Sequence[
        GeneralValueFunctionLearner[MinimalSubjectiveState, Action, MinimalInfo]
    ]:
        return tuple(self._learners.values())

    def predict(
        self, subjective_state: MinimalSubjectiveState
    ) -> Mapping[GeneralValueFunctionId, float]:
        return {
            general_value_function_id: learner.predict(subjective_state)
            for general_value_function_id, learner in self._learners.items()
        }

    def update(
        self,
        transition: Transition[Any, Action, MinimalSubjectiveState, MinimalInfo],
    ) -> Mapping[GeneralValueFunctionId, float]:
        return {
            general_value_function_id: learner.update(transition)
            for general_value_function_id, learner in self._learners.items()
        }

    def add_or_replace(
        self,
        learner: GeneralValueFunctionLearner[
            MinimalSubjectiveState, Action, MinimalInfo
        ],
    ) -> None:
        self._learners[learner.spec.general_value_function_id] = learner  # type: ignore[assignment]

    def remove(
        self, general_value_function_ids: Sequence[GeneralValueFunctionId]
    ) -> None:
        for general_value_function_id in general_value_function_ids:
            if general_value_function_id != "main":
                self._learners.pop(general_value_function_id, None)


class MinimalOption(Option[MinimalSubjectiveState, Action]):
    """Option that always emits the same primitive action."""

    def __init__(self, descriptor: OptionDescriptor, action: Action = 1) -> None:
        self._descriptor = descriptor
        self._action = action

    @property
    def descriptor(self) -> OptionDescriptor:
        return self._descriptor

    def is_available(self, subjective_state: MinimalSubjectiveState) -> bool:
        return True

    def act(self, subjective_state: MinimalSubjectiveState) -> Action:
        return self._action

    def stop_probability(self, subjective_state: MinimalSubjectiveState) -> float:
        return 1.0


class MinimalOptionLibrary(OptionLibrary[MinimalSubjectiveState, Action]):
    """In-memory storage for smoke-test options."""

    def __init__(self) -> None:
        self._options: dict[OptionId, Option[MinimalSubjectiveState, Action]] = {}

    def list_options(self) -> Sequence[Option[MinimalSubjectiveState, Action]]:
        return tuple(self._options.values())

    def get(self, option_id: OptionId) -> Option[MinimalSubjectiveState, Action]:
        return self._options[option_id]

    def add_or_replace(self, option: Option[MinimalSubjectiveState, Action]) -> None:
        self._options[option.descriptor.option_id] = option

    def remove(self, option_ids: Sequence[OptionId]) -> None:
        for option_id in option_ids:
            self._options.pop(option_id, None)


class MinimalOptionLearner(OptionLearner[MinimalSubjectiveState, Action, MinimalInfo]):
    """Creates one trivial option per discovered subtask."""

    def __init__(self) -> None:
        self._subtasks: dict[SubtaskId, SubtaskSpec] = {}
        self._options: dict[OptionId, MinimalOption] = {}

    def ingest_subtasks(self, subtasks: Sequence[SubtaskSpec]) -> None:
        for subtask in subtasks:
            self._subtasks[subtask.subtask_id] = subtask
            option_id = f"option:{subtask.subtask_id}"
            self._options[option_id] = MinimalOption(
                OptionDescriptor(
                    option_id=option_id,
                    name=f"Option for {subtask.subtask_id}",
                    subtask_id=subtask.subtask_id,
                )
            )

    def update(
        self,
        transition: Transition[Any, Action, MinimalSubjectiveState, MinimalInfo],
    ) -> None:
        return None

    def export_options(self) -> Sequence[Option[MinimalSubjectiveState, Action]]:
        return tuple(self._options.values())

    def remove_subtasks(self, subtask_ids: Sequence[SubtaskId]) -> None:
        for subtask_id in subtask_ids:
            self._subtasks.pop(subtask_id, None)
            self._options.pop(f"option:{subtask_id}", None)


class MinimalOptionModel(OptionModel[MinimalSubjectiveState]):
    """Option model that predicts no meaningful change."""

    def __init__(self, option_id: OptionId) -> None:
        self._option_id = option_id

    @property
    def option_id(self) -> OptionId:
        return self._option_id

    def predict(
        self, subjective_state: MinimalSubjectiveState
    ) -> ModelPrediction[MinimalSubjectiveState]:
        return ModelPrediction(
            predicted_subjective_state=subjective_state,
            cumulative_reward=0.0,
            steps=1,
            terminated=False,
        )


class MinimalOptionModelLearner(
    OptionModelLearner[MinimalSubjectiveState, Action, MinimalInfo]
):
    """Wraps the smoke-test options in smoke-test models."""

    def __init__(self, option_learner: MinimalOptionLearner) -> None:
        self._option_learner = option_learner

    def update(
        self,
        transition: Transition[Any, Action, MinimalSubjectiveState, MinimalInfo],
    ) -> None:
        return None

    def export_models(self) -> Sequence[OptionModel[MinimalSubjectiveState]]:
        return tuple(
            MinimalOptionModel(option.descriptor.option_id)
            for option in self._option_learner.export_options()
        )


class MinimalTransitionModel(
    TransitionModel[MinimalSubjectiveState, Action, MinimalInfo]
):
    """Very small predictive model used only to exercise planner calls."""

    def __init__(self) -> None:
        self._option_models: dict[OptionId, OptionModel[MinimalSubjectiveState]] = {}

    def update(
        self,
        transition: Transition[Any, Action, MinimalSubjectiveState, MinimalInfo],
    ) -> None:
        return None

    def predict_action(
        self,
        subjective_state: MinimalSubjectiveState,
        action: Action,
    ) -> ModelPrediction[MinimalSubjectiveState]:
        return ModelPrediction(
            predicted_subjective_state=MinimalSubjectiveState(
                step_index=subjective_state.step_index + 1,
                observation=subjective_state.observation + 1,
                reward=1.0 if action == 1 else 0.0,
                last_action=action,
            ),
            cumulative_reward=1.0 if action == 1 else 0.0,
            steps=1,
            terminated=False,
        )

    def predict_option(
        self,
        subjective_state: MinimalSubjectiveState,
        option_id: OptionId,
    ) -> ModelPrediction[MinimalSubjectiveState]:
        model = self._option_models.get(option_id)
        if model is not None:
            return model.predict(subjective_state)
        return ModelPrediction(
            predicted_subjective_state=subjective_state,
            cumulative_reward=0.0,
            steps=1,
            terminated=False,
        )

    def add_or_replace_option_models(
        self,
        models: Sequence[OptionModel[MinimalSubjectiveState]],
    ) -> None:
        for model in models:
            self._option_models[model.option_id] = model

    def remove_option_models(self, option_ids: Sequence[OptionId]) -> None:
        for option_id in option_ids:
            self._option_models.pop(option_id, None)


class MinimalPlanner(Planner[MinimalSubjectiveState, Action, MinimalInfo]):
    """Consumes the transition model once and returns a tiny planning update."""

    def plan_step(
        self,
        subjective_state: MinimalSubjectiveState,
        model: TransitionModel[MinimalSubjectiveState, Action, MinimalInfo],
        value_function: ValueFunction[MinimalSubjectiveState, Action, MinimalInfo],
        budget: int,
    ) -> PlanningUpdate[Action]:
        _ = model.predict_action(subjective_state, 0)
        return PlanningUpdate(
            value_targets=value_function.predict(subjective_state),
            policy_targets={"preferred_action": 0},
            search_statistics={"budget_used": budget},
        )


class MinimalReactivePolicy(ReactivePolicy[MinimalSubjectiveState, Action]):
    """Alternates between primitive actions and the first available option."""

    def __init__(self) -> None:
        self.last_td_errors: Mapping[GeneralValueFunctionId, float] = {}
        self.last_planning_update: Optional[PlanningUpdate[Action]] = None

    def decide(
        self,
        subjective_state: MinimalSubjectiveState,
        active_option: Optional[Option[MinimalSubjectiveState, Action]],
        available_options: Sequence[Option[MinimalSubjectiveState, Action]],
    ) -> PolicyDecision[Action]:
        if subjective_state.observation % 2 == 0:
            return PolicyDecision(action=0)
        if available_options:
            return PolicyDecision(option_id=available_options[0].descriptor.option_id)
        return PolicyDecision(action=1)

    def update_from_values(
        self,
        subjective_state: MinimalSubjectiveState,
        td_errors: Mapping[GeneralValueFunctionId, float],
    ) -> None:
        self.last_td_errors = dict(td_errors)

    def apply_planning_update(self, update: PlanningUpdate[Action]) -> None:
        self.last_planning_update = update


class MinimalUtilityAssessor(UtilityAssessor):
    """Accumulates raw usage counts into simple utility scores."""

    def __init__(self) -> None:
        self._usage_records: list[UsageRecord] = []

    def observe(self, usage: Sequence[UsageRecord]) -> None:
        self._usage_records.extend(usage)

    def scores(self) -> Sequence[UtilityRecord]:
        totals: dict[tuple[str, str], float] = {}
        latest_records: dict[tuple[str, str], UsageRecord] = {}
        for record in self._usage_records:
            key = (record.kind.value, record.component_id)
            totals[key] = totals.get(key, 0.0) + record.amount
            latest_records[key] = record
        return tuple(
            UtilityRecord(
                kind=record.kind,
                component_id=record.component_id,
                utility=totals[key],
            )
            for key, record in latest_records.items()
        )


class MinimalCurator(Curator):
    """Returns no pruning decisions in the smoke implementation."""

    def curate(self, utilities: Sequence[UtilityRecord]) -> CurationDecision:
        return CurationDecision()


class MinimalMetaStepSizeLearner(MetaStepSizeLearner):
    """Stores the latest error signals without adapting anything."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, float]] = {}

    def update(self, component_id: str, error_signals: Mapping[str, float]) -> None:
        self._store[component_id] = dict(error_signals)

    def step_sizes(self, component_id: str) -> Mapping[str, float]:
        return self._store.get(component_id, {})


def build_minimal_agent() -> (
    OaKAgent[Observation, Action, MinimalSubjectiveState, MinimalInfo]
):
    """Construct a fully wired smoke-test OaK agent."""
    option_learner = MinimalOptionLearner()
    return OaKAgent(
        perception=MinimalPerception(),
        feature_bank=MinimalFeatureBank(),
        feature_constructor=MinimalFeatureConstructor(),
        feature_ranker=MinimalFeatureRanker(),
        subtask_generator=MinimalSubtaskGenerator(),
        value_function=MinimalValueFunction(),
        reactive_policy=MinimalReactivePolicy(),
        option_library=MinimalOptionLibrary(),
        option_learner=option_learner,
        option_model_learner=MinimalOptionModelLearner(option_learner),
        transition_model=MinimalTransitionModel(),
        planner=MinimalPlanner(),
        utility_assessor=MinimalUtilityAssessor(),
        curator=MinimalCurator(),
        meta_step_sizes=MinimalMetaStepSizeLearner(),
        planning_budget=4,
    )


def run_minimal_episode(horizon: int = 5) -> list[dict[str, Any]]:
    """Run a short smoke episode and return a compact trace."""
    world = MinimalWorld(horizon=horizon)
    agent = build_minimal_agent()
    step = world.reset()
    agent.reset()

    trace: list[dict[str, Any]] = []

    for _ in range(horizon):
        result = agent.step(step)
        action = result.action
        trace.append(
            {
                "subjective_state": result.subjective_state,
                "action": action,
                "active_option_id": result.active_option_id,
                "created_subtasks": [
                    subtask.subtask_id for subtask in result.created_subtasks
                ],
                "planning_budget_used": (
                    result.planning_update.search_statistics["budget_used"]
                    if result.planning_update is not None
                    else None
                ),
            }
        )
        step = world.step(action)
        if step.terminated:
            break

    return trace
