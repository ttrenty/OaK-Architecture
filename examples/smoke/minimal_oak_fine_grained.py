from __future__ import annotations

"""Bare-minimum OaK example built from fine-grained components.

This mirrors `examples/minimal_oak.py`, but instead of implementing the four
main OaK interfaces directly, it assembles them from the optional fine-grained
building blocks in `oak.fine_grained`.

The behavior is intentionally the same as the direct example:

- a tiny integer world
- a direct observation-to-subjective_state state builder
- one fixed identity feature
- no-op model learning with trivial planning
- a simple value tracker with usage counting and no curation
- a reactive policy that alternates actions and options
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

from oak.agent import OaKAgent
from oak.fine_grained import (
    ActionSelector,
    CompositePerception,
    CompositeReactivePolicy,
    CompositeTransitionModel,
    CompositeValueFunction,
    Curator,
    FeatureBank,
    FeatureConstructor,
    FeatureRanker,
    GeneralValueFunctionLearner,
    Option,
    OptionLearner,
    OptionLibrary,
    OptionModel,
    OptionModelLearner,
    Planner,
    StateBuilder,
    SubtaskGenerator,
    UtilityAssessor,
    ValueEstimator,
    WorldModel,
)
from oak.types import (
    CurationDecision,
    FeatureCandidate,
    FeatureId,
    FeatureSpec,
    GeneralValueFunctionId,
    ModelPrediction,
    OptionDescriptor,
    OptionId,
    PlanningUpdate,
    PolicyDecision,
    SubtaskId,
    SubtaskSpec,
    Transition,
    UsageRecord,
    UtilityRecord,
)

from .minimal_oak import (
    Action,
    MinimalInfo,
    MinimalSubjectiveState,
    MinimalTraceStep,
    MinimalWorld,
    Observation,
    _planning_budget_used,
)


# ─────────────────────────────────────────────────────────────────────
# Perception components
# ─────────────────────────────────────────────────────────────────────


class MinimalStateBuilder(StateBuilder[Observation, Action, MinimalSubjectiveState]):
    """Direct observation-to-state mapping."""

    def __init__(self) -> None:
        self._state = MinimalSubjectiveState(0, 0, 0.0, None)

    def reset(self) -> None:
        self._state = MinimalSubjectiveState(0, 0, 0.0, None)

    def update(
        self,
        observation: Observation,
        reward: float,
        last_action: Action | None,
    ) -> MinimalSubjectiveState:
        self._state = MinimalSubjectiveState(
            step_index=observation,
            observation=observation,
            reward=reward,
            last_action=last_action,
        )
        return self._state

    def current_subjective_state(self) -> MinimalSubjectiveState:
        return self._state


class MinimalFeatureBank(FeatureBank[MinimalSubjectiveState]):
    """Stores one fixed identity feature."""

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
        self,
        subjective_state: MinimalSubjectiveState,
    ) -> Mapping[FeatureId, float]:
        return {"observation": float(subjective_state.observation)}

    def add_candidates(
        self, candidates: Sequence[FeatureCandidate]
    ) -> Sequence[FeatureSpec]:
        added: list[FeatureSpec] = []
        for candidate in candidates:
            feature = FeatureSpec(
                feature_id=candidate.feature_id,
                name=candidate.name,
                description=candidate.description,
                metadata=candidate.metadata,
            )
            self._features[feature.feature_id] = feature
            added.append(feature)
        return tuple(added)

    def remove(self, feature_ids: Sequence[FeatureId]) -> None:
        for feature_id in feature_ids:
            self._features.pop(feature_id, None)


class MinimalFeatureConstructor(FeatureConstructor[MinimalSubjectiveState]):
    """Never proposes new features."""

    def propose(
        self,
        subjective_state: MinimalSubjectiveState,
        active_features: Sequence[FeatureSpec],
    ) -> Sequence[FeatureCandidate]:
        return ()


class MinimalFeatureRanker(FeatureRanker):
    """Ranks features in their existing order."""

    def rank(
        self,
        features: Sequence[FeatureSpec],
        utilities: Sequence[UtilityRecord],
        limit: int | None = None,
    ) -> Sequence[FeatureId]:
        feature_ids = [feature.feature_id for feature in features]
        if limit is None:
            return tuple(feature_ids)
        return tuple(feature_ids[:limit])


class MinimalSubtaskGenerator(SubtaskGenerator[MinimalSubjectiveState]):
    """Creates at most one subtask per feature."""

    def __init__(self) -> None:
        self._created_subtask_for: set[FeatureId] = set()

    def generate(
        self,
        ranked_feature_ids: Sequence[FeatureId],
        feature_bank: FeatureBank[MinimalSubjectiveState],
    ) -> Sequence[SubtaskSpec]:
        created: list[SubtaskSpec] = []
        feature_specs = {
            feature.feature_id: feature for feature in feature_bank.list_features()
        }
        for feature_id in ranked_feature_ids:
            if feature_id in self._created_subtask_for:
                continue
            self._created_subtask_for.add(feature_id)
            feature = feature_specs[feature_id]
            created.append(
                SubtaskSpec(
                    subtask_id=f"subtask:{feature_id}",
                    name=f"Track {feature.name}",
                    feature_id=feature_id,
                )
            )
        return tuple(created)


# ─────────────────────────────────────────────────────────────────────
# Transition-model components
# ─────────────────────────────────────────────────────────────────────


class MinimalWorldModel(WorldModel[MinimalSubjectiveState, Action, MinimalInfo]):
    """Trivial planner-facing model."""

    def update(
        self,
        transition: Transition[Action, MinimalSubjectiveState, MinimalInfo],
    ) -> None:
        pass

    def predict_action(
        self,
        subjective_state: MinimalSubjectiveState,
        action: Action,
    ) -> ModelPrediction[MinimalSubjectiveState]:
        return ModelPrediction(
            predicted_subjective_state=subjective_state,
            cumulative_reward=0.0,
            steps=1,
        )

    def predict_option(
        self,
        subjective_state: MinimalSubjectiveState,
        option_id: OptionId,
    ) -> ModelPrediction[MinimalSubjectiveState]:
        return ModelPrediction(
            predicted_subjective_state=subjective_state,
            cumulative_reward=0.0,
            steps=1,
        )

    def add_or_replace_option_models(
        self, models: Sequence[OptionModel[MinimalSubjectiveState]]
    ) -> None:
        pass

    def remove_option_models(self, option_ids: Sequence[OptionId]) -> None:
        pass


class MinimalOptionModelLearner(
    OptionModelLearner[MinimalSubjectiveState, Action, MinimalInfo]
):
    """No-op option-model learner."""

    def update(
        self,
        transition: Transition[Action, MinimalSubjectiveState, MinimalInfo],
    ) -> None:
        pass

    def export_models(self) -> Sequence[OptionModel[MinimalSubjectiveState]]:
        return ()


class MinimalPlanner(Planner[MinimalSubjectiveState, Action, MinimalInfo]):
    """Returns one-step value targets without real search."""

    def plan_step(
        self,
        subjective_state: MinimalSubjectiveState,
        model: WorldModel[MinimalSubjectiveState, Action, MinimalInfo],
        value_function: ValueEstimator[MinimalSubjectiveState, Action, MinimalInfo],
        budget: int,
    ) -> PlanningUpdate[Action]:
        return PlanningUpdate(
            value_targets=value_function.predict(subjective_state),
            policy_targets={"preferred_action": 0},
            search_statistics={"budget_used": budget},
        )


# ─────────────────────────────────────────────────────────────────────
# Value-function components
# ─────────────────────────────────────────────────────────────────────


class MinimalValueEstimator(
    ValueEstimator[MinimalSubjectiveState, Action, MinimalInfo]
):
    """Stores latest reward as the only value estimate."""

    def __init__(self) -> None:
        self._value: float = 0.0

    def list_general_value_functions(
        self,
    ) -> Sequence[
        GeneralValueFunctionLearner[MinimalSubjectiveState, Action, MinimalInfo]
    ]:
        return ()

    def predict(
        self,
        subjective_state: MinimalSubjectiveState,
    ) -> Mapping[GeneralValueFunctionId, float]:
        return {"main": self._value}

    def update(
        self,
        transition: Transition[Action, MinimalSubjectiveState, MinimalInfo],
    ) -> Mapping[GeneralValueFunctionId, float]:
        self._value = transition.reward
        return {"main": 0.0}

    def add_or_replace(
        self,
        learner: GeneralValueFunctionLearner[
            MinimalSubjectiveState, Action, MinimalInfo
        ],
    ) -> None:
        pass

    def remove(
        self,
        general_value_function_ids: Sequence[GeneralValueFunctionId],
    ) -> None:
        pass


class MinimalUtilityAssessor(UtilityAssessor):
    """Aggregates usage records into simple counts."""

    def __init__(self) -> None:
        self._usage_records: list[UsageRecord] = []

    def observe(self, usage: Sequence[UsageRecord]) -> None:
        self._usage_records.extend(usage)

    def scores(self) -> Sequence[UtilityRecord]:
        totals: dict[tuple[str, str], float] = {}
        latest: dict[tuple[str, str], UsageRecord] = {}
        for record in self._usage_records:
            key = (record.kind.value, record.component_id)
            totals[key] = totals.get(key, 0.0) + record.amount
            latest[key] = record
        return tuple(
            UtilityRecord(
                kind=record.kind,
                component_id=record.component_id,
                utility=totals[key],
            )
            for key, record in latest.items()
        )


class MinimalCurator(Curator):
    """Never prunes."""

    def curate(self, utilities: Sequence[UtilityRecord]) -> CurationDecision:
        return CurationDecision()


# ─────────────────────────────────────────────────────────────────────
# Reactive-policy components
# ─────────────────────────────────────────────────────────────────────


@dataclass
class MinimalOption(Option[MinimalSubjectiveState, Action]):
    """Trivial option that always emits action=1 and stops immediately."""

    _descriptor: OptionDescriptor
    _action: Action = 1

    @property
    def descriptor(self) -> OptionDescriptor:
        return self._descriptor

    def is_available(self, subjective_state: MinimalSubjectiveState) -> bool:
        return True

    def act(self, subjective_state: MinimalSubjectiveState) -> Action:
        return self._action

    def stop_probability(self, subjective_state: MinimalSubjectiveState) -> float:
        return 1.0


class MinimalActionSelector(ActionSelector[MinimalSubjectiveState, Action]):
    """Alternates primitive actions and option selection."""

    def __init__(self) -> None:
        self.last_td_errors: Mapping[GeneralValueFunctionId, float] = {}
        self.last_planning_update: PlanningUpdate[Action] | None = None

    def decide(
        self,
        subjective_state: MinimalSubjectiveState,
        active_option: Option[MinimalSubjectiveState, Action] | None,
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


class MinimalOptionLibrary(OptionLibrary[MinimalSubjectiveState, Action]):
    """Stores learned options."""

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
    """Creates one trivial option per subtask."""

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
        transition: Transition[Action, MinimalSubjectiveState, MinimalInfo],
    ) -> None:
        pass

    def export_options(self) -> Sequence[Option[MinimalSubjectiveState, Action]]:
        return tuple(self._options.values())

    def remove_subtasks(self, subtask_ids: Sequence[SubtaskId]) -> None:
        for subtask_id in subtask_ids:
            self._subtasks.pop(subtask_id, None)
            self._options.pop(f"option:{subtask_id}", None)


# ─────────────────────────────────────────────────────────────────────
# Wiring
# ─────────────────────────────────────────────────────────────────────


def build_minimal_agent() -> (
    OaKAgent[Observation, Action, MinimalSubjectiveState, MinimalInfo]
):
    """Construct a fully wired fine-grained smoke-test OaK agent."""
    perception = CompositePerception(
        state_builder=MinimalStateBuilder(),
        feature_bank=MinimalFeatureBank(),
        feature_constructor=MinimalFeatureConstructor(),
        feature_ranker=MinimalFeatureRanker(),
        subtask_generator=MinimalSubtaskGenerator(),
    )
    transition_model = CompositeTransitionModel(
        world_model=MinimalWorldModel(),
        option_model_learner=MinimalOptionModelLearner(),
        planner=MinimalPlanner(),
    )
    value_function = CompositeValueFunction(
        value_estimator=MinimalValueEstimator(),
        utility_assessor=MinimalUtilityAssessor(),
        curator=MinimalCurator(),
    )
    action_selector = MinimalActionSelector()
    reactive_policy = CompositeReactivePolicy(
        action_selector=action_selector,
        option_library=MinimalOptionLibrary(),
        option_learner=MinimalOptionLearner(),
    )
    return OaKAgent(
        perception=perception,
        transition_model=transition_model,
        value_function=value_function,
        reactive_policy=reactive_policy,
        planning_budget=4,
    )


def run_minimal_episode(horizon: int = 5) -> list[MinimalTraceStep]:
    """Run a short smoke episode and return a compact trace."""
    world = MinimalWorld(horizon=horizon)
    agent = build_minimal_agent()
    step = world.reset()
    agent.reset()

    trace: list[MinimalTraceStep] = []

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
                "planning_budget_used": _planning_budget_used(result.planning_update),
            }
        )
        step = world.step(action)
        if step.terminated:
            break

    return trace
