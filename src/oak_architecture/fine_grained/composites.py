from __future__ import annotations

"""Composite implementations of the four main OaK interfaces.

Each composite builds one of the four main interfaces from finer-grained
component interfaces (defined in
`oak_architecture.fine_grained.components`).  Use these when you want to
independently swap individual building blocks inside a module.

If you prefer a simpler setup, implement the four main interfaces directly.
"""

from typing import Generic, Mapping, Sequence

from .components import (
    ActionSelector,
    Curator,
    FeatureBank,
    FeatureConstructor,
    FeatureRanker,
    GeneralValueFunctionLearner,
    MetaStepSizeLearner,
    Option,
    OptionLearner,
    OptionLibrary,
    OptionModelLearner,
    Planner,
    StateBuilder,
    SubtaskGenerator,
    UtilityAssessor,
    ValueEstimator,
    WorldModel,
)
from ..interfaces import (
    Perception,
    ReactivePolicy,
    TransitionModel,
    ValueFunction,
)
from ..types import (
    ActionT,
    ComponentId,
    CurationDecision,
    FeatureId,
    FeatureSpec,
    GeneralValueFunctionId,
    InfoT,
    ObservationT,
    OptionId,
    PlanningUpdate,
    SubjectiveStateT,
    SubtaskId,
    SubtaskSpec,
    Transition,
    UsageRecord,
    UtilityRecord,
)


# ─────────────────────────────────────────────────────────────────────
# CompositePerception
# ─────────────────────────────────────────────────────────────────────


class CompositePerception(
    Perception[ObservationT, ActionT, SubjectiveStateT],
    Generic[ObservationT, ActionT, SubjectiveStateT],
):
    """Perception built from fine-grained components.

    Components: `StateBuilder`, `FeatureBank`, `FeatureConstructor`,
    `FeatureRanker`, `SubtaskGenerator`.
    """

    def __init__(
        self,
        state_builder: StateBuilder[ObservationT, ActionT, SubjectiveStateT],
        feature_bank: FeatureBank[SubjectiveStateT],
        feature_constructor: FeatureConstructor[SubjectiveStateT],
        feature_ranker: FeatureRanker,
        subtask_generator: SubtaskGenerator[SubjectiveStateT],
    ) -> None:
        self._state_builder = state_builder
        self._feature_bank = feature_bank
        self._feature_constructor = feature_constructor
        self._feature_ranker = feature_ranker
        self._subtask_generator = subtask_generator

    def reset(self) -> None:
        self._state_builder.reset()

    def update(
        self,
        observation: ObservationT,
        reward: float,
        last_action: ActionT | None,
    ) -> SubjectiveStateT:
        return self._state_builder.update(observation, reward, last_action)

    def current_subjective_state(self) -> SubjectiveStateT:
        return self._state_builder.current_subjective_state()

    def discover_and_rank_features(
        self,
        subjective_state: SubjectiveStateT,
        utility_scores: Sequence[UtilityRecord],
        feature_budget: int,
    ) -> Sequence[FeatureId]:
        candidates = self._feature_constructor.propose(
            subjective_state, self._feature_bank.list_features()
        )
        if candidates:
            self._feature_bank.add_candidates(candidates)
        return self._feature_ranker.rank(
            self._feature_bank.list_features(), utility_scores, limit=feature_budget
        )

    def generate_subtasks(
        self,
        ranked_feature_ids: Sequence[FeatureId],
    ) -> Sequence[SubtaskSpec]:
        return self._subtask_generator.generate(ranked_feature_ids, self._feature_bank)

    def list_features(self) -> Sequence[FeatureSpec]:
        return self._feature_bank.list_features()

    def remove_features(self, feature_ids: Sequence[FeatureId]) -> None:
        self._feature_bank.remove(feature_ids)


# ─────────────────────────────────────────────────────────────────────
# CompositeValueFunction
# ─────────────────────────────────────────────────────────────────────


class CompositeValueFunction(
    ValueFunction[SubjectiveStateT, ActionT, InfoT],
    Generic[SubjectiveStateT, ActionT, InfoT],
):
    """ValueFunction built from fine-grained components.

    Components: `ValueEstimator`, `UtilityAssessor`, `Curator`,
    and optionally `MetaStepSizeLearner`.
    """

    def __init__(
        self,
        value_estimator: ValueEstimator[SubjectiveStateT, ActionT, InfoT],
        utility_assessor: UtilityAssessor,
        curator: Curator,
        meta_step_sizes: MetaStepSizeLearner | None = None,
    ) -> None:
        self._value_estimator = value_estimator
        self._utility_assessor = utility_assessor
        self._curator = curator
        self._meta_step_sizes = meta_step_sizes

    def update(
        self,
        transition: Transition[ActionT, SubjectiveStateT, InfoT],
    ) -> Mapping[GeneralValueFunctionId, float]:
        td_errors = self._value_estimator.update(transition)
        self.update_meta("value_functions", td_errors)
        self.update_meta("transition_model", {"reward": transition.reward})
        return td_errors

    def predict(
        self,
        subjective_state: SubjectiveStateT,
    ) -> Mapping[GeneralValueFunctionId, float]:
        return self._value_estimator.predict(subjective_state)

    def observe_usage(self, usage_records: Sequence[UsageRecord]) -> None:
        self._utility_assessor.observe(usage_records)

    def utility_scores(self) -> Sequence[UtilityRecord]:
        return self._utility_assessor.scores()

    def curate(self) -> CurationDecision:
        scores = self._utility_assessor.scores()
        if not scores:
            return CurationDecision()
        return self._curator.curate(scores)

    def remove(
        self,
        general_value_function_ids: Sequence[GeneralValueFunctionId],
    ) -> None:
        self._value_estimator.remove(general_value_function_ids)

    def update_meta(
        self,
        component_id: ComponentId,
        error_signals: Mapping[str, float],
    ) -> None:
        if self._meta_step_sizes is not None:
            self._meta_step_sizes.update(component_id, error_signals)


# ─────────────────────────────────────────────────────────────────────
# CompositeTransitionModel
# ─────────────────────────────────────────────────────────────────────


class _ValueEstimatorAdapter(
    ValueEstimator[SubjectiveStateT, ActionT, InfoT],
    Generic[SubjectiveStateT, ActionT, InfoT],
):
    """Adapts a main `ValueFunction` to the `ValueEstimator` interface.

    Used internally by `CompositeTransitionModel` so that the
    fine-grained `Planner` (which expects a `ValueEstimator`) can
    receive the module-level `ValueFunction` passed into `plan()`.
    """

    def __init__(
        self,
        value_function: ValueFunction[SubjectiveStateT, ActionT, InfoT],
    ) -> None:
        self._vf = value_function

    def list_general_value_functions(
        self,
    ) -> Sequence[GeneralValueFunctionLearner[SubjectiveStateT, ActionT, InfoT]]:
        return ()

    def predict(
        self,
        subjective_state: SubjectiveStateT,
    ) -> Mapping[GeneralValueFunctionId, float]:
        return self._vf.predict(subjective_state)

    def update(
        self,
        transition: Transition[ActionT, SubjectiveStateT, InfoT],
    ) -> Mapping[GeneralValueFunctionId, float]:
        return self._vf.update(transition)

    def add_or_replace(
        self,
        learner: GeneralValueFunctionLearner[SubjectiveStateT, ActionT, InfoT],
    ) -> None:
        pass

    def remove(
        self,
        general_value_function_ids: Sequence[GeneralValueFunctionId],
    ) -> None:
        self._vf.remove(general_value_function_ids)


class CompositeTransitionModel(
    TransitionModel[SubjectiveStateT, ActionT, InfoT],
    Generic[SubjectiveStateT, ActionT, InfoT],
):
    """TransitionModel built from fine-grained components.

    Components: `WorldModel`, `OptionModelLearner`, `Planner`.
    """

    def __init__(
        self,
        world_model: WorldModel[SubjectiveStateT, ActionT, InfoT],
        option_model_learner: OptionModelLearner[SubjectiveStateT, ActionT, InfoT],
        planner: Planner[SubjectiveStateT, ActionT, InfoT],
    ) -> None:
        self._world_model = world_model
        self._option_model_learner = option_model_learner
        self._planner = planner

    def update(
        self,
        transition: Transition[ActionT, SubjectiveStateT, InfoT],
    ) -> None:
        self._world_model.update(transition)
        self._option_model_learner.update(transition)

    def integrate_option_models(self) -> None:
        models = self._option_model_learner.export_models()
        self._world_model.add_or_replace_option_models(models)

    def plan(
        self,
        subjective_state: SubjectiveStateT,
        value_function: ValueFunction[SubjectiveStateT, ActionT, InfoT],
        budget: int,
    ) -> PlanningUpdate[ActionT]:
        adapter = _ValueEstimatorAdapter(value_function)
        return self._planner.plan_step(
            subjective_state, self._world_model, adapter, budget
        )

    def remove_option_models(self, option_ids: Sequence[OptionId]) -> None:
        self._world_model.remove_option_models(option_ids)


# ─────────────────────────────────────────────────────────────────────
# CompositeReactivePolicy
# ─────────────────────────────────────────────────────────────────────


class CompositeReactivePolicy(
    ReactivePolicy[SubjectiveStateT, ActionT, InfoT],
    Generic[SubjectiveStateT, ActionT, InfoT],
):
    """ReactivePolicy built from fine-grained components.

    Components: `ActionSelector`, `OptionLibrary`, `OptionLearner`.
    """

    def __init__(
        self,
        action_selector: ActionSelector[SubjectiveStateT, ActionT],
        option_library: OptionLibrary[SubjectiveStateT, ActionT],
        option_learner: OptionLearner[SubjectiveStateT, ActionT, InfoT],
    ) -> None:
        self._action_selector = action_selector
        self._option_library = option_library
        self._option_learner = option_learner
        self._active_option: Option[SubjectiveStateT, ActionT] | None = None

    def update(
        self,
        transition: Transition[ActionT, SubjectiveStateT, InfoT],
        td_errors: Mapping[GeneralValueFunctionId, float],
    ) -> None:
        self._action_selector.update_from_values(
            transition.next_subjective_state, td_errors
        )
        self._option_learner.update(transition)

    def apply_planning_update(self, update: PlanningUpdate[ActionT]) -> None:
        self._action_selector.apply_planning_update(update)

    def ingest_subtasks(self, subtasks: Sequence[SubtaskSpec]) -> None:
        self._option_learner.ingest_subtasks(subtasks)

    def integrate_options(self) -> None:
        for option in self._option_learner.export_options():
            self._option_library.add_or_replace(option)

    def select_action(
        self,
        subjective_state: SubjectiveStateT,
        option_stop_threshold: float,
    ) -> tuple[ActionT, OptionId | None]:
        if self._active_option is not None:
            stop_prob = self._active_option.stop_probability(subjective_state)
            if stop_prob < option_stop_threshold:
                return (
                    self._active_option.act(subjective_state),
                    self._active_option.descriptor.option_id,
                )
            self._active_option = None

        decision = self._action_selector.decide(
            subjective_state=subjective_state,
            active_option=None,
            available_options=self._option_library.list_options(),
        )

        if decision.option_id is not None:
            self._active_option = self._option_library.get(decision.option_id)
            return (
                self._active_option.act(subjective_state),
                self._active_option.descriptor.option_id,
            )

        if decision.action is None:
            raise RuntimeError(
                "ActionSelector returned neither a primitive action nor an option."
            )

        return decision.action, None

    def clear_active_option(self) -> None:
        self._active_option = None

    def remove_options(self, option_ids: Sequence[OptionId]) -> None:
        self._option_library.remove(option_ids)
        if (
            self._active_option is not None
            and self._active_option.descriptor.option_id in option_ids
        ):
            self._active_option = None

    def remove_subtasks(self, subtask_ids: Sequence[SubtaskId]) -> None:
        self._option_learner.remove_subtasks(subtask_ids)
