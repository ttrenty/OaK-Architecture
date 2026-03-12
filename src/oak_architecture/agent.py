from __future__ import annotations
"""Reference agent wiring for the OaK architecture."""

from dataclasses import dataclass
from typing import Generic, Optional, Sequence

from .interfaces import (
    Curator,
    FeatureBank,
    FeatureConstructor,
    FeatureRanker,
    MetaStepSizeLearner,
    Option,
    OptionLearner,
    OptionLibrary,
    OptionModelLearner,
    Perception,
    Planner,
    ReactivePolicy,
    SubtaskGenerator,
    TransitionModel,
    UtilityAssessor,
    ValueFunction,
)
from .types import (
    ActT,
    AgentStepResult,
    ComponentKind,
    CurationDecision,
    FeatureId,
    InfoT,
    ObsT,
    OptionId,
    PlanningUpdate,
    StateT,
    SubtaskSpec,
    TimeStep,
    Transition,
    UsageRecord,
)


@dataclass
class OaKAgent(Generic[ObsT, ActT, StateT, InfoT]):
    """Coordinates one full OaK step across all registered components."""

    perception: Perception[ObsT, ActT, StateT]
    feature_bank: FeatureBank[StateT]
    feature_constructor: FeatureConstructor[StateT]
    feature_ranker: FeatureRanker
    subtask_generator: SubtaskGenerator[StateT]
    value_function: ValueFunction[StateT, ActT, InfoT]
    reactive_policy: ReactivePolicy[StateT, ActT]
    option_library: OptionLibrary[StateT, ActT]
    option_learner: OptionLearner[StateT, ActT, InfoT]
    option_model_learner: OptionModelLearner[StateT, ActT, InfoT]
    transition_model: TransitionModel[StateT, ActT, InfoT]
    planner: Planner[StateT, ActT, InfoT]
    utility_assessor: UtilityAssessor
    curator: Curator
    meta_step_sizes: Optional[MetaStepSizeLearner] = None
    planning_budget: int = 4
    feature_budget: int = 4
    option_stop_threshold: float = 0.5
    active_option: Optional[Option[StateT, ActT]] = None
    last_action: Optional[ActT] = None
    last_state: Optional[StateT] = None
    last_observation: Optional[ObsT] = None

    def reset(self) -> None:
        """Clear transient execution state."""
        self.perception.reset()
        self.active_option = None
        self.last_action = None
        self.last_state = None
        self.last_observation = None

    def step(
        self, time_step: TimeStep[ObsT, InfoT]
    ) -> AgentStepResult[ActT, StateT]:
        """Run one temporally uniform agent step."""
        state = self.perception.update(
            observation=time_step.observation,
            reward=time_step.reward,
            last_action=self.last_action,
        )

        created_subtasks: Sequence[SubtaskSpec] = ()
        ranked_feature_ids: Sequence[FeatureId] = ()
        planning_update: Optional[PlanningUpdate[ActT]] = None
        curation_decision: Optional[CurationDecision] = None

        if self.last_state is not None and self.last_action is not None:
            transition = Transition(
                state=self.last_state,
                action=self.last_action,
                reward=time_step.reward,
                next_state=state,
                observation=self.last_observation,
                next_observation=time_step.observation,
                terminated=time_step.terminated or time_step.truncated,
                info=time_step.info,
            )
            self._update_from_transition(transition)

        candidates = self.feature_constructor.propose(
            state,
            self.feature_bank.list_features(),
        )
        if candidates:
            self.feature_bank.add_candidates(candidates)

        ranked_feature_ids = self.feature_ranker.rank(
            self.feature_bank.list_features(),
            self.utility_assessor.scores(),
            limit=self.feature_budget,
        )
        if ranked_feature_ids:
            created_subtasks = self.subtask_generator.generate(
                ranked_feature_ids, self.feature_bank
            )
            if created_subtasks:
                self.option_learner.ingest_subtasks(created_subtasks)

        for option in self.option_learner.export_options():
            self.option_library.add_or_replace(option)
        self.transition_model.add_or_replace_option_models(
            self.option_model_learner.export_models()
        )

        planning_update = self.planner.plan_step(
            state=state,
            model=self.transition_model,
            value_function=self.value_function,
            budget=self.planning_budget,
        )
        self.reactive_policy.apply_planning_update(planning_update)

        action, active_option_id = self._select_action(state)

        usage_records = self._build_usage_records(ranked_feature_ids, active_option_id)
        if usage_records:
            self.utility_assessor.observe(usage_records)

        utility_scores = self.utility_assessor.scores()
        if utility_scores:
            curation_decision = self.curator.curate(utility_scores)
            self._apply_curation(curation_decision)

        self.last_state = state
        self.last_observation = time_step.observation
        self.last_action = action

        if time_step.terminated or time_step.truncated:
            self.active_option = None

        return AgentStepResult(
            action=action,
            state=state,
            active_option_id=active_option_id,
            planning_update=planning_update,
            created_subtasks=created_subtasks,
            curation_decision=curation_decision,
        )

    def _update_from_transition(
        self,
        transition: Transition[ObsT, ActT, StateT, InfoT],
    ) -> None:
        """Apply one observed transition to the learning subsystems."""
        td_errors = self.value_function.update(transition)
        self.reactive_policy.update_from_values(
            transition.next_state,
            td_errors,
        )
        self.option_learner.update(transition)
        self.option_model_learner.update(transition)
        self.transition_model.update(transition)

        if self.meta_step_sizes is not None:
            self.meta_step_sizes.update("value_functions", td_errors)
            self.meta_step_sizes.update(
                "transition_model",
                {"reward": transition.reward},
            )

    def _select_action(
        self,
        state: StateT,
    ) -> tuple[ActT, Optional[OptionId]]:
        """Select a primitive action, continuing any active option if needed."""
        if self.active_option is not None:
            stop_probability = self.active_option.stop_probability(state)
            if stop_probability < self.option_stop_threshold:
                return (
                    self.active_option.act(state),
                    self.active_option.descriptor.option_id,
                )
            self.active_option = None

        decision = self.reactive_policy.decide(
            state=state,
            active_option=None,
            available_options=self.option_library.list_options(),
        )

        if decision.option_id is not None:
            self.active_option = self.option_library.get(decision.option_id)
            return (
                self.active_option.act(state),
                self.active_option.descriptor.option_id,
            )

        if decision.action is None:
            raise RuntimeError(
                "ReactivePolicy returned neither a primitive action nor an option."
            )

        return decision.action, None

    def _build_usage_records(
        self,
        ranked_feature_ids: Sequence[FeatureId],
        active_option_id: Optional[OptionId],
    ) -> Sequence[UsageRecord]:
        """Build minimal utility-accounting observations for the current step."""
        usage_records = [
            UsageRecord(ComponentKind.FEATURE, feature_id)
            for feature_id in ranked_feature_ids
        ]
        if active_option_id is not None:
            usage_records.append(UsageRecord(ComponentKind.OPTION, active_option_id))
        return tuple(usage_records)

    def _apply_curation(self, decision: CurationDecision) -> None:
        """Apply curator pruning decisions to the current agent state."""
        if decision.drop_features:
            self.feature_bank.remove(decision.drop_features)
        if decision.drop_subtasks:
            self.option_learner.remove_subtasks(decision.drop_subtasks)
        if decision.drop_options:
            self.option_library.remove(decision.drop_options)
            if (
                self.active_option is not None
                and self.active_option.descriptor.option_id in decision.drop_options
            ):
                self.active_option = None
        if decision.drop_option_models:
            self.transition_model.remove_option_models(decision.drop_option_models)
        if decision.drop_gvfs:
            self.value_function.remove(decision.drop_gvfs)
