from __future__ import annotations

"""Reference agent wiring for the OaK architecture.

`OaKAgent` is the runtime coordinator for the package. It does not implement
the concrete learning algorithms itself; instead it defines the order in which
the supplied components are called during one temporally uniform step.

At a high level, one `step(...)` call does the following:

1. Update `Perception` to obtain the current subjective state.
2. If a previous subjective_state/action exists, build a `Transition` and update the
   learning subsystems.
3. Grow or rank features and subtasks.
4. Refresh options and option models.
5. Ask the planner for a bounded planning update.
6. Let the reactive policy choose the next primitive action or option.
7. Record utility evidence and apply any curation decision.
"""

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
    ActionT,
    AgentStepResult,
    ComponentKind,
    CurationDecision,
    FeatureId,
    InfoT,
    ObservationT,
    OptionId,
    PlanningUpdate,
    SubjectiveStateT,
    SubtaskSpec,
    TimeStep,
    Transition,
    UsageRecord,
)


@dataclass
class OaKAgent(Generic[ObservationT, ActionT, SubjectiveStateT, InfoT]):
    """Coordinates one full OaK step across all registered components.

    The agent is deliberately interface-first. It is best understood as a
    wiring object: you provide the concrete implementations, and `OaKAgent`
    ensures they are invoked in a consistent order.
    """

    perception: Perception[ObservationT, ActionT, SubjectiveStateT]
    feature_bank: FeatureBank[SubjectiveStateT]
    feature_constructor: FeatureConstructor[SubjectiveStateT]
    feature_ranker: FeatureRanker
    subtask_generator: SubtaskGenerator[SubjectiveStateT]
    value_function: ValueFunction[SubjectiveStateT, ActionT, InfoT]
    reactive_policy: ReactivePolicy[SubjectiveStateT, ActionT]
    option_library: OptionLibrary[SubjectiveStateT, ActionT]
    option_learner: OptionLearner[SubjectiveStateT, ActionT, InfoT]
    option_model_learner: OptionModelLearner[SubjectiveStateT, ActionT, InfoT]
    transition_model: TransitionModel[SubjectiveStateT, ActionT, InfoT]
    planner: Planner[SubjectiveStateT, ActionT, InfoT]
    utility_assessor: UtilityAssessor
    curator: Curator
    meta_step_sizes: Optional[MetaStepSizeLearner] = None
    planning_budget: int = 4
    feature_budget: int = 4
    option_stop_threshold: float = 0.5
    active_option: Optional[Option[SubjectiveStateT, ActionT]] = None
    last_action: Optional[ActionT] = None
    last_subjective_state: Optional[SubjectiveStateT] = None
    last_observation: Optional[ObservationT] = None

    def reset(self) -> None:
        """Clear transient execution memory."""
        self.perception.reset()
        self.active_option = None
        self.last_action = None
        self.last_subjective_state = None
        self.last_observation = None

    def step(
        self, time_step: TimeStep[ObservationT, InfoT]
    ) -> AgentStepResult[ActionT, SubjectiveStateT]:
        """Run one temporally uniform agent step."""
        subjective_state = self.perception.update(
            observation=time_step.observation,
            reward=time_step.reward,
            last_action=self.last_action,
        )

        created_subtasks: Sequence[SubtaskSpec] = ()
        ranked_feature_ids: Sequence[FeatureId] = ()
        planning_update: Optional[PlanningUpdate[ActionT]] = None
        curation_decision: Optional[CurationDecision] = None

        if self.last_subjective_state is not None and self.last_action is not None:
            transition = Transition(
                subjective_state=self.last_subjective_state,
                action=self.last_action,
                reward=time_step.reward,
                next_subjective_state=subjective_state,
                observation=self.last_observation,
                next_observation=time_step.observation,
                terminated=time_step.terminated or time_step.truncated,
                info=time_step.info,
            )
            self._update_from_transition(transition)

        candidates = self.feature_constructor.propose(
            subjective_state,
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
            subjective_state=subjective_state,
            model=self.transition_model,
            value_function=self.value_function,
            budget=self.planning_budget,
        )
        self.reactive_policy.apply_planning_update(planning_update)

        action, active_option_id = self._select_action(subjective_state)

        usage_records = self._build_usage_records(ranked_feature_ids, active_option_id)
        if usage_records:
            self.utility_assessor.observe(usage_records)

        utility_scores = self.utility_assessor.scores()
        if utility_scores:
            curation_decision = self.curator.curate(utility_scores)
            self._apply_curation(curation_decision)

        self.last_subjective_state = subjective_state
        self.last_observation = time_step.observation
        self.last_action = action

        if time_step.terminated or time_step.truncated:
            self.active_option = None

        return AgentStepResult(
            action=action,
            subjective_state=subjective_state,
            active_option_id=active_option_id,
            planning_update=planning_update,
            created_subtasks=created_subtasks,
            curation_decision=curation_decision,
        )

    def _update_from_transition(
        self,
        transition: Transition[ObservationT, ActionT, SubjectiveStateT, InfoT],
    ) -> None:
        """Apply one observed transition to the learning subsystems."""
        td_errors = self.value_function.update(transition)
        self.reactive_policy.update_from_values(
            transition.next_subjective_state,
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
        subjective_state: SubjectiveStateT,
    ) -> tuple[ActionT, Optional[OptionId]]:
        """Select a primitive action, continuing any active option if needed."""
        if self.active_option is not None:
            stop_probability = self.active_option.stop_probability(subjective_state)
            if stop_probability < self.option_stop_threshold:
                return (
                    self.active_option.act(subjective_state),
                    self.active_option.descriptor.option_id,
                )
            self.active_option = None

        decision = self.reactive_policy.decide(
            subjective_state=subjective_state,
            active_option=None,
            available_options=self.option_library.list_options(),
        )

        if decision.option_id is not None:
            self.active_option = self.option_library.get(decision.option_id)
            return (
                self.active_option.act(subjective_state),
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
        """Apply curator pruning decisions to the current live agent fields."""
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
        if decision.drop_general_value_functions:
            self.value_function.remove(decision.drop_general_value_functions)
