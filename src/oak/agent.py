from __future__ import annotations

"""Reference agent wiring for the OaK architecture.

`OaKAgent` is the runtime coordinator.  It composes the four OaK
interfaces (Perception, Transition Model, Value Function, and Reactive
Policy) and defines the order in which they are called during one
temporally uniform step.

At a high level, one `step(...)` call follows six phases:

1. **Perceive**: update perception to obtain the current subjective state.
2. **Learn**: if a previous state/action exists, update all modules from
   the observed transition.
3. **Grow**: discover and rank features, generate subtasks, integrate
   new options and option models.
4. **Plan**: run bounded planning and inform the reactive policy responsible for action selection.
5. **Act**: select the next primitive action.
6. **Maintain**: record utility evidence and apply curation decisions.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Self, Sequence

from .interfaces import (
    Perception,
    ReactivePolicy,
    TransitionModel,
    ValueFunction,
    World,
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
    """Coordinates one full OaK step across the four modules.

    The agent is a wiring object: you provide concrete implementations of
    `Perception`, `TransitionModel`, `ValueFunction`, and
    `ReactivePolicy`, and `OaKAgent` ensures they are called in a
    consistent order.
    """

    perception: Perception[ObservationT, ActionT, SubjectiveStateT]
    transition_model: TransitionModel[SubjectiveStateT, ActionT, InfoT]
    value_function: ValueFunction[SubjectiveStateT, ActionT, InfoT]
    reactive_policy: ReactivePolicy[SubjectiveStateT, ActionT, InfoT]

    planning_budget: int = 4
    feature_budget: int = 4
    option_stop_threshold: float = 0.5

    last_action: ActionT | None = None
    last_subjective_state: SubjectiveStateT | None = None

    def __init__(
        self,
        perception: Perception[ObservationT, ActionT, SubjectiveStateT],
        transition_model: TransitionModel[SubjectiveStateT, ActionT, InfoT],
        value_function: ValueFunction[SubjectiveStateT, ActionT, InfoT],
        reactive_policy: ReactivePolicy[SubjectiveStateT, ActionT, InfoT],
        planning_budget: int = 4,
        feature_budget: int = 4,
        option_stop_threshold: float = 0.5,
    ):
        self.perception = perception
        self.transition_model = transition_model
        self.value_function = value_function
        self.reactive_policy = reactive_policy
        self.planning_budget = planning_budget
        self.feature_budget = feature_budget
        self.option_stop_threshold = option_stop_threshold
        self.last_action = None
        self.last_subjective_state = None

    def __post_init__(self):
        """Validate that the modules are compatible."""
        if self.planning_budget < 1:
            raise ValueError("Planning budget must be at least 1.")
        if self.feature_budget < 1:
            raise ValueError("Feature budget must be at least 1.")
        if self.option_stop_threshold < 0 or self.option_stop_threshold > 1:
            raise ValueError("Option stop threshold must be in [0, 1].")

    def reset(self) -> None:
        """Clear transient execution memory."""
        self.perception.reset()
        self.reactive_policy.clear_active_option()
        self.last_action = None
        self.last_subjective_state = None

    def step(
        self, time_step: TimeStep[ObservationT, InfoT]
    ) -> AgentStepResult[ActionT, SubjectiveStateT]:
        """Run one temporally uniform agent step.

        The step follows six phases: perceive, learn, grow, plan, act, maintain.
        """

        # ================== 1. Perceive ================= #
        subjective_state = self.perception.update(
            observation=time_step.observation,
            reward=time_step.reward,
            last_action=self.last_action,
        )

        created_subtasks: Sequence[SubtaskSpec] = ()
        ranked_feature_ids: Sequence[FeatureId] = ()
        planning_update: PlanningUpdate[ActionT] | None = None
        curation_decision: CurationDecision | None = None

        # ================== 2. Learn ================== #
        if self.last_subjective_state is not None and self.last_action is not None:
            transition = Transition(
                subjective_state=self.last_subjective_state,
                action=self.last_action,
                reward=time_step.reward,
                next_subjective_state=subjective_state,
                terminated=time_step.terminated or time_step.truncated,
                info=time_step.info,
            )
            td_errors = self.value_function.update(transition)
            self.reactive_policy.update(transition, td_errors)
            self.transition_model.update(transition)

            # Meta step-size adaptation (Sutton's IDBD / online cross-validation)
            meta_signals = dict(td_errors)
            meta_signals["reward"] = transition.reward
            self.perception.update_meta(meta_signals)
            self.value_function.update_meta(meta_signals)
            self.reactive_policy.update_meta(meta_signals)
            self.transition_model.update_meta(meta_signals)

        # ================== 3. Grow ================== #
        ranked_feature_ids = self.perception.discover_and_rank_features(
            subjective_state,
            self.value_function.utility_scores(),
            self.feature_budget,
        )
        if ranked_feature_ids:
            created_subtasks = self.perception.generate_subtasks(ranked_feature_ids)
            if created_subtasks:
                self.reactive_policy.ingest_subtasks(created_subtasks)

        self.reactive_policy.integrate_options()
        self.transition_model.integrate_option_models()

        # ================== 4. Plan ================== #
        planning_update = self.transition_model.plan(
            subjective_state, self.value_function, self.planning_budget
        )
        self.reactive_policy.apply_planning_update(planning_update)

        # ================== 5. Act ================== #
        action, active_option_id = self.reactive_policy.select_action(
            subjective_state, self.option_stop_threshold
        )

        # ================== 6. Maintain ================== #
        usage_records = self._build_usage_records(ranked_feature_ids, active_option_id)
        if usage_records:
            self.value_function.observe_usage(usage_records)

        curation_decision = self.value_function.curate()
        self._apply_curation(curation_decision)

        # ================== Update Memory ================== #
        self.last_subjective_state = subjective_state
        self.last_action = action

        if time_step.terminated or time_step.truncated:
            self.reactive_policy.clear_active_option()

        return AgentStepResult(
            action=action,
            subjective_state=subjective_state,
            active_option_id=active_option_id,
            planning_update=planning_update,
            created_subtasks=created_subtasks,
            curation_decision=curation_decision,
        )

    # ── training loop ─────────────────────────────────────────────────

    def train(
        self,
        world: World[ObservationT, ActionT, InfoT],
        *,  # enforce keyword arguments after for clarity
        num_episodes: int = 500,
        average_window: int = 100,
        solved_threshold: float | None = None,
        episode_logger: Callable[[int, float, float, Self], None] | None = None,
    ) -> list[float]:
        """Run the standard OaK episode loop on the given world.

        Parameters
        ----------
        world:
            An environment implementing the `World` protocol.
        num_episodes:
            Maximum number of training episodes.
        average_window:
            Number of recent episodes to average for performance tracking.
        solved_threshold:
            If set, stop early when the average reward over the last `average_window`
            episodes reaches this value.
        episode_logger:
            Optional callback `(episode, episode_reward, avg_reward, agent)`
            called after each episode. Use this to own all per-episode logging
            or other training-side effects at the call site.

        Returns
        -------
        list[float]
            Per-episode reward history.
        """
        if average_window < 1:
            raise ValueError("average_window must be at least 1.")

        reward_history: list[float] = []

        for episode in range(num_episodes):
            time_step = world.reset()
            self.reset()
            episode_reward = 0.0

            while True:
                result = self.step(time_step)

                if time_step.terminated or time_step.truncated:
                    break

                time_step = world.step(result.action)
                episode_reward += time_step.reward

            reward_history.append(episode_reward)
            recent_window = reward_history[-average_window:]
            avg_reward = sum(recent_window) / len(recent_window)

            solved = (
                solved_threshold is not None
                and len(reward_history) >= average_window
                and avg_reward >= solved_threshold
            )

            if episode_logger is not None:
                episode_logger(episode, episode_reward, avg_reward, self)

            if solved:
                break

        return reward_history

    # ── private helpers ──────────────────────────────────────────────

    def _build_usage_records(
        self,
        ranked_feature_ids: Sequence[FeatureId],
        active_option_id: OptionId | None,
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
        """Dispatch curation decisions to the relevant modules."""
        if decision.drop_features:
            self.perception.remove_features(decision.drop_features)
        if decision.drop_subtasks:
            self.reactive_policy.remove_subtasks(decision.drop_subtasks)
        if decision.drop_options:
            self.reactive_policy.remove_options(decision.drop_options)
        if decision.drop_option_models:
            self.transition_model.remove_option_models(decision.drop_option_models)
        if decision.drop_general_value_functions:
            self.value_function.remove(decision.drop_general_value_functions)
