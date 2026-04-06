from __future__ import annotations

"""Bare-minimum external implementation used to smoke-test the interface.

This module answers a single question: can the current package interfaces be
instantiated and run through a complete OaK step loop?

The implementation shows the **direct** approach: each of Sutton's four
modules (Perception, Transition Model, Value Function, Reactive Policy) is
implemented as a single class.  There is no need to use the fine-grained
component interfaces or the composite wrappers, which exist for projects
that need more modularity inside each module.

What this module is:

- a tiny integer world
- a direct observation-to-subjective_state perception with one fixed feature
- a no-op transition model with trivial one-step planning
- a simple value tracker with usage counting and no curation
- a reactive policy that alternates actions and options

What this module is not:

- a trained agent
- a realistic planner
- a serious option-learning system
- a benchmark implementation
"""

from dataclasses import dataclass
from typing import Mapping, Sequence, TypedDict

from oak.agent import OaKAgent
from oak.interfaces import (
    Perception,
    ReactivePolicy,
    TransitionModel,
    ValueFunction,
    World,
)
from oak.types import (
    CurationDecision,
    FeatureId,
    FeatureSpec,
    GeneralValueFunctionId,
    OptionDescriptor,
    OptionId,
    PlanningUpdate,
    SubtaskId,
    SubtaskSpec,
    TimeStep,
    Transition,
    UsageRecord,
    UtilityRecord,
)

Observation = int
Action = int


class MinimalInfo(TypedDict, total=False):
    reset: bool
    echo_action: Action


class MinimalTraceStep(TypedDict):
    subjective_state: "MinimalSubjectiveState"
    action: Action
    active_option_id: OptionId | None
    created_subtasks: list[SubtaskId]
    planning_budget_used: int | None


def _planning_budget_used(update: PlanningUpdate[Action] | None) -> int | None:
    """Extract an integer planning budget from structured search statistics."""
    if update is None:
        return None

    value = update.search_statistics.get("budget_used")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


@dataclass(slots=True, frozen=True)
class MinimalSubjectiveState:
    """Small concrete subjective state used by the smoke implementation."""

    step_index: int
    observation: Observation
    reward: float
    last_action: Action | None


# ─────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────


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

    def close(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# Perception
# ─────────────────────────────────────────────────────────────────────


class MinimalPerception(Perception[Observation, Action, MinimalSubjectiveState]):
    """Direct observation-to-state mapping with one fixed feature.

    - The subjective state is a thin wrapper around the observation.
    - One identity feature ("observation") is always present.
    - No new features are ever proposed.
    - One subtask is created per feature (deduplicated).
    """

    def __init__(self) -> None:
        self._state = MinimalSubjectiveState(0, 0, 0.0, None)
        self._features: dict[FeatureId, FeatureSpec] = {
            "observation": FeatureSpec(
                feature_id="observation",
                name="Observation value",
                description="Identity feature for the integer observation.",
            )
        }
        self._created_subtask_for: set[FeatureId] = set()

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

    def discover_and_rank_features(
        self,
        subjective_state: MinimalSubjectiveState,
        utility_scores: Sequence[UtilityRecord],
        feature_budget: int,
    ) -> Sequence[FeatureId]:
        # No new features proposed; rank existing ones in insertion order.
        ids = list(self._features.keys())
        return tuple(ids[:feature_budget])

    def generate_subtasks(
        self,
        ranked_feature_ids: Sequence[FeatureId],
    ) -> Sequence[SubtaskSpec]:
        created: list[SubtaskSpec] = []
        for fid in ranked_feature_ids:
            if fid in self._created_subtask_for:
                continue
            self._created_subtask_for.add(fid)
            created.append(
                SubtaskSpec(
                    subtask_id=f"subtask:{fid}",
                    name=f"Track {fid}",
                    feature_id=fid,
                )
            )
        return tuple(created)

    def list_features(self) -> Sequence[FeatureSpec]:
        return tuple(self._features.values())

    def remove_features(self, feature_ids: Sequence[FeatureId]) -> None:
        for fid in feature_ids:
            self._features.pop(fid, None)


# ─────────────────────────────────────────────────────────────────────
# Transition Model
# ─────────────────────────────────────────────────────────────────────


class MinimalTransitionModel(
    TransitionModel[MinimalSubjectiveState, Action, MinimalInfo]
):
    """Trivial world model with one-step lookahead planning.

    - No real model learning (update is a no-op).
    - No option models.
    - Planning calls predict once and returns value targets.
    """

    def update(
        self,
        transition: Transition[Action, MinimalSubjectiveState, MinimalInfo],
    ) -> None:
        pass

    def integrate_option_models(self) -> None:
        pass

    def plan(
        self,
        subjective_state: MinimalSubjectiveState,
        value_function: ValueFunction[MinimalSubjectiveState, Action, MinimalInfo],
        budget: int,
    ) -> PlanningUpdate[Action]:
        return PlanningUpdate(
            value_targets=value_function.predict(subjective_state),
            policy_targets={"preferred_action": 0},
            search_statistics={"budget_used": budget},
        )

    def remove_option_models(self, option_ids: Sequence[OptionId]) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# Value Function
# ─────────────────────────────────────────────────────────────────────


class MinimalValueFunction(ValueFunction[MinimalSubjectiveState, Action, MinimalInfo]):
    """Stores latest reward as a value, counts usage, never curates.

    - One implicit value learner ("main") that stores the latest reward.
    - Usage records are accumulated for utility scoring.
    - Curation always returns an empty decision (no pruning).
    """

    def __init__(self) -> None:
        self._value: float = 0.0
        self._usage_records: list[UsageRecord] = []

    def update(
        self,
        transition: Transition[Action, MinimalSubjectiveState, MinimalInfo],
        *,
        planning: bool = False,
    ) -> Mapping[GeneralValueFunctionId, float]:
        self._value = transition.reward
        return {"main": 0.0}

    def predict(
        self,
        subjective_state: MinimalSubjectiveState,
    ) -> Mapping[GeneralValueFunctionId, float]:
        return {"main": self._value}

    def observe_usage(self, usage_records: Sequence[UsageRecord]) -> None:
        self._usage_records.extend(usage_records)

    def utility_scores(self) -> Sequence[UtilityRecord]:
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

    def curate(self) -> CurationDecision:
        return CurationDecision()

    def remove(
        self,
        general_value_function_ids: Sequence[GeneralValueFunctionId],
    ) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# Reactive Policy
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _MinimalOption:
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


class MinimalReactivePolicy(
    ReactivePolicy[MinimalSubjectiveState, Action, MinimalInfo]
):
    """Alternates primitive actions and options, creates options from subtasks.

    - On even observations: primitive action 0.
    - On odd observations with options available: executes the first option.
    - On odd observations without options: primitive action 1.
    - Options are created 1:1 from ingested subtasks.
    """

    def __init__(self) -> None:
        self._active_option: _MinimalOption | None = None
        self._options: dict[OptionId, _MinimalOption] = {}
        self._subtasks: dict[SubtaskId, SubtaskSpec] = {}
        self.last_td_errors: Mapping[GeneralValueFunctionId, float] = {}
        self.last_planning_update: PlanningUpdate[Action] | None = None

    def update(
        self,
        transition: Transition[Action, MinimalSubjectiveState, MinimalInfo],
        td_errors: Mapping[GeneralValueFunctionId, float],
    ) -> None:
        self.last_td_errors = dict(td_errors)

    def apply_planning_update(self, update: PlanningUpdate[Action]) -> None:
        self.last_planning_update = update

    def ingest_subtasks(self, subtasks: Sequence[SubtaskSpec]) -> None:
        for subtask in subtasks:
            self._subtasks[subtask.subtask_id] = subtask
            option_id = f"option:{subtask.subtask_id}"
            self._options[option_id] = _MinimalOption(
                OptionDescriptor(
                    option_id=option_id,
                    name=f"Option for {subtask.subtask_id}",
                    subtask_id=subtask.subtask_id,
                )
            )

    def integrate_options(self) -> None:
        pass  # options already registered in ingest_subtasks

    def select_action(
        self,
        subjective_state: MinimalSubjectiveState,
        option_stop_threshold: float,
    ) -> tuple[Action, OptionId | None]:
        # Check if active option should continue
        if self._active_option is not None:
            stop_prob = self._active_option.stop_probability(subjective_state)
            if stop_prob < option_stop_threshold:
                return (
                    self._active_option.act(subjective_state),
                    self._active_option.descriptor.option_id,
                )
            self._active_option = None

        # Even observation → primitive action 0
        if subjective_state.observation % 2 == 0:
            return (0, None)

        # Odd observation → first available option, or primitive action 1
        options = list(self._options.values())
        if options:
            self._active_option = options[0]
            return (
                self._active_option.act(subjective_state),
                self._active_option.descriptor.option_id,
            )
        return (1, None)

    def clear_active_option(self) -> None:
        self._active_option = None

    def remove_options(self, option_ids: Sequence[OptionId]) -> None:
        for oid in option_ids:
            self._options.pop(oid, None)
        if (
            self._active_option is not None
            and self._active_option.descriptor.option_id in option_ids
        ):
            self._active_option = None

    def remove_subtasks(self, subtask_ids: Sequence[SubtaskId]) -> None:
        for sid in subtask_ids:
            self._subtasks.pop(sid, None)
            self._options.pop(f"option:{sid}", None)


# ─────────────────────────────────────────────────────────────────────
# Wiring
# ─────────────────────────────────────────────────────────────────────


def build_minimal_agent() -> (
    OaKAgent[Observation, Action, MinimalSubjectiveState, MinimalInfo]
):
    """Construct a fully wired smoke-test OaK agent."""
    return OaKAgent(
        perception=MinimalPerception(),
        transition_model=MinimalTransitionModel(),
        value_function=MinimalValueFunction(),
        reactive_policy=MinimalReactivePolicy(),
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


def run_minimal_training(
    num_episodes: int = 3,
    *,
    horizon: int = 5,
    average_window: int = 100,
    solved_threshold: float | None = None,
) -> list[float]:
    """Train the minimal smoke agent for a few episodes and return rewards."""
    world = MinimalWorld(horizon=horizon)
    agent = build_minimal_agent()
    try:
        return agent.train(
            world,
            num_episodes=num_episodes,
            average_window=average_window,
            solved_threshold=solved_threshold,
        )
    finally:
        world.close()
