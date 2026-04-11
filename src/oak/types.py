from __future__ import annotations

"""Shared data structures used by the OaK interface package.

This module defines the vocabulary passed between the main OaK components.

The most important objects are:

- `TimeStep`
  One emission from the world or environment.
- `Transition`
  The agent-centric view of a before/after update, used by learning
  components.
- `PlanningUpdate`
  The information returned by planning and consumed by the reactive policy.
- `AgentStepResult`
  The externally visible result of one `OaKAgent.step(...)` call.

In practice, most projects start by making `SubjectiveStateT` concrete, then choosing a
small `ObservationT`, `ActionT`, and `InfoT` that match their environment wrapper.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Generic,
    Mapping,
    Sequence,
    TypeAlias,
    TypeVar,
)

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
# SubjectiveStateT is the agent's learned internal subjective state summary.
SubjectiveStateT = TypeVar("SubjectiveStateT")
InfoT = TypeVar("InfoT")

StructuredValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | Sequence["StructuredValue"]
    | Mapping[str, "StructuredValue"]
)
OpenPayload: TypeAlias = Mapping[str, object]
StructuredPayload: TypeAlias = Mapping[str, StructuredValue]

FeatureId: TypeAlias = str
SubtaskId: TypeAlias = str
OptionId: TypeAlias = str
GeneralValueFunctionId: TypeAlias = str
ComponentId: TypeAlias = str


class ComponentKind(str, Enum):
    """Kinds of learnable or managed elements in the architecture."""

    FEATURE = "feature"
    SUBTASK = "subtask"
    OPTION = "option"
    VALUE_FUNCTION = "value_function"
    OPTION_MODEL = "option_model"
    TRANSITION_MODEL = "transition_model"
    POLICY = "policy"
    PERCEPTION = "perception"
    PLANNER = "planner"


@dataclass(slots=True, frozen=True)
class TimeStep(Generic[ObservationT, InfoT]):
    """One environment emission seen by the agent.

    `TimeStep` is the object passed into `OaKAgent.step(...)`. It contains the
    raw observation, scalar reward, episode-control flags, and optional
    environment metadata.
    """

    observation: ObservationT
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: InfoT | None = None


@dataclass(slots=True, frozen=True)
class Transition(Generic[ActionT, SubjectiveStateT, InfoT]):
    """One subjective-state transition in agent terms.

    `Transition` is constructed by the agent after two consecutive time steps.
    Learners use it instead of the raw world stream so they can access both the
    previous and next subjective state representations together with reward,
    termination, and optional environment metadata.
    """

    subjective_state: SubjectiveStateT
    action: ActionT
    reward: float
    next_subjective_state: SubjectiveStateT
    terminated: bool = False
    option_id: OptionId | None = None
    info: InfoT | None = None


ScalarSignal: TypeAlias = Callable[
    [Transition[ActionT, SubjectiveStateT, InfoT]], float
]
ContinuationFn: TypeAlias = Callable[
    [Transition[ActionT, SubjectiveStateT, InfoT]], float
]
TerminationValueFn: TypeAlias = Callable[
    [Transition[ActionT, SubjectiveStateT, InfoT]], float
]


@dataclass(slots=True, frozen=True)
class FeatureSpec:
    """Metadata describing a feature tracked by the agent."""

    feature_id: FeatureId
    name: str
    description: str = ""
    metadata: OpenPayload = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FeatureCandidate:
    """A proposed feature that may be admitted into the feature bank."""

    feature_id: FeatureId
    name: str
    origin: str
    description: str = ""
    metadata: OpenPayload = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class GeneralValueFunctionSpec(Generic[ActionT, SubjectiveStateT, InfoT]):
    """General value function specification."""

    general_value_function_id: GeneralValueFunctionId
    name: str
    cumulant: ScalarSignal
    continuation: ContinuationFn
    termination_value: TerminationValueFn
    metadata: OpenPayload = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SubtaskSpec:
    """A feature-grounded subtask description."""

    subtask_id: SubtaskId
    name: str
    feature_id: FeatureId
    intensity: float = 1.0
    general_value_function_id: GeneralValueFunctionId | None = None
    metadata: OpenPayload = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OptionDescriptor:
    """Lightweight metadata for an option."""

    option_id: OptionId
    name: str
    subtask_id: SubtaskId | None = None
    metadata: OpenPayload = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PolicyDecision(Generic[ActionT]):
    """Return type for reactive policy selection."""

    action: ActionT | None = None
    option_id: OptionId | None = None
    metadata: OpenPayload = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_action = self.action is not None
        has_option = self.option_id is not None
        if has_action == has_option:
            raise ValueError(
                "PolicyDecision requires exactly one of action or option_id."
            )


@dataclass(slots=True, frozen=True)
class ModelPrediction(Generic[SubjectiveStateT]):
    """Prediction returned by an action or option model."""

    predicted_subjective_state: SubjectiveStateT
    cumulative_reward: float
    steps: int | None = None
    terminated: bool = False
    metadata: OpenPayload = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PlanningUpdate(Generic[ActionT]):
    """Outputs from one planning pass."""

    value_targets: Mapping[GeneralValueFunctionId, float] = field(default_factory=dict)
    policy_targets: StructuredPayload = field(default_factory=dict)
    search_statistics: StructuredPayload = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UsageRecord:
    """Usage evidence gathered for utility assessment."""

    kind: ComponentKind
    component_id: ComponentId
    amount: float = 1.0
    metadata: OpenPayload = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UtilityRecord:
    """Utility score for one architectural element."""

    kind: ComponentKind
    component_id: ComponentId
    utility: float
    evidence: StructuredPayload = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CurationDecision:
    """Pruning decision returned by the curator."""

    drop_features: Sequence[FeatureId] = field(default_factory=tuple)
    drop_subtasks: Sequence[SubtaskId] = field(default_factory=tuple)
    drop_options: Sequence[OptionId] = field(default_factory=tuple)
    drop_option_models: Sequence[OptionId] = field(default_factory=tuple)
    drop_general_value_functions: Sequence[GeneralValueFunctionId] = field(
        default_factory=tuple
    )
    notes: StructuredPayload = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentStepResult(Generic[ActionT, SubjectiveStateT]):
    """Observable result of one OaK agent step.

    This is the compact object a caller receives after stepping the agent. It
    includes the primitive action actually executed, the current subjective
    state, and any structures or planning signals created during that step.
    """

    action: ActionT
    subjective_state: SubjectiveStateT
    active_option_id: OptionId | None = None
    planning_update: PlanningUpdate[ActionT] | None = None
    created_subtasks: Sequence[SubtaskSpec] = field(default_factory=tuple)
    curation_decision: CurationDecision | None = None


@dataclass(slots=True, frozen=True)
class EpisodeStepRecord(Generic[ObservationT, ActionT, SubjectiveStateT, InfoT]):
    """One executed environment transition collected during training."""

    step_index: int
    time_step: TimeStep[ObservationT, InfoT]
    action: ActionT
    next_time_step: TimeStep[ObservationT, InfoT]
    active_option_id: OptionId | None = None
    planning_update: PlanningUpdate[ActionT] | None = None
    created_subtasks: Sequence[SubtaskSpec] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class EpisodeTrace(Generic[ObservationT, ActionT, SubjectiveStateT, InfoT]):
    """Rich episode artifact exposed to optional training-side callbacks."""

    episode: int
    episode_reward: float
    avg_reward: float
    step_count: int
    solved: bool
    initial_time_step: TimeStep[ObservationT, InfoT]
    final_time_step: TimeStep[ObservationT, InfoT]
    steps: Sequence[EpisodeStepRecord[ObservationT, ActionT, SubjectiveStateT, InfoT]] = field(
        default_factory=tuple
    )
    frames: Sequence[object] = field(default_factory=tuple)
    world: object | None = None
    agent: object | None = None
    metadata: OpenPayload = field(default_factory=dict)
