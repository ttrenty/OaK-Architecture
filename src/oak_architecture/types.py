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
small `ObsT`, `ActT`, and `InfoT` that match their environment wrapper.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
)

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
# SubjectiveStateT is the agent's learned internal subjective state summary.
SubjectiveStateT = TypeVar("SubjectiveStateT")
InfoT = TypeVar("InfoT", bound=Mapping[str, Any])

FeatureId: TypeAlias = str
SubtaskId: TypeAlias = str
OptionId: TypeAlias = str
GVFId: TypeAlias = str
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
class TimeStep(Generic[ObsT, InfoT]):
    """One environment emission seen by the agent.

    `TimeStep` is the object passed into `OaKAgent.step(...)`. It contains the
    raw observation, scalar reward, episode-control flags, and optional
    environment metadata.
    """

    observation: ObsT
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: Optional[InfoT] = None


@dataclass(slots=True, frozen=True)
class Transition(Generic[ObsT, ActT, SubjectiveStateT, InfoT]):
    """One subjective-state transition in agent terms.

    `Transition` is constructed by the agent after two consecutive time steps.
    Learners use it instead of the raw world stream so they can access both the
    previous and next subjective state representations.
    """

    subjective_state: SubjectiveStateT
    action: ActT
    reward: float
    next_subjective_state: SubjectiveStateT
    observation: Optional[ObsT] = None
    next_observation: Optional[ObsT] = None
    terminated: bool = False
    info: Optional[InfoT] = None


ScalarSignal: TypeAlias = Callable[[Transition[Any, Any, Any, Any]], float]
ContinuationFn: TypeAlias = Callable[[Transition[Any, Any, Any, Any]], float]
TerminationValueFn: TypeAlias = Callable[[Transition[Any, Any, Any, Any]], float]


@dataclass(slots=True, frozen=True)
class FeatureSpec:
    """Metadata describing a feature tracked by the agent."""

    feature_id: FeatureId
    name: str
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FeatureCandidate:
    """A proposed feature that may be admitted into the feature bank."""

    feature_id: FeatureId
    name: str
    origin: str
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class GVFSpec:
    """General value function specification."""

    gvf_id: GVFId
    name: str
    cumulant: ScalarSignal
    continuation: ContinuationFn
    termination_value: TerminationValueFn
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SubtaskSpec:
    """A feature-grounded subtask description."""

    subtask_id: SubtaskId
    name: str
    feature_id: FeatureId
    intensity: float = 1.0
    gvf_id: Optional[GVFId] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OptionDescriptor:
    """Lightweight metadata for an option."""

    option_id: OptionId
    name: str
    subtask_id: Optional[SubtaskId] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PolicyDecision(Generic[ActT]):
    """Return type for reactive policy selection."""

    action: Optional[ActT] = None
    option_id: Optional[OptionId] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

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
    steps: Optional[int] = None
    terminated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PlanningUpdate(Generic[ActT]):
    """Outputs from one planning pass."""

    value_targets: Mapping[GVFId, float] = field(default_factory=dict)
    policy_targets: Mapping[str, Any] = field(default_factory=dict)
    search_statistics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UsageRecord:
    """Usage evidence gathered for utility assessment."""

    kind: ComponentKind
    component_id: ComponentId
    amount: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UtilityRecord:
    """Utility score for one architectural element."""

    kind: ComponentKind
    component_id: ComponentId
    utility: float
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CurationDecision:
    """Pruning decision returned by the curator."""

    drop_features: Sequence[FeatureId] = field(default_factory=tuple)
    drop_subtasks: Sequence[SubtaskId] = field(default_factory=tuple)
    drop_options: Sequence[OptionId] = field(default_factory=tuple)
    drop_option_models: Sequence[OptionId] = field(default_factory=tuple)
    drop_gvfs: Sequence[GVFId] = field(default_factory=tuple)
    notes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentStepResult(Generic[ActT, SubjectiveStateT]):
    """Observable result of one OaK agent step.

    This is the compact object a caller receives after stepping the agent. It
    includes the primitive action actually executed, the current subjective
    state, and any structures or planning signals created during that step.
    """

    action: ActT
    subjective_state: SubjectiveStateT
    active_option_id: Optional[OptionId] = None
    planning_update: Optional[PlanningUpdate[ActT]] = None
    created_subtasks: Sequence[SubtaskSpec] = field(default_factory=tuple)
    curation_decision: Optional[CurationDecision] = None
