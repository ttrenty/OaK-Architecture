from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, Mapping, Optional, Sequence, TypeAlias, TypeVar

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
StateT = TypeVar("StateT")
InfoT = TypeVar("InfoT", bound=Mapping[str, Any])

FeatureId: TypeAlias = str
SubtaskId: TypeAlias = str
OptionId: TypeAlias = str
GVFId: TypeAlias = str
ComponentId: TypeAlias = str


class ComponentKind(str, Enum):
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
    observation: ObsT
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: Optional[InfoT] = None


@dataclass(slots=True, frozen=True)
class Transition(Generic[ObsT, ActT, StateT]):
    state: StateT
    action: ActT
    reward: float
    next_state: StateT
    observation: Optional[ObsT] = None
    next_observation: Optional[ObsT] = None
    terminated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


ScalarSignal: TypeAlias = Callable[[Transition[Any, Any, Any]], float]
ContinuationFn: TypeAlias = Callable[[Transition[Any, Any, Any]], float]
TerminationValueFn: TypeAlias = Callable[[Transition[Any, Any, Any]], float]


@dataclass(slots=True, frozen=True)
class FeatureSpec:
    feature_id: FeatureId
    name: str
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FeatureCandidate:
    feature_id: FeatureId
    name: str
    origin: str
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class GVFSpec:
    gvf_id: GVFId
    name: str
    cumulant: ScalarSignal
    continuation: ContinuationFn
    termination_value: TerminationValueFn
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SubtaskSpec:
    subtask_id: SubtaskId
    name: str
    feature_id: FeatureId
    intensity: float = 1.0
    gvf_id: Optional[ ] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OptionDescriptor:
    option_id: OptionId
    name: str
    subtask_id: Optional[SubtaskId] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PolicyDecision(Generic[ActT]):
    action: Optional[ActT] = None
    option_id: Optional[OptionId] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_action = self.action is not None
        has_option = self.option_id is not None
        if has_action == has_option:
            raise ValueError("PolicyDecision requires exactly one of action or option_id.")


@dataclass(slots=True, frozen=True)
class ModelPrediction(Generic[StateT]):
    next_state: StateT
    cumulative_reward: float
    steps: Optional[int] = None
    terminated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PlanningUpdate(Generic[ActT]):
    value_targets: Mapping[GVFId, float] = field(default_factory=dict)
    policy_targets: Mapping[str, Any] = field(default_factory=dict)
    search_statistics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UsageRecord:
    kind: ComponentKind
    component_id: ComponentId
    amount: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UtilityRecord:
    kind: ComponentKind
    component_id: ComponentId
    utility: float
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CurationDecision:
    drop_features: Sequence[FeatureId] = field(default_factory=tuple)
    drop_subtasks: Sequence[SubtaskId] = field(default_factory=tuple)
    drop_options: Sequence[OptionId] = field(default_factory=tuple)
    drop_option_models: Sequence[OptionId] = field(default_factory=tuple)
    drop_gvfs: Sequence[GVFId] = field(default_factory=tuple)
    notes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentStepResult(Generic[ActT, StateT]):
    action: ActT
    state: StateT
    active_option_id: Optional[OptionId] = None
    planning_update: Optional[PlanningUpdate[ActT]] = None
    created_subtasks: Sequence[SubtaskSpec] = field(default_factory=tuple)
    curation_decision: Optional[CurationDecision] = None
