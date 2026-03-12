"""Public package surface for the OaK Architecture project.

The package is organized around four pieces:

- `oak_architecture.types`
  Shared data objects such as `TimeStep`, `Transition`, `PlanningUpdate`, and
  `AgentStepResult`.
- `oak_architecture.interfaces`
  Abstract contracts for the main OaK components and the supporting mechanisms
  around them.
- `oak_architecture.agent`
  The reference `OaKAgent` step loop that wires the interfaces together.
- `oak_architecture.implementations`
  Small runnable examples that demonstrate how the interfaces can be
  instantiated.

Recommended reading order:

1. Read the package overview in `README.md`.
2. Read `oak_architecture.interfaces` to understand the expected components.
3. Read `oak_architecture.agent` to see how those components interact at
   runtime.
4. Read `oak_architecture.implementations.minimal_oak` for a concrete but
   intentionally tiny example.
"""

from .agent import OaKAgent
from . import implementations
from .interfaces import (
    Curator,
    FeatureBank,
    FeatureConstructor,
    FeatureRanker,
    GVFLearner,
    MetaStepSizeLearner,
    Option,
    OptionKeyboard,
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
from .types import (
    AgentStepResult,
    ComponentKind,
    CurationDecision,
    FeatureCandidate,
    FeatureSpec,
    GVFSpec,
    ModelPrediction,
    OptionDescriptor,
    PlanningUpdate,
    PolicyDecision,
    SubtaskSpec,
    TimeStep,
    Transition,
    UsageRecord,
    UtilityRecord,
)

__all__ = [
    "AgentStepResult",
    "ComponentKind",
    "CurationDecision",
    "Curator",
    "FeatureBank",
    "FeatureCandidate",
    "FeatureConstructor",
    "FeatureRanker",
    "FeatureSpec",
    "GVFLearner",
    "GVFSpec",
    "implementations",
    "MetaStepSizeLearner",
    "ModelPrediction",
    "OaKAgent",
    "Option",
    "OptionDescriptor",
    "OptionKeyboard",
    "OptionLearner",
    "OptionLibrary",
    "OptionModel",
    "OptionModelLearner",
    "Perception",
    "Planner",
    "PlanningUpdate",
    "PolicyDecision",
    "ReactivePolicy",
    "SubtaskGenerator",
    "SubtaskSpec",
    "TimeStep",
    "Transition",
    "TransitionModel",
    "UsageRecord",
    "UtilityAssessor",
    "UtilityRecord",
    "ValueFunction",
    "World",
]
