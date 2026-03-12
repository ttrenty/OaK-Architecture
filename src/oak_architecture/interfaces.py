from __future__ import annotations
"""Abstract interfaces for the OaK architecture."""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .types import (
    ActT,
    ComponentId,
    CurationDecision,
    FeatureCandidate,
    FeatureId,
    FeatureSpec,
    GVFId,
    GVFSpec,
    InfoT,
    ModelPrediction,
    ObsT,
    OptionDescriptor,
    OptionId,
    PlanningUpdate,
    PolicyDecision,
    StateT,
    SubtaskId,
    SubtaskSpec,
    TimeStep,
    Transition,
    UsageRecord,
    UtilityRecord,
)


@runtime_checkable
class World(Protocol[ObsT, ActT, InfoT]):
    """Minimal environment protocol used by the agent."""

    def reset(self) -> TimeStep[ObsT, InfoT]: ...

    def step(self, action: ActT) -> TimeStep[ObsT, InfoT]: ...


class Perception(ABC, Generic[ObsT, ActT, StateT]):
    """Builds and updates the state seen by the other AoK blocks."""

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        observation: ObsT,
        reward: float,
        last_action: Optional[ActT],
    ) -> StateT:
        raise NotImplementedError

    @abstractmethod
    def current_state(self) -> StateT:
        raise NotImplementedError


class FeatureBank(ABC, Generic[StateT]):
    """Stores currently active features and their activations."""

    @abstractmethod
    def list_features(self) -> Sequence[FeatureSpec]:
        raise NotImplementedError

    @abstractmethod
    def activations(
        self,
        state: StateT,
    ) -> Mapping[FeatureId, float]:
        raise NotImplementedError

    @abstractmethod
    def add_candidates(
        self, candidates: Sequence[FeatureCandidate]
    ) -> Sequence[FeatureSpec]:
        raise NotImplementedError

    @abstractmethod
    def remove(self, feature_ids: Sequence[FeatureId]) -> None:
        raise NotImplementedError


class FeatureConstructor(ABC, Generic[StateT]):
    """Proposes new candidate features."""

    @abstractmethod
    def propose(
        self,
        state: StateT,
        active_features: Sequence[FeatureSpec],
    ) -> Sequence[FeatureCandidate]:
        raise NotImplementedError


class FeatureRanker(ABC):
    """Ranks features for downstream use."""

    @abstractmethod
    def rank(
        self,
        features: Sequence[FeatureSpec],
        utilities: Sequence[UtilityRecord],
        limit: Optional[int] = None,
    ) -> Sequence[FeatureId]:
        raise NotImplementedError


class SubtaskGenerator(ABC, Generic[StateT]):
    """Maps ranked features to subtasks."""

    @abstractmethod
    def generate(
        self,
        ranked_feature_ids: Sequence[FeatureId],
        feature_bank: FeatureBank[StateT],
    ) -> Sequence[SubtaskSpec]:
        raise NotImplementedError


class GVFLearner(ABC, Generic[StateT, ActT, InfoT]):
    """Learns one GVF online."""

    @property
    @abstractmethod
    def spec(self) -> GVFSpec:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        state: StateT,
        action: Optional[ActT] = None,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, transition: Transition[Any, ActT, StateT, InfoT]) -> float:
        raise NotImplementedError


class ValueFunction(ABC, Generic[StateT, ActT, InfoT]):
    """Owns the main and auxiliary value learners."""

    @abstractmethod
    def list_gvfs(self) -> Sequence[GVFLearner[StateT, ActT, InfoT]]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: StateT) -> Mapping[GVFId, float]:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, transition: Transition[Any, ActT, StateT, InfoT]
    ) -> Mapping[GVFId, float]:
        raise NotImplementedError

    @abstractmethod
    def add_or_replace(self, learner: GVFLearner[StateT, ActT, InfoT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, gvf_ids: Sequence[GVFId]) -> None:
        raise NotImplementedError


class Option(ABC, Generic[StateT, ActT]):
    """Temporal abstraction consisting of policy and termination."""

    @property
    @abstractmethod
    def descriptor(self) -> OptionDescriptor:
        raise NotImplementedError

    @abstractmethod
    def is_available(self, state: StateT) -> bool:
        raise NotImplementedError

    @abstractmethod
    def act(self, state: StateT) -> ActT:
        raise NotImplementedError

    @abstractmethod
    def stop_probability(self, state: StateT) -> float:
        raise NotImplementedError


class OptionLibrary(ABC, Generic[StateT, ActT]):
    """Stores learned options."""

    @abstractmethod
    def list_options(self) -> Sequence[Option[StateT, ActT]]:
        raise NotImplementedError

    @abstractmethod
    def get(self, option_id: OptionId) -> Option[StateT, ActT]:
        raise NotImplementedError

    @abstractmethod
    def add_or_replace(self, option: Option[StateT, ActT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, option_ids: Sequence[OptionId]) -> None:
        raise NotImplementedError


class OptionLearner(ABC, Generic[StateT, ActT, InfoT]):
    """Learns options from subtasks and experience."""

    @abstractmethod
    def ingest_subtasks(self, subtasks: Sequence[SubtaskSpec]) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, transition: Transition[Any, ActT, StateT, InfoT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_options(self) -> Sequence[Option[StateT, ActT]]:
        raise NotImplementedError

    @abstractmethod
    def remove_subtasks(self, subtask_ids: Sequence[SubtaskId]) -> None:
        raise NotImplementedError


class OptionModel(ABC, Generic[StateT]):
    """Predictive model for one option."""

    @property
    @abstractmethod
    def option_id(self) -> OptionId:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        state: StateT,
    ) -> ModelPrediction[StateT]:
        raise NotImplementedError


class OptionModelLearner(ABC, Generic[StateT, ActT, InfoT]):
    """Learns option models from experience."""

    @abstractmethod
    def update(self, transition: Transition[Any, ActT, StateT, InfoT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_models(self) -> Sequence[OptionModel[StateT]]:
        raise NotImplementedError


class TransitionModel(ABC, Generic[StateT, ActT, InfoT]):
    """Predictive world model for actions and options."""

    @abstractmethod
    def update(self, transition: Transition[Any, ActT, StateT, InfoT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_action(
        self,
        state: StateT,
        action: ActT,
    ) -> ModelPrediction[StateT]:
        raise NotImplementedError

    @abstractmethod
    def predict_option(
        self,
        state: StateT,
        option_id: OptionId,
    ) -> ModelPrediction[StateT]:
        raise NotImplementedError

    @abstractmethod
    def add_or_replace_option_models(
        self, models: Sequence[OptionModel[StateT]]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_option_models(self, option_ids: Sequence[OptionId]) -> None:
        raise NotImplementedError


class Planner(ABC, Generic[StateT, ActT, InfoT]):
    """Produces planning updates from the transition model."""

    @abstractmethod
    def plan_step(
        self,
        state: StateT,
        model: TransitionModel[StateT, ActT, InfoT],
        value_function: ValueFunction[StateT, ActT, InfoT],
        budget: int,
    ) -> PlanningUpdate[ActT]:
        raise NotImplementedError


class ReactivePolicy(ABC, Generic[StateT, ActT]):
    """Chooses primitive actions or options from the current state."""

    @abstractmethod
    def decide(
        self,
        state: StateT,
        active_option: Optional[Option[StateT, ActT]],
        available_options: Sequence[Option[StateT, ActT]],
    ) -> PolicyDecision[ActT]:
        raise NotImplementedError

    @abstractmethod
    def update_from_values(
        self,
        state: StateT,
        td_errors: Mapping[GVFId, float],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_planning_update(self, update: PlanningUpdate[ActT]) -> None:
        raise NotImplementedError


class OptionKeyboard(ABC):
    """Optional composition interface for combining options."""

    @abstractmethod
    def compose(self, intensities: Sequence[float]) -> OptionDescriptor:
        raise NotImplementedError


class MetaStepSizeLearner(ABC):
    """Tracks or adapts per-component step-size metadata."""

    @abstractmethod
    def update(
        self, component_id: ComponentId, error_signals: Mapping[str, float]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def step_sizes(self, component_id: ComponentId) -> Mapping[str, float]:
        raise NotImplementedError


class UtilityAssessor(ABC):
    """Aggregates usage signals into utility estimates."""

    @abstractmethod
    def observe(self, usage: Sequence[UsageRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def scores(self) -> Sequence[UtilityRecord]:
        raise NotImplementedError


class Curator(ABC):
    """Prunes low-utility architectural elements."""

    @abstractmethod
    def curate(self, utilities: Sequence[UtilityRecord]) -> CurationDecision:
        raise NotImplementedError
