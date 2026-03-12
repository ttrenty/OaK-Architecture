from __future__ import annotations

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
    AgentStepResult,
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
    def reset(self) -> TimeStep[ObsT, InfoT]: ...

    def step(self, action: ActT) -> TimeStep[ObsT, InfoT]: ...


class Perception(ABC, Generic[ObsT, ActT, StateT]):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, observation: ObsT, reward: float, last_action: Optional[ActT]
    ) -> StateT:
        raise NotImplementedError

    @abstractmethod
    def current_state(self) -> StateT:
        raise NotImplementedError


class FeatureBank(ABC, Generic[StateT]):
    @abstractmethod
    def list_features(self) -> Sequence[FeatureSpec]:
        raise NotImplementedError

    @abstractmethod
    def activations(self, state: StateT) -> Mapping[FeatureId, float]:
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
    @abstractmethod
    def propose(
        self,
        state: StateT,
        active_features: Sequence[FeatureSpec],
    ) -> Sequence[FeatureCandidate]:
        raise NotImplementedError


class FeatureRanker(ABC):
    @abstractmethod
    def rank(
        self,
        features: Sequence[FeatureSpec],
        utilities: Sequence[UtilityRecord],
        limit: Optional[int] = None,
    ) -> Sequence[FeatureId]:
        raise NotImplementedError


class SubtaskGenerator(ABC, Generic[StateT]):
    @abstractmethod
    def generate(
        self,
        ranked_feature_ids: Sequence[FeatureId],
        feature_bank: FeatureBank[StateT],
    ) -> Sequence[SubtaskSpec]:
        raise NotImplementedError


class GVFLearner(ABC, Generic[StateT, ActT]):
    @property
    @abstractmethod
    def spec(self) -> GVFSpec:
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: StateT, action: Optional[ActT] = None) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, transition: Transition[Any, ActT, StateT]) -> float:
        raise NotImplementedError


class ValueFunctionBank(ABC, Generic[StateT, ActT]):
    @abstractmethod
    def list_gvfs(self) -> Sequence[GVFLearner[StateT, ActT]]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: StateT) -> Mapping[GVFId, float]:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, transition: Transition[Any, ActT, StateT]
    ) -> Mapping[GVFId, float]:
        raise NotImplementedError

    @abstractmethod
    def add_or_replace(self, learner: GVFLearner[StateT, ActT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, gvf_ids: Sequence[GVFId]) -> None:
        raise NotImplementedError


class Option(ABC, Generic[StateT, ActT]):
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


class OptionLearner(ABC, Generic[StateT, ActT]):
    @abstractmethod
    def ingest_subtasks(self, subtasks: Sequence[SubtaskSpec]) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, transition: Transition[Any, ActT, StateT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_options(self) -> Sequence[Option[StateT, ActT]]:
        raise NotImplementedError

    @abstractmethod
    def remove_subtasks(self, subtask_ids: Sequence[SubtaskId]) -> None:
        raise NotImplementedError


class OptionModel(ABC, Generic[StateT]):
    @property
    @abstractmethod
    def option_id(self) -> OptionId:
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: StateT) -> ModelPrediction[StateT]:
        raise NotImplementedError


class OptionModelLearner(ABC, Generic[StateT, ActT]):
    @abstractmethod
    def update(self, transition: Transition[Any, ActT, StateT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_models(self) -> Sequence[OptionModel[StateT]]:
        raise NotImplementedError


class TransitionModel(ABC, Generic[StateT, ActT]):
    @abstractmethod
    def update(self, transition: Transition[Any, ActT, StateT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_action(self, state: StateT, action: ActT) -> ModelPrediction[StateT]:
        raise NotImplementedError

    @abstractmethod
    def predict_option(
        self, state: StateT, option_id: OptionId
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


class Planner(ABC, Generic[StateT, ActT]):
    @abstractmethod
    def plan_step(
        self,
        state: StateT,
        model: TransitionModel[StateT, ActT],
        values: ValueFunctionBank[StateT, ActT],
        budget: int,
    ) -> PlanningUpdate[ActT]:
        raise NotImplementedError


class ReactivePolicy(ABC, Generic[StateT, ActT]):
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
        self, state: StateT, td_errors: Mapping[GVFId, float]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_planning_update(self, update: PlanningUpdate[ActT]) -> None:
        raise NotImplementedError


class OptionKeyboard(ABC):
    @abstractmethod
    def compose(self, intensities: Sequence[float]) -> OptionDescriptor:
        raise NotImplementedError


class MetaStepSizeLearner(ABC):
    @abstractmethod
    def update(
        self, component_id: ComponentId, error_signals: Mapping[str, float]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def step_sizes(self, component_id: ComponentId) -> Mapping[str, float]:
        raise NotImplementedError


class UtilityAssessor(ABC):
    @abstractmethod
    def observe(self, usage: Sequence[UsageRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def scores(self) -> Sequence[UtilityRecord]:
        raise NotImplementedError


class Curator(ABC):
    @abstractmethod
    def curate(self, utilities: Sequence[UtilityRecord]) -> CurationDecision:
        raise NotImplementedError
