from __future__ import annotations

"""Abstract interfaces for the OaK architecture.

How to read this module:

1. Start with `World`, `Perception`, `ReactivePolicy`, `ValueFunction`, and
   `TransitionModel`. These are the main runtime pieces.
2. Then read the feature, subtask, option, and planning interfaces. These add
   representational growth and temporal abstraction.
3. Finish with `UtilityAssessor`, `Curator`, and `MetaStepSizeLearner`. These
   capture the self-maintenance and adaptation machinery around the core agent.

The interfaces are intentionally split so a project can begin with a small
continual-learning agent and only add planning, options, or curation once the
core loop is working.
"""

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
    ActionT,
    ComponentId,
    CurationDecision,
    FeatureCandidate,
    FeatureId,
    FeatureSpec,
    GeneralValueFunctionId,
    GeneralValueFunctionSpec,
    InfoT,
    ModelPrediction,
    ObservationT,
    OptionDescriptor,
    OptionId,
    PlanningUpdate,
    PolicyDecision,
    SubjectiveStateT,
    SubtaskId,
    SubtaskSpec,
    TimeStep,
    Transition,
    UsageRecord,
    UtilityRecord,
)


@runtime_checkable
class World(Protocol[ObservationT, ActionT, InfoT]):
    """Minimal environment protocol used by the agent.

    A `World` may wrap a simulator, a benchmark environment, or a custom
    continual data source. The protocol is intentionally small so the package
    does not depend on a specific environment library.
    """

    def reset(self) -> TimeStep[ObservationT, InfoT]: ...

    def step(self, action: ActionT) -> TimeStep[ObservationT, InfoT]: ...


class Perception(ABC, Generic[ObservationT, ActionT, SubjectiveStateT]):
    """Builds and updates the subjective state seen by the other OaK blocks.

    This is where an implementation decides what `subjective_state` means. For a simple
    domain it may be a hand-built summary; for a more ambitious project it may
    be the output of a learned encoder or recurrent memory system.
    """

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        observation: ObservationT,
        reward: float,
        last_action: Optional[ActionT],
    ) -> SubjectiveStateT:
        raise NotImplementedError

    @abstractmethod
    def current_subjective_state(self) -> SubjectiveStateT:
        raise NotImplementedError


class FeatureBank(ABC, Generic[SubjectiveStateT]):
    """Stores currently active features and their activations."""

    @abstractmethod
    def list_features(self) -> Sequence[FeatureSpec]:
        raise NotImplementedError

    @abstractmethod
    def activations(
        self,
        subjective_state: SubjectiveStateT,
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


class FeatureConstructor(ABC, Generic[SubjectiveStateT]):
    """Proposes new candidate features."""

    @abstractmethod
    def propose(
        self,
        subjective_state: SubjectiveStateT,
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


class SubtaskGenerator(ABC, Generic[SubjectiveStateT]):
    """Maps ranked features to subtasks."""

    @abstractmethod
    def generate(
        self,
        ranked_feature_ids: Sequence[FeatureId],
        feature_bank: FeatureBank[SubjectiveStateT],
    ) -> Sequence[SubtaskSpec]:
        raise NotImplementedError


class GeneralValueFunctionLearner(ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Learns one GeneralValueFunction online."""

    @property
    @abstractmethod
    def spec(self) -> GeneralValueFunctionSpec:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        subjective_state: SubjectiveStateT,
        action: Optional[ActionT] = None,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, transition: Transition[Any, ActionT, SubjectiveStateT, InfoT]
    ) -> float:
        raise NotImplementedError


class ValueFunction(ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Owns the main and auxiliary value learners.

    A minimal implementation can expose a single predictive learner. A richer
    implementation can maintain a bank of GeneralValueFunctions or related predictive signals.
    """

    @abstractmethod
    def list_general_value_functions(
        self,
    ) -> Sequence[GeneralValueFunctionLearner[SubjectiveStateT, ActionT, InfoT]]:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, subjective_state: SubjectiveStateT
    ) -> Mapping[GeneralValueFunctionId, float]:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, transition: Transition[Any, ActionT, SubjectiveStateT, InfoT]
    ) -> Mapping[GeneralValueFunctionId, float]:
        raise NotImplementedError

    @abstractmethod
    def add_or_replace(
        self, learner: GeneralValueFunctionLearner[SubjectiveStateT, ActionT, InfoT]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(
        self, general_value_function_ids: Sequence[GeneralValueFunctionId]
    ) -> None:
        raise NotImplementedError


class Option(ABC, Generic[SubjectiveStateT, ActionT]):
    """Temporal abstraction consisting of policy and termination."""

    @property
    @abstractmethod
    def descriptor(self) -> OptionDescriptor:
        raise NotImplementedError

    @abstractmethod
    def is_available(self, subjective_state: SubjectiveStateT) -> bool:
        raise NotImplementedError

    @abstractmethod
    def act(self, subjective_state: SubjectiveStateT) -> ActionT:
        raise NotImplementedError

    @abstractmethod
    def stop_probability(self, subjective_state: SubjectiveStateT) -> float:
        raise NotImplementedError


class OptionLibrary(ABC, Generic[SubjectiveStateT, ActionT]):
    """Stores learned options."""

    @abstractmethod
    def list_options(self) -> Sequence[Option[SubjectiveStateT, ActionT]]:
        raise NotImplementedError

    @abstractmethod
    def get(self, option_id: OptionId) -> Option[SubjectiveStateT, ActionT]:
        raise NotImplementedError

    @abstractmethod
    def add_or_replace(self, option: Option[SubjectiveStateT, ActionT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, option_ids: Sequence[OptionId]) -> None:
        raise NotImplementedError


class OptionLearner(ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Learns options from subtasks and experience."""

    @abstractmethod
    def ingest_subtasks(self, subtasks: Sequence[SubtaskSpec]) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, transition: Transition[Any, ActionT, SubjectiveStateT, InfoT]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_options(self) -> Sequence[Option[SubjectiveStateT, ActionT]]:
        raise NotImplementedError

    @abstractmethod
    def remove_subtasks(self, subtask_ids: Sequence[SubtaskId]) -> None:
        raise NotImplementedError


class OptionModel(ABC, Generic[SubjectiveStateT]):
    """Predictive model for one option."""

    @property
    @abstractmethod
    def option_id(self) -> OptionId:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        subjective_state: SubjectiveStateT,
    ) -> ModelPrediction[SubjectiveStateT]:
        raise NotImplementedError


class OptionModelLearner(ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Learns option models from experience."""

    @abstractmethod
    def update(
        self, transition: Transition[Any, ActionT, SubjectiveStateT, InfoT]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_models(self) -> Sequence[OptionModel[SubjectiveStateT]]:
        raise NotImplementedError


class TransitionModel(ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Predictive world model for actions and options.

    This interface is the planner-facing model of what will happen next. It may
    be learned, analytic, approximate, or hybrid, as long as it can answer the
    bounded queries the planner needs.
    """

    @abstractmethod
    def update(
        self, transition: Transition[Any, ActionT, SubjectiveStateT, InfoT]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_action(
        self,
        subjective_state: SubjectiveStateT,
        action: ActionT,
    ) -> ModelPrediction[SubjectiveStateT]:
        raise NotImplementedError

    @abstractmethod
    def predict_option(
        self,
        subjective_state: SubjectiveStateT,
        option_id: OptionId,
    ) -> ModelPrediction[SubjectiveStateT]:
        raise NotImplementedError

    @abstractmethod
    def add_or_replace_option_models(
        self, models: Sequence[OptionModel[SubjectiveStateT]]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_option_models(self, option_ids: Sequence[OptionId]) -> None:
        raise NotImplementedError


class Planner(ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Produces planning updates from the transition model.

    The planner does not directly act in the world. Instead it returns
    improvement signals, targets, or search statistics that the reactive policy
    and value learners can use.
    """

    @abstractmethod
    def plan_step(
        self,
        subjective_state: SubjectiveStateT,
        model: TransitionModel[SubjectiveStateT, ActionT, InfoT],
        value_function: ValueFunction[SubjectiveStateT, ActionT, InfoT],
        budget: int,
    ) -> PlanningUpdate[ActionT]:
        raise NotImplementedError


class ReactivePolicy(ABC, Generic[SubjectiveStateT, ActionT]):
    """Chooses primitive actions or options from the current subjective state.

    This is the foreground action-selection mechanism. It may be as small as a
    hand-written policy for a toy domain or as complex as a learned policy head
    over a rich subjective state representation.
    """

    @abstractmethod
    def decide(
        self,
        subjective_state: SubjectiveStateT,
        active_option: Optional[Option[SubjectiveStateT, ActionT]],
        available_options: Sequence[Option[SubjectiveStateT, ActionT]],
    ) -> PolicyDecision[ActionT]:
        raise NotImplementedError

    @abstractmethod
    def update_from_values(
        self,
        subjective_state: SubjectiveStateT,
        td_errors: Mapping[GeneralValueFunctionId, float],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_planning_update(self, update: PlanningUpdate[ActionT]) -> None:
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
    """Aggregates usage signals into utility estimates.

    This is the accounting layer that estimates whether learned structures are
    worth retaining. Feature generators, option learners, and model builders can
    use these scores to decide what to keep improving.
    """

    @abstractmethod
    def observe(self, usage: Sequence[UsageRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def scores(self) -> Sequence[UtilityRecord]:
        raise NotImplementedError


class Curator(ABC):
    """Prunes low-utility architectural elements.

    The curator turns utility estimates into concrete keep/drop decisions. A
    conservative curator can return empty decisions; a more aggressive one can
    actively delete obsolete features, options, models, or predictions.
    """

    @abstractmethod
    def curate(self, utilities: Sequence[UtilityRecord]) -> CurationDecision:
        raise NotImplementedError
