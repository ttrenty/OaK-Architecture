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
    SubjectiveStateT,
    SubtaskId,
    SubtaskSpec,
    TimeStep,
    Transition,
    UsageRecord,
    UtilityRecord,
)


@runtime_checkable
class World(Protocol[ObsT, ActT, InfoT]):
    """Minimal environment protocol used by the agent.

    A `World` may wrap a simulator, a benchmark environment, or a custom
    continual data source. The protocol is intentionally small so the package
    does not depend on a specific environment library.
    """

    def reset(self) -> TimeStep[ObsT, InfoT]: ...

    def step(self, action: ActT) -> TimeStep[ObsT, InfoT]: ...


class Perception(ABC, Generic[ObsT, ActT, SubjectiveStateT]):
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
        observation: ObsT,
        reward: float,
        last_action: Optional[ActT],
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


class GVFLearner(ABC, Generic[SubjectiveStateT, ActT, InfoT]):
    """Learns one GVF online."""

    @property
    @abstractmethod
    def spec(self) -> GVFSpec:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        subjective_state: SubjectiveStateT,
        action: Optional[ActT] = None,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, transition: Transition[Any, ActT, SubjectiveStateT, InfoT]
    ) -> float:
        raise NotImplementedError


class ValueFunction(ABC, Generic[SubjectiveStateT, ActT, InfoT]):
    """Owns the main and auxiliary value learners.

    A minimal implementation can expose a single predictive learner. A richer
    implementation can maintain a bank of GVFs or related predictive signals.
    """

    @abstractmethod
    def list_gvfs(self) -> Sequence[GVFLearner[SubjectiveStateT, ActT, InfoT]]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, subjective_state: SubjectiveStateT) -> Mapping[GVFId, float]:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, transition: Transition[Any, ActT, SubjectiveStateT, InfoT]
    ) -> Mapping[GVFId, float]:
        raise NotImplementedError

    @abstractmethod
    def add_or_replace(
        self, learner: GVFLearner[SubjectiveStateT, ActT, InfoT]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, gvf_ids: Sequence[GVFId]) -> None:
        raise NotImplementedError


class Option(ABC, Generic[SubjectiveStateT, ActT]):
    """Temporal abstraction consisting of policy and termination."""

    @property
    @abstractmethod
    def descriptor(self) -> OptionDescriptor:
        raise NotImplementedError

    @abstractmethod
    def is_available(self, subjective_state: SubjectiveStateT) -> bool:
        raise NotImplementedError

    @abstractmethod
    def act(self, subjective_state: SubjectiveStateT) -> ActT:
        raise NotImplementedError

    @abstractmethod
    def stop_probability(self, subjective_state: SubjectiveStateT) -> float:
        raise NotImplementedError


class OptionLibrary(ABC, Generic[SubjectiveStateT, ActT]):
    """Stores learned options."""

    @abstractmethod
    def list_options(self) -> Sequence[Option[SubjectiveStateT, ActT]]:
        raise NotImplementedError

    @abstractmethod
    def get(self, option_id: OptionId) -> Option[SubjectiveStateT, ActT]:
        raise NotImplementedError

    @abstractmethod
    def add_or_replace(self, option: Option[SubjectiveStateT, ActT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, option_ids: Sequence[OptionId]) -> None:
        raise NotImplementedError


class OptionLearner(ABC, Generic[SubjectiveStateT, ActT, InfoT]):
    """Learns options from subtasks and experience."""

    @abstractmethod
    def ingest_subtasks(self, subtasks: Sequence[SubtaskSpec]) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, transition: Transition[Any, ActT, SubjectiveStateT, InfoT]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_options(self) -> Sequence[Option[SubjectiveStateT, ActT]]:
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


class OptionModelLearner(ABC, Generic[SubjectiveStateT, ActT, InfoT]):
    """Learns option models from experience."""

    @abstractmethod
    def update(
        self, transition: Transition[Any, ActT, SubjectiveStateT, InfoT]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_models(self) -> Sequence[OptionModel[SubjectiveStateT]]:
        raise NotImplementedError


class TransitionModel(ABC, Generic[SubjectiveStateT, ActT, InfoT]):
    """Predictive world model for actions and options.

    This interface is the planner-facing model of what will happen next. It may
    be learned, analytic, approximate, or hybrid, as long as it can answer the
    bounded queries the planner needs.
    """

    @abstractmethod
    def update(
        self, transition: Transition[Any, ActT, SubjectiveStateT, InfoT]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_action(
        self,
        subjective_state: SubjectiveStateT,
        action: ActT,
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


class Planner(ABC, Generic[SubjectiveStateT, ActT, InfoT]):
    """Produces planning updates from the transition model.

    The planner does not directly act in the world. Instead it returns
    improvement signals, targets, or search statistics that the reactive policy
    and value learners can use.
    """

    @abstractmethod
    def plan_step(
        self,
        subjective_state: SubjectiveStateT,
        model: TransitionModel[SubjectiveStateT, ActT, InfoT],
        value_function: ValueFunction[SubjectiveStateT, ActT, InfoT],
        budget: int,
    ) -> PlanningUpdate[ActT]:
        raise NotImplementedError


class ReactivePolicy(ABC, Generic[SubjectiveStateT, ActT]):
    """Chooses primitive actions or options from the current subjective state.

    This is the foreground action-selection mechanism. It may be as small as a
    hand-written policy for a toy domain or as complex as a learned policy head
    over a rich subjective state representation.
    """

    @abstractmethod
    def decide(
        self,
        subjective_state: SubjectiveStateT,
        active_option: Optional[Option[SubjectiveStateT, ActT]],
        available_options: Sequence[Option[SubjectiveStateT, ActT]],
    ) -> PolicyDecision[ActT]:
        raise NotImplementedError

    @abstractmethod
    def update_from_values(
        self,
        subjective_state: SubjectiveStateT,
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
