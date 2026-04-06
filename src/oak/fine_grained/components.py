from __future__ import annotations

"""Fine-grained component interfaces for the OaK architecture.

These are optional building blocks that can be composed into the four main
OaK interfaces (`Perception`, `TransitionModel`, `ValueFunction`,
`ReactivePolicy`) using the composites in
`oak.fine_grained.composites`.

You do **not** need these to build an OaK agent; implementing the four
main interfaces directly is simpler and sufficient.  These components exist
for projects that need to independently swap one piece inside a module
(e.g. replace the planner without touching the world model).

Mapping to the four main interfaces
------------------------------------

====================  ==================================================
Main interface        Component building blocks
====================  ==================================================
`Perception`        `StateBuilder`, `FeatureBank`,
                      `FeatureConstructor`, `FeatureRanker`,
                      `SubtaskGenerator`,
                      `MetaStepSizeLearner` (optional)
`TransitionModel`   `WorldModel`, `OptionModel`,
                      `OptionModelLearner`, `Planner`,
                      `MetaStepSizeLearner` (optional)
`ValueFunction`     `ValueEstimator`, `GeneralValueFunctionLearner`,
                      `UtilityAssessor`, `Curator`,
                      `MetaStepSizeLearner` (optional)
`ReactivePolicy`    `ActionSelector`, `Option`, `OptionLibrary`,
                      `OptionLearner`, `OptionKeyboard`,
                      `MetaStepSizeLearner` (optional)
====================  ==================================================
"""

from abc import ABC, abstractmethod
from typing import (
    Generic,
    Mapping,
    Sequence,
)

from ..types import (
    ActionT,
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
    SubjectiveStateT,
    SubtaskId,
    SubtaskSpec,
    Transition,
    UsageRecord,
    UtilityRecord,
    PolicyDecision,
)


# ─────────────────────────────────────────────────────────────────────
# Perception components
# ─────────────────────────────────────────────────────────────────────


class StateBuilder(ABC, Generic[ObservationT, ActionT, SubjectiveStateT]):
    """Builds and updates the subjective state seen by every other component.

    This is where an implementation decides what *subjective_state* means.
    For a simple domain it may be a hand-built summary; for a more ambitious
    project it may be the output of a learned encoder or recurrent memory.
    """

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        observation: ObservationT,
        reward: float,
        last_action: ActionT | None,
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
        """Return per-feature activation values for the given state.

        Intended for `SubtaskGenerator` implementations, which receive
        the `FeatureBank` and may use activations to decide which
        features warrant new subtasks.
        """
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
        limit: int | None = None,
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


# ─────────────────────────────────────────────────────────────────────
# Transition Model components
# ─────────────────────────────────────────────────────────────────────


class WorldModel(ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Predictive world model for actions and options.

    This is the planner-facing model of what will happen next.  It may be
    learned, analytic, approximate, or hybrid, as long as it can answer the
    bounded queries the planner needs.
    """

    @abstractmethod
    def update(self, transition: Transition[ActionT, SubjectiveStateT, InfoT]) -> None:
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
    def update(self, transition: Transition[ActionT, SubjectiveStateT, InfoT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_models(self) -> Sequence[OptionModel[SubjectiveStateT]]:
        raise NotImplementedError


class Planner(ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Produces planning updates from the world model.

    The planner does not directly act in the world.  Instead it returns
    improvement signals, targets, or search statistics that the reactive
    policy and value learners can use.
    """

    @abstractmethod
    def plan_step(
        self,
        subjective_state: SubjectiveStateT,
        model: WorldModel[SubjectiveStateT, ActionT, InfoT],
        value_function: ValueEstimator[SubjectiveStateT, ActionT, InfoT],
        budget: int,
    ) -> PlanningUpdate[ActionT]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────
# Value Function components
# ─────────────────────────────────────────────────────────────────────


class GeneralValueFunctionLearner(ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Learns one General Value Function online."""

    @property
    @abstractmethod
    def spec(self) -> GeneralValueFunctionSpec[ActionT, SubjectiveStateT, InfoT]:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        subjective_state: SubjectiveStateT,
        action: ActionT | None = None,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, transition: Transition[ActionT, SubjectiveStateT, InfoT]) -> float:
        raise NotImplementedError


class ValueEstimator(ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Owns the main and auxiliary value learners.

    A minimal implementation can expose a single predictive learner.  A
    richer implementation can maintain a bank of General Value Functions.
    """

    @abstractmethod
    def list_general_value_functions(
        self,
    ) -> Sequence[GeneralValueFunctionLearner[SubjectiveStateT, ActionT, InfoT]]:
        """Return all managed GVF learners.

        Intended for `Planner` implementations that need to inspect
        the GVF bank (e.g., to evaluate auxiliary predictions during
        planning).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, subjective_state: SubjectiveStateT
    ) -> Mapping[GeneralValueFunctionId, float]:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, transition: Transition[ActionT, SubjectiveStateT, InfoT]
    ) -> Mapping[GeneralValueFunctionId, float]:
        raise NotImplementedError

    @abstractmethod
    def add_or_replace(
        self, learner: GeneralValueFunctionLearner[SubjectiveStateT, ActionT, InfoT]
    ) -> None:
        """Add or replace a GVF learner in the bank.

        Used for dynamic GVF management, e.g., creating new GVFs when
        new subtasks or options are discovered.
        """
        raise NotImplementedError

    @abstractmethod
    def remove(
        self, general_value_function_ids: Sequence[GeneralValueFunctionId]
    ) -> None:
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


class MetaStepSizeLearner(ABC):
    """Adapts per-weight step sizes using meta-gradient methods.

    Implementations may use IDBD (Sutton 1992), Adam-IDBD
    (Degris et al. 2024), or other online cross-validation algorithms.
    Each learned weight in the target module gets a dedicated step-size
    parameter adapted by this learner.

    The agent loop passes error signals (TD errors, reward, etc.) to
    each module's `update_meta()`; composite implementations delegate
    to this learner.
    """

    @abstractmethod
    def update(self, error_signals: Mapping[str, float]) -> None:
        """Receive error signals and adapt per-weight step sizes."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────
# Reactive Policy components
# ─────────────────────────────────────────────────────────────────────


class Option(ABC, Generic[SubjectiveStateT, ActionT]):
    """Temporal abstraction consisting of a policy and termination condition."""

    @property
    @abstractmethod
    def descriptor(self) -> OptionDescriptor:
        raise NotImplementedError

    @abstractmethod
    def is_available(self, subjective_state: SubjectiveStateT) -> bool:
        """Whether this option can be initiated in the given state.

        Intended for `ActionSelector` implementations, which receive
        available options and may filter by initiation conditions.
        """
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
    def update(self, transition: Transition[ActionT, SubjectiveStateT, InfoT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_options(self) -> Sequence[Option[SubjectiveStateT, ActionT]]:
        raise NotImplementedError

    @abstractmethod
    def remove_subtasks(self, subtask_ids: Sequence[SubtaskId]) -> None:
        raise NotImplementedError


class ActionSelector(ABC, Generic[SubjectiveStateT, ActionT]):
    """Chooses primitive actions or options from the current subjective state.

    This is the foreground action-selection mechanism.  It may be as small
    as a hand-written policy for a toy domain or as complex as a learned
    policy head over a rich subjective state representation.
    """

    @abstractmethod
    def decide(
        self,
        subjective_state: SubjectiveStateT,
        active_option: Option[SubjectiveStateT, ActionT] | None,
        available_options: Sequence[Option[SubjectiveStateT, ActionT]],
    ) -> "PolicyDecision[ActionT]":
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
    """Composes multiple options into a single blended behavior.

    Named after Sutton's analogy: each option is a key on a keyboard,
    and playing a "chord" (setting per-option intensities) produces a
    composed temporal abstraction.  The `ActionSelector` determines
    the intensities, then the keyboard produces a new option descriptor
    representing the blended behavior.

    Used by `CompositeReactivePolicy` when an `ActionSelector`
    returns a `PolicyDecision` with `option_intensities` set.
    """

    @abstractmethod
    def compose(self, intensities: Sequence[float]) -> OptionDescriptor:
        """Blend options according to *intensities* and return the result."""
        raise NotImplementedError
