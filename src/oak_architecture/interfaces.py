from __future__ import annotations

"""The four main OaK interfaces.

These correspond directly to the four modules in Richard Sutton's OaK
architecture: Perception, Transition Model, Value Function, and Reactive
Policy.

An `OaKAgent` is composed of exactly these four objects.  Each interface
captures one of Sutton's architectural roles:

- `Perception`: raw observations → subjective state, features, subtasks.
- `ValueFunction`: value learning, utility assessment, curation.
- `TransitionModel`: world dynamics, option models, planning.
- `ReactivePolicy`: action selection, options, option learning.

To build an OaK agent, implement these four interfaces and pass them to
`OaKAgent`.  For finer-grained control, see
`oak_architecture.fine_grained.components` for the building blocks and
`oak_architecture.fine_grained.composites` for ready-made implementations
that compose those building blocks into these four interfaces.
"""

from abc import ABC, abstractmethod
from typing import (
    Generic,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .types import (
    ActionT,
    CurationDecision,
    FeatureId,
    FeatureSpec,
    GeneralValueFunctionId,
    InfoT,
    ObservationT,
    OptionId,
    PlanningUpdate,
    SubjectiveStateT,
    SubtaskId,
    SubtaskSpec,
    TimeStep,
    Transition,
    UsageRecord,
    UtilityRecord,
)


# ─────────────────────────────────────────────────────────────────────
# Environment protocol
# ─────────────────────────────────────────────────────────────────────


@runtime_checkable
class World(Protocol[ObservationT, ActionT, InfoT]):
    """Minimal environment protocol.

    A `World` may wrap a simulator, a benchmark environment, or a custom
    continual data source.  The protocol is intentionally small so the
    package does not depend on a specific environment library.

    Implement this protocol for any environment you want to use with
    `OaKAgent.train()`.
    """

    def reset(self) -> TimeStep[ObservationT, InfoT]: ...

    def step(self, action: ActionT) -> TimeStep[ObservationT, InfoT]: ...

    def close(self) -> None:
        """Release environment resources.  Default is a no-op."""
        ...


# ─────────────────────────────────────────────────────────────────────
# Continual-learning mixin (meta-learned step sizes)
# ─────────────────────────────────────────────────────────────────────


class ContinualLearner:
    """Mixin for modules whose weights are adapted by meta-learned step sizes.

    In Sutton's OaK architecture, every learned weight has a dedicated
    step-size parameter adapted via online cross-validation (e.g. IDBD,
    Sutton 1992; Adam-IDBD, Degris et al. 2024).

    The agent loop calls `update_meta()` on all four modules after each
    learning step, passing the same error-signals dict.  Each module
    internally decides which signals are relevant and routes them to its
    per-weight step-size adaptation.

    The default implementation is a no-op so that modules without
    meta-learning still work unchanged.
    """

    def update_meta(self, error_signals: Mapping[str, float]) -> None:
        """Adapt internal per-weight step sizes given error signals.

        Parameters
        ----------
        error_signals:
            Named scalar error signals from the current learning step,
            e.g. `{"main_td_error": 0.05, "reward": 1.0}`.
            Implementations pick the signals they need and ignore the rest.
        """


# ─────────────────────────────────────────────────────────────────────
# The four main OaK interfaces
# ─────────────────────────────────────────────────────────────────────


class Perception(
    ContinualLearner, ABC, Generic[ObservationT, ActionT, SubjectiveStateT]
):
    """Sutton's Perception: observations → subjective state + feature management.

    Turns raw observations into the agent's **subjective state**, the
    internal representation that every other module sees.  Also discovers,
    ranks, and manages **features** (learned representational structures
    that grow over the agent's lifetime) and generates **subtasks** from
    the most useful ones.

    The finer-grained layer splits this into `StateBuilder`,
    `FeatureBank`, `FeatureConstructor`, `FeatureRanker`, and
    `SubtaskGenerator` (see `oak_architecture.fine_grained.components`).
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset all perception state for a new episode."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        observation: ObservationT,
        reward: float,
        last_action: ActionT | None,
    ) -> SubjectiveStateT:
        """Process a new observation and return the updated subjective state."""
        raise NotImplementedError

    @abstractmethod
    def current_subjective_state(self) -> SubjectiveStateT:
        """Return the most recently computed subjective state."""
        raise NotImplementedError

    @abstractmethod
    def discover_and_rank_features(
        self,
        subjective_state: SubjectiveStateT,
        utility_scores: Sequence[UtilityRecord],
        feature_budget: int,
    ) -> Sequence[FeatureId]:
        """Propose new features, integrate them, and return the top-ranked IDs.

        A typical implementation:

        1. Proposes candidate features from the current subjective state.
        2. Adds accepted candidates to its internal feature store.
        3. Ranks all features using the provided utility scores.
        4. Returns the top feature IDs (up to *feature_budget*).
        """
        raise NotImplementedError

    @abstractmethod
    def generate_subtasks(
        self,
        ranked_feature_ids: Sequence[FeatureId],
    ) -> Sequence[SubtaskSpec]:
        """Turn ranked feature IDs into subtask specifications."""
        raise NotImplementedError

    @abstractmethod
    def list_features(self) -> Sequence[FeatureSpec]:
        """Return all currently tracked features."""
        raise NotImplementedError

    @abstractmethod
    def remove_features(self, feature_ids: Sequence[FeatureId]) -> None:
        """Remove features by ID (called during curation)."""
        raise NotImplementedError


class ValueFunction(ContinualLearner, ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Sutton's Value Function: value learning + utility assessment + curation.

    Learns **predictive value signals** (TD errors, GVF predictions) from
    observed transitions and predicts cumulative signals for any given
    subjective state.  Also assesses the **utility** of the agent's learned
    structures (features, options, models) and produces concrete keep/drop
    **curation** decisions.

    The finer-grained layer splits this into `ValueEstimator`,
    `GeneralValueFunctionLearner`, `UtilityAssessor`, `Curator`, and
    `MetaStepSizeLearner` (see `oak_architecture.fine_grained.components`).
    """

    @abstractmethod
    def update(
        self,
        transition: Transition[ActionT, SubjectiveStateT, InfoT],
    ) -> Mapping[GeneralValueFunctionId, float]:
        """Learn from a transition and return TD-error signals."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        subjective_state: SubjectiveStateT,
    ) -> Mapping[GeneralValueFunctionId, float]:
        """Predict values for the given subjective state."""
        raise NotImplementedError

    @abstractmethod
    def observe_usage(self, usage_records: Sequence[UsageRecord]) -> None:
        """Record usage evidence for utility assessment."""
        raise NotImplementedError

    @abstractmethod
    def utility_scores(self) -> Sequence[UtilityRecord]:
        """Return current utility estimates for all tracked structures."""
        raise NotImplementedError

    @abstractmethod
    def curate(self) -> CurationDecision:
        """Decide which learned structures to drop."""
        raise NotImplementedError

    @abstractmethod
    def remove(
        self,
        general_value_function_ids: Sequence[GeneralValueFunctionId],
    ) -> None:
        """Remove value functions by ID (called during curation)."""
        raise NotImplementedError


class TransitionModel(ContinualLearner, ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Sutton's Transition Model: world dynamics + option models + planning.

    Learns from observed transitions, maintains **option models** that
    predict the effect of temporal abstractions, and runs bounded
    **planning** using the world model and the value function to produce
    improvement signals for the reactive policy.

    The finer-grained layer splits this into `WorldModel`,
    `OptionModelLearner`, individual `OptionModel` objects, and a
    `Planner` (see `oak_architecture.fine_grained.components`).
    """

    @abstractmethod
    def update(
        self,
        transition: Transition[ActionT, SubjectiveStateT, InfoT],
    ) -> None:
        """Learn from an observed transition.

        This should update both the world model and any option-model learners.
        """
        raise NotImplementedError

    @abstractmethod
    def integrate_option_models(self) -> None:
        """Export learned option models and integrate them into the world model.

        Called after option learning so that planning reasons over fresh models.
        """
        raise NotImplementedError

    @abstractmethod
    def plan(
        self,
        subjective_state: SubjectiveStateT,
        value_function: ValueFunction[SubjectiveStateT, ActionT, InfoT],
        budget: int,
    ) -> PlanningUpdate[ActionT]:
        """Run bounded planning and return improvement signals.

        The planner uses the internal world model together with the supplied
        *value_function* (for state evaluation) to produce value targets,
        policy targets, or search statistics.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_option_models(self, option_ids: Sequence[OptionId]) -> None:
        """Remove option models by ID (called during curation)."""
        raise NotImplementedError


class ReactivePolicy(ContinualLearner, ABC, Generic[SubjectiveStateT, ActionT, InfoT]):
    """Sutton's Reactive Policy: action selection + option management.

    Selects **actions**, primitive or temporal abstractions (options),
    based on the current subjective state.  Manages the **option library**
    and **option learning** pipeline, and integrates **planning updates**
    into decision-making.

    The finer-grained layer splits this into `ActionSelector`,
    `OptionLibrary`, and `OptionLearner`
    (see `oak_architecture.fine_grained.components`).
    """

    @abstractmethod
    def update(
        self,
        transition: Transition[ActionT, SubjectiveStateT, InfoT],
        td_errors: Mapping[GeneralValueFunctionId, float],
    ) -> None:
        """Update the policy and option learners from an observed transition."""
        raise NotImplementedError

    @abstractmethod
    def apply_planning_update(self, update: PlanningUpdate[ActionT]) -> None:
        """Integrate planning improvement signals into the policy."""
        raise NotImplementedError

    @abstractmethod
    def ingest_subtasks(self, subtasks: Sequence[SubtaskSpec]) -> None:
        """Feed newly created subtasks into the option learner."""
        raise NotImplementedError

    @abstractmethod
    def integrate_options(self) -> None:
        """Export learned options into the option library."""
        raise NotImplementedError

    @abstractmethod
    def select_action(
        self,
        subjective_state: SubjectiveStateT,
        option_stop_threshold: float,
    ) -> tuple[ActionT, OptionId | None]:
        """Choose a primitive action, possibly by continuing an active option.

        Returns a `(primitive_action, active_option_id)` pair.  When no
        option is active, *active_option_id* is `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def clear_active_option(self) -> None:
        """Clear the currently executing option (e.g. at episode boundaries)."""
        raise NotImplementedError

    @abstractmethod
    def remove_options(self, option_ids: Sequence[OptionId]) -> None:
        """Remove options by ID (called during curation)."""
        raise NotImplementedError

    @abstractmethod
    def remove_subtasks(self, subtask_ids: Sequence[SubtaskId]) -> None:
        """Remove subtasks by ID (called during curation)."""
        raise NotImplementedError
