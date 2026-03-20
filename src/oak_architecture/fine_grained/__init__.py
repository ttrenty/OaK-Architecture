"""Optional fine-grained OaK building blocks and composites.

The default public surface of OaK is the four main interfaces in
`oak_architecture.interfaces` together with `OaKAgent`.

This subpackage exposes a more detailed assembly layer for projects that want
to swap internal pieces such as a planner, world model, or feature constructor
independently.
"""

from .composites import (
    CompositePerception,
    CompositeReactivePolicy,
    CompositeTransitionModel,
    CompositeValueFunction,
)
from .components import (
    ActionSelector,
    Curator,
    FeatureBank,
    FeatureConstructor,
    FeatureRanker,
    GeneralValueFunctionLearner,
    MetaStepSizeLearner,
    Option,
    OptionKeyboard,
    OptionLearner,
    OptionLibrary,
    OptionModel,
    OptionModelLearner,
    Planner,
    StateBuilder,
    SubtaskGenerator,
    UtilityAssessor,
    ValueEstimator,
    WorldModel,
)

__all__ = [
    "CompositePerception",
    "CompositeTransitionModel",
    "CompositeValueFunction",
    "CompositeReactivePolicy",
    "ActionSelector",
    "Curator",
    "FeatureBank",
    "FeatureConstructor",
    "FeatureRanker",
    "GeneralValueFunctionLearner",
    "MetaStepSizeLearner",
    "Option",
    "OptionKeyboard",
    "OptionLearner",
    "OptionLibrary",
    "OptionModel",
    "OptionModelLearner",
    "Planner",
    "StateBuilder",
    "SubtaskGenerator",
    "UtilityAssessor",
    "ValueEstimator",
    "WorldModel",
]
