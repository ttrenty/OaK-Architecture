"""Example 01: a full OaK RL agent with discovery and described-world support.

The learning modules are environment-agnostic: pass any compatible `World` to
`run_training()` and the same agent pipeline can be reused.

CartPole is bundled as one sample gymnasium world description, but it is not
part of the example's identity.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_agent",
    "run_training",
    "AgentObservation",
    "ExampleAgentSpec",
    "ExampleSubjectiveState",
    "PerceptionPlan",
    "GymWorld",
    "DescribedGymWorld",
    "CARTPOLE_WORLD_DESCRIPTION",
    "ACROBOT_WORLD_DESCRIPTION",
    "PIXEL_CARTPOLE_WORLD_DESCRIPTION",
    "MULTIMODAL_CARTPOLE_WORLD_DESCRIPTION",
    "WorldDescription",
    "ArcWorld",
    "EpisodeAnimationRecorder",
    "EpisodeCaptureSchedule",
    "TrainingCurveRecorder",
    "animation_recorder_from_env",
    "curve_recorder_from_env",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the public example API without importing torch eagerly."""
    exports: dict[str, object]
    if name in {"build_agent", "run_training"}:
        from .runner import build_agent, run_training

        exports = {
            "build_agent": build_agent,
            "run_training": run_training,
        }
    elif name in {
        "AgentObservation",
        "ExampleAgentSpec",
        "ExampleSubjectiveState",
        "PerceptionPlan",
        "GymWorld",
    }:
        from .schema import AgentObservation, ExampleAgentSpec, ExampleSubjectiveState, PerceptionPlan
        from .world import GymWorld

        exports = {
            "AgentObservation": AgentObservation,
            "ExampleAgentSpec": ExampleAgentSpec,
            "ExampleSubjectiveState": ExampleSubjectiveState,
            "PerceptionPlan": PerceptionPlan,
            "GymWorld": GymWorld,
        }
    elif name in {
        "DescribedGymWorld",
        "CARTPOLE_WORLD_DESCRIPTION",
        "ACROBOT_WORLD_DESCRIPTION",
        "PIXEL_CARTPOLE_WORLD_DESCRIPTION",
        "MULTIMODAL_CARTPOLE_WORLD_DESCRIPTION",
        "WorldDescription",
        "EpisodeAnimationRecorder",
        "EpisodeCaptureSchedule",
        "TrainingCurveRecorder",
        "animation_recorder_from_env",
        "curve_recorder_from_env",
    }:
        from .world_embedded import (
            ACROBOT_WORLD_DESCRIPTION,
            CARTPOLE_WORLD_DESCRIPTION,
            DescribedGymWorld,
            WorldDescription,
        )
        from .world_pixel import (
            MULTIMODAL_CARTPOLE_WORLD_DESCRIPTION,
            PIXEL_CARTPOLE_WORLD_DESCRIPTION,
        )
        from .training_logging import (
            EpisodeAnimationRecorder,
            EpisodeCaptureSchedule,
            TrainingCurveRecorder,
            animation_recorder_from_env,
            curve_recorder_from_env,
        )

        exports = {
            "DescribedGymWorld": DescribedGymWorld,
            "CARTPOLE_WORLD_DESCRIPTION": CARTPOLE_WORLD_DESCRIPTION,
            "ACROBOT_WORLD_DESCRIPTION": ACROBOT_WORLD_DESCRIPTION,
            "PIXEL_CARTPOLE_WORLD_DESCRIPTION": PIXEL_CARTPOLE_WORLD_DESCRIPTION,
            "MULTIMODAL_CARTPOLE_WORLD_DESCRIPTION": MULTIMODAL_CARTPOLE_WORLD_DESCRIPTION,
            "WorldDescription": WorldDescription,
            "EpisodeAnimationRecorder": EpisodeAnimationRecorder,
            "EpisodeCaptureSchedule": EpisodeCaptureSchedule,
            "TrainingCurveRecorder": TrainingCurveRecorder,
            "animation_recorder_from_env": animation_recorder_from_env,
            "curve_recorder_from_env": curve_recorder_from_env,
        }
    elif name == "ArcWorld":
        from .world_arc import ArcWorld

        exports = {"ArcWorld": ArcWorld}
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals().update(exports)
    return globals()[name]
