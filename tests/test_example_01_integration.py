"""Integration tests for Example 01 and the smoke examples.

Verifies that:
- All World implementations satisfy the World protocol
- build_agent() produces a valid OaKAgent from a config dict
- agent.train(world) runs the episode loop correctly
- run_training() works in both discovery and embedded modes
- Smoke tests still pass after refactoring
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from examples.example_01 import runner as example_01_runner
from examples.example_01 import (
    DescribedGymWorld,
    EpisodeAnimationRecorder,
    EpisodeCaptureSchedule,
    GymWorld,
    TrainingCurveRecorder,
    animation_recorder_from_env,
    build_agent,
    curve_recorder_from_env,
    run_training,
)
from examples.example_01.llm import _plan_from_llm_result
from examples.example_01.perception import AdaptivePerception
from examples.example_01.schema import (
    ExampleAgentSpec,
    ExampleSubjectiveState,
    PerceptionPlan,
    TensorViewPlan,
    subjective_state_from_tensor,
)
from examples.example_01.schema import StateTensorAdapter
from examples.example_01.transition_model import DynaTransitionModel
from examples.example_01.world_pixel import (
    MULTIMODAL_CARTPOLE_WORLD_DESCRIPTION,
    PIXEL_CARTPOLE_WORLD_DESCRIPTION,
)
from examples.smoke.minimal_oak import MinimalWorld
from examples.smoke.minimal_oak import run_minimal_episode as run_minimal_smoke
from examples.smoke.minimal_oak_fine_grained import run_minimal_episode as run_minimal_fine_grained
from oak import OaKAgent
from oak.interfaces import World
from oak.types import EpisodeTrace, TimeStep, Transition


def test_world_protocol() -> None:
    """All World implementations satisfy the World protocol."""
    world_builders: list[tuple[str, Callable[[], object]]] = [
        ("GymWorld(CartPole)", lambda: GymWorld("CartPole-v1")),
        ("GymWorld(Acrobot)", lambda: GymWorld("Acrobot-v1")),
        ("DescribedGymWorld(CartPole)", lambda: DescribedGymWorld("CartPole-v1")),
        ("DescribedGymWorld(Acrobot)", lambda: DescribedGymWorld("Acrobot-v1")),
        ("MinimalWorld", MinimalWorld),
    ]
    for name, build_world in world_builders:
        w = build_world()
        assert isinstance(w, World), f"{name} does not satisfy World protocol"
        w.close()
        print(f"  {name}: OK")

    print("PASS: all World implementations satisfy the protocol")


def test_build_agent() -> None:
    """build_agent() returns a valid OaKAgent."""
    config = {
        "obs_shape": (4,),
        "action_n": 2,
        "encoder_type": "identity",
    }
    agent = build_agent(config)
    assert isinstance(agent, OaKAgent)
    print("PASS: build_agent() returns OaKAgent")


def test_grouped_feature_hints() -> None:
    """CartPole described mode should expose grouped semantic features."""
    world = DescribedGymWorld("CartPole-v1")
    agent = build_agent(world.description.to_config())
    world.close()

    feature_ids = [feature.feature_id for feature in agent.perception.list_features()]
    assert feature_ids == ["cart_motion", "pole_balance"]
    print("PASS: grouped feature hints preserved")


def test_perception_prioritizes_uncreated_features() -> None:
    """Feature discovery should not get stuck re-ranking the first created feature."""
    world = DescribedGymWorld("CartPole-v1")
    agent = build_agent(world.description.to_config())
    world.close()

    perception = agent.perception
    if not isinstance(perception, AdaptivePerception):
        raise TypeError(f"Unexpected perception type: {type(perception)!r}")

    state = perception.update(
        np.zeros(4, dtype=np.float32),
        reward=0.0,
        last_action=None,
    )
    first_rank = perception.discover_and_rank_features(state, (), 1)
    assert first_rank == ("cart_motion",)
    created = perception.generate_subtasks(first_rank)
    assert [subtask.feature_id for subtask in created] == ["cart_motion"]

    perception._update_count = (
        perception._last_subtask_creation_update + perception._subtask_creation_interval
    )
    second_rank = perception.discover_and_rank_features(state, (), 1)
    assert second_rank == ("pole_balance",)
    print("PASS: perception prioritizes unseen features for later option growth")


def test_perception_bootstraps_multiple_subtasks_when_budget_allows() -> None:
    """The initial perception bootstrap should create all ranked raw-value subtasks."""
    world = DescribedGymWorld("CartPole-v1")
    agent = build_agent(world.description.to_config(), feature_budget=2)
    world.close()

    perception = agent.perception
    if not isinstance(perception, AdaptivePerception):
        raise TypeError(f"Unexpected perception type: {type(perception)!r}")

    state = perception.update(
        np.zeros(4, dtype=np.float32),
        reward=0.0,
        last_action=None,
    )
    ranked = perception.discover_and_rank_features(state, (), 2)
    created = perception.generate_subtasks(ranked)

    assert [subtask.feature_id for subtask in created] == ["cart_motion", "pole_balance"]
    print("PASS: initial raw-value subtasks bootstrap together when budget allows")


def test_perception_normalizes_raw_tensor_views() -> None:
    """Raw-value tensor views should stay finite and bounded after warmup samples."""
    world = DescribedGymWorld("Acrobot-v1")
    agent = build_agent(world.description.to_config())
    world.close()

    perception = agent.perception
    if not isinstance(perception, AdaptivePerception):
        raise TypeError(f"Unexpected perception type: {type(perception)!r}")

    warmup_observations = (
        np.asarray([1.0, 0.0, 1.0, 0.0, -0.5, 0.5], dtype=np.float32),
        np.asarray([0.9, 0.2, 0.8, -0.1, -1.0, 1.0], dtype=np.float32),
        np.asarray([0.7, 0.6, 0.4, -0.5, -2.0, 2.0], dtype=np.float32),
    )
    final_state: ExampleSubjectiveState | None = None
    for observation in warmup_observations:
        final_state = perception.update(observation, reward=-1.0, last_action=0)

    assert final_state is not None
    tensor = final_state.tensor_view()
    assert bool(torch.isfinite(tensor).all().item())
    assert float(torch.max(torch.abs(tensor)).item()) <= 5.01
    print("PASS: raw tensor views are normalized into a finite bounded range")


def test_small_raw_worlds_prefer_cpu_device() -> None:
    """Default device selection should avoid CUDA overhead on tiny raw-state tasks."""
    world = DescribedGymWorld("CartPole-v1")
    original_cuda_available = example_01_runner.torch.cuda.is_available
    example_01_runner.torch.cuda.is_available = lambda: True
    try:
        agent = build_agent(world.description.to_config())
    finally:
        example_01_runner.torch.cuda.is_available = original_cuda_available
        world.close()

    assert agent.value_function._device.type == "cpu"
    print("PASS: tiny raw-value worlds default to CPU")


def test_agent_train_embedded() -> None:
    """agent.train(world) runs on an embedded world."""
    world = DescribedGymWorld("CartPole-v1")
    config = world.description.to_config()
    agent = build_agent(config)
    logged_episodes: list[tuple[int, float, float]] = []

    def log_episode(
        episode: int,
        reward: float,
        avg_reward: float,
        logged_agent: OaKAgent,
    ) -> None:
        if logged_agent is not agent:
            raise AssertionError("episode_logger received a different agent instance")
        logged_episodes.append((episode, reward, avg_reward))

    num_episodes = 3
    rewards = agent.train(world, num_episodes=num_episodes, episode_logger=log_episode)
    world.close()

    assert len(rewards) == num_episodes, f"Expected {num_episodes} episodes, got {len(rewards)}"
    assert len(logged_episodes) == num_episodes, "episode_logger should run once per episode"
    print(f"PASS: agent.train() ran {len(rewards)} episodes")


def test_run_training_embedded() -> None:
    """run_training() works with an embedded world."""
    world = DescribedGymWorld("CartPole-v1")
    num_episodes = 3
    rewards = run_training(world, num_episodes=num_episodes, verbose=False)

    assert len(rewards) == num_episodes, f"Expected {num_episodes} episodes, got {len(rewards)}"
    print(f"PASS: run_training(embedded) ran {len(rewards)} episodes")


def test_run_training_discovery() -> None:
    """run_training() works with a discovery world."""
    world = GymWorld("CartPole-v1")
    original_analyze_world = example_01_runner.analyze_world
    example_01_runner.analyze_world = lambda *_args, **_kwargs: None
    try:
        rewards = run_training(world, num_episodes=10, verbose=False)
    finally:
        example_01_runner.analyze_world = original_analyze_world

    assert len(rewards) == 10, f"Expected 10 episodes, got {len(rewards)}"
    print(f"PASS: run_training(discovery) ran {len(rewards)} episodes")


def test_episode_trace_logger_can_access_world_and_frames() -> None:
    """Selected episode traces should expose the world and rendered frames."""
    world = DescribedGymWorld("CartPole-v1")
    frame_counter = {"count": 0}

    def synthetic_frame() -> np.ndarray:
        frame_counter["count"] += 1
        value = min(frame_counter["count"] * 16, 255)
        return np.full((16, 16, 3), value, dtype=np.uint8)

    world.render_frame = synthetic_frame
    original_analyze_world = example_01_runner.analyze_world
    example_01_runner.analyze_world = lambda *_args, **_kwargs: None
    traces: list[EpisodeTrace[Any, Any, Any, Any]] = []
    try:
        rewards = run_training(
            world,
            num_episodes=2,
            verbose=False,
            episode_trace_logger=traces.append,
            trace_selector=lambda episode, _total: episode == 1,
            capture_rendered_frames=True,
        )
    finally:
        example_01_runner.analyze_world = original_analyze_world

    assert len(rewards) == 2
    assert len(traces) == 1
    trace = traces[0]
    assert trace.world is world
    assert trace.agent is not None
    assert trace.episode == 1
    assert trace.frames, "Expected at least one captured render frame"
    assert isinstance(trace.frames[0], np.ndarray)
    assert trace.step_count == len(trace.steps)
    print("PASS: episode traces expose the world and captured frames")


def test_raw_world_preserves_gym_observation() -> None:
    """Discovery worlds should expose the raw Gym observation unchanged."""
    world = GymWorld("CartPole-v1")
    ts = world.reset()
    world.close()

    assert isinstance(ts.observation, np.ndarray)
    assert tuple(ts.observation.shape) == (4,)
    print("PASS: raw discovery world preserves Gym observation")


def test_dyna_planning_skips_terminal_samples() -> None:
    """Synthetic planning should never start from transitions that already ended."""

    class _NoOpValueFunction:
        def __init__(self) -> None:
            self.calls = 0

        def update(self, transition: Any, *, planning: bool = False) -> dict[str, float]:
            self.calls += 1
            return {}

    model = DynaTransitionModel(
        state_dim=4,
        num_actions=2,
        model_train_batch=1,
        planning_warmup_steps=1,
    )
    state = subjective_state_from_tensor(torch.zeros(4), view_name="main")
    next_state = subjective_state_from_tensor(torch.ones(4), view_name="main")
    transition: Transition[Any, Any, dict[str, Any]] = Transition(
        subjective_state=state,
        action=0,
        reward=1.0,
        next_subjective_state=next_state,
        terminated=True,
        option_id="option:test",
    )
    model.update(transition)

    dummy_value_function = _NoOpValueFunction()
    planning_update = model.plan(
        state,
        cast(Any, dummy_value_function),
        budget=4,
    )

    assert planning_update.search_statistics["planning_steps"] == 0
    assert dummy_value_function.calls == 0
    print("PASS: Dyna planning ignores terminal transition starts")


def test_run_training_supports_split_identity_llm_views() -> None:
    """LLM split identity views should not crash due to latent/input mismatch."""
    world = DescribedGymWorld("CartPole-v1")
    original_analyze_world = example_01_runner.analyze_world

    split_plan = PerceptionPlan(
        world_description=world.description,
        feature_groups=world.description.feature_hints,
        tensor_views=(
            TensorViewPlan(
                view_id="cart_2d",
                source_channel="main",
                encoder_type="identity",
                input_shape=(2,),
                input_dim=2,
                latent_dim=2,
                selector_names=("cart_position", "cart_velocity"),
                description="Cart-only raw-value slice.",
            ),
            TensorViewPlan(
                view_id="pole_2d",
                source_channel="main",
                encoder_type="identity",
                input_shape=(2,),
                input_dim=2,
                latent_dim=2,
                selector_names=("pole_angle", "pole_angular_velocity"),
                description="Pole-only raw-value slice.",
            ),
        ),
        default_tensor_view="cart_2d",
        llm_used=True,
    )
    example_01_runner.analyze_world = lambda *_args, **_kwargs: split_plan
    try:
        rewards = run_training(world, num_episodes=3, verbose=False)
    finally:
        example_01_runner.analyze_world = original_analyze_world

    assert len(rewards) == 3
    print("PASS: split identity LLM tensor views train without shape mismatch")


def test_llm_plan_prefers_full_raw_state_as_default_tensor_view() -> None:
    """Subset raw-value views should not become the default control state."""
    world = DescribedGymWorld("CartPole-v1")
    result = {
        "default_tensor_view": "cart_motion_view",
        "tensor_views": [
            {
                "view_id": "cart_motion_view",
                "source_channel": "main",
                "encoder_type": "identity",
                "latent_dim": 2,
                "description": "Cart-only state slice.",
                "selector_names": ["cart_position", "cart_velocity"],
            },
            {
                "view_id": "pole_balance_view",
                "source_channel": "main",
                "encoder_type": "identity",
                "latent_dim": 2,
                "description": "Pole-only state slice.",
                "selector_names": ["pole_angle", "pole_angular_velocity"],
            },
            {
                "view_id": "full_state_view",
                "source_channel": "main",
                "encoder_type": "identity",
                "latent_dim": 4,
                "description": "Complete raw state.",
                "selector_names": [
                    "cart_position",
                    "cart_velocity",
                    "pole_angle",
                    "pole_angular_velocity",
                ],
            },
        ],
        "feature_groups": [
            {
                "field_id": "cart_motion",
                "name": "Cart motion",
                "source_channel": "main",
                "description": "Cart coordinates.",
                "selector_names": ["cart_position", "cart_velocity"],
                "selector_indices": [],
            }
        ],
        "notes": "Planner output with an unsafe subset default.",
    }

    plan = _plan_from_llm_result(world.description, result)
    world.close()

    assert plan.default_tensor_view == "full_state_view"
    print("PASS: LLM subset defaults are promoted to the full raw state view")


def test_llm_plan_adds_identity_full_raw_control_view() -> None:
    """Raw-value control should not default to an untrained MLP full-state view."""
    world = DescribedGymWorld("CartPole-v1")
    result = {
        "default_tensor_view": "full_state_view",
        "tensor_views": [
            {
                "view_id": "full_state_view",
                "source_channel": "main",
                "encoder_type": "mlp",
                "latent_dim": 32,
                "description": "LLM proposed MLP full state.",
                "selector_names": [
                    "cart_position",
                    "cart_velocity",
                    "pole_angle",
                    "pole_angular_velocity",
                ],
            }
        ],
        "feature_groups": [
            {
                "field_id": "cart_motion",
                "name": "Cart motion",
                "source_channel": "main",
                "description": "Cart coordinates.",
                "selector_names": ["cart_position", "cart_velocity"],
                "selector_indices": [],
            }
        ],
        "notes": "Unsafe raw-value MLP control proposal.",
    }

    plan = _plan_from_llm_result(world.description, result)
    world.close()

    default_view = plan.view()
    assert default_view.encoder_type == "identity"
    assert default_view.input_dim == 4
    print("PASS: raw-value plans gain an identity full-state control view")


def test_llm_plan_preserves_curated_raw_feature_groups() -> None:
    """Small raw-value worlds should keep the curated feature grouping."""
    world = DescribedGymWorld("CartPole-v1")
    result = {
        "default_tensor_view": "full_state_view",
        "tensor_views": [
            {
                "view_id": "full_state_view",
                "source_channel": "main",
                "encoder_type": "identity",
                "latent_dim": 4,
                "description": "Full state.",
                "selector_names": [
                    "cart_position",
                    "cart_velocity",
                    "pole_angle",
                    "pole_angular_velocity",
                ],
            }
        ],
        "feature_groups": [
            {
                "field_id": "cart_motion",
                "name": "Cart motion",
                "source_channel": "main",
                "description": "Cart coordinates.",
                "selector_names": ["cart_position", "cart_velocity"],
                "selector_indices": [],
            },
            {
                "field_id": "pole_balance",
                "name": "Pole balance",
                "source_channel": "main",
                "description": "Pole coordinates.",
                "selector_names": ["pole_angle", "pole_angular_velocity"],
                "selector_indices": [],
            },
            {
                "field_id": "state",
                "name": "State",
                "source_channel": "main",
                "description": "Redundant whole-state group.",
                "selector_names": [
                    "cart_position",
                    "cart_velocity",
                    "pole_angle",
                    "pole_angular_velocity",
                ],
                "selector_indices": [],
            },
        ],
        "notes": "LLM over-specified the raw-value groups.",
    }

    plan = _plan_from_llm_result(world.description, result)
    world.close()

    assert [field.field_id for field in plan.feature_groups] == [
        "cart_motion",
        "pole_balance",
    ]
    print("PASS: curated raw-value feature groups override redundant LLM groups")


def test_episode_animation_recorder_writes_gif() -> None:
    """EpisodeAnimationRecorder should persist GIF and metadata artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = EpisodeAnimationRecorder(
            output_dir=Path(tmpdir),
            schedule=EpisodeCaptureSchedule(episode_indices=(0,)),
            prefix="test_trace",
            fps=12,
        )
        trace = EpisodeTrace[Any, Any, Any, Any](
            episode=0,
            episode_reward=12.0,
            avg_reward=12.0,
            step_count=2,
            solved=False,
            initial_time_step=TimeStep(observation=np.zeros(4, dtype=np.float32), reward=0.0),
            final_time_step=TimeStep(
                observation=np.ones(4, dtype=np.float32),
                reward=1.0,
                terminated=True,
            ),
            frames=(
                np.zeros((16, 16, 3), dtype=np.uint8),
                np.full((16, 16, 3), 255, dtype=np.uint8),
            ),
            steps=(),
            metadata={},
        )
        recorder(trace)

        saved = sorted(Path(tmpdir).glob("test_trace_episode_0000_*"))
        assert any(path.suffix == ".gif" for path in saved)
        assert any(path.suffix == ".json" for path in saved)
    print("PASS: animation recorder writes GIF artifacts")


def test_animation_recorder_from_env_uses_defaults_and_opt_out() -> None:
    """Main example runners should auto-enable animation capture with safe defaults."""
    env_keys = (
        "OAK_EXAMPLE_DISABLE_ANIMATIONS",
        "OAK_EXAMPLE_ANIMATION_DIR",
        "OAK_EXAMPLE_ANIMATION_EPISODES",
        "OAK_EXAMPLE_ANIMATION_EVERY",
        "OAK_EXAMPLE_ANIMATION_LAST",
        "OAK_EXAMPLE_ANIMATION_FPS",
        "OAK_EXAMPLE_ANIMATION_MAX_FRAMES",
    )
    previous = {key: os.environ.get(key) for key in env_keys}
    try:
        for key in env_keys:
            os.environ.pop(key, None)

        recorder = animation_recorder_from_env("cartpole_v1_described")
        assert recorder is not None
        assert recorder.output_dir == Path("tests/results/animations/cartpole_v1_described")
        assert recorder.schedule.every_n_episodes == 100
        assert recorder.schedule.last_n_episodes == 1
        assert recorder.fps == 30
        assert recorder.max_frames == 400

        os.environ["OAK_EXAMPLE_DISABLE_ANIMATIONS"] = "1"
        assert animation_recorder_from_env("cartpole_v1_described") is None
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("PASS: animation recorder defaults are automatic and can be disabled")


def test_training_curve_recorder_writes_svg_and_json() -> None:
    """TrainingCurveRecorder should persist reward curves and raw history."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = TrainingCurveRecorder(
            output_dir=Path(tmpdir),
            prefix="test_curve",
            average_window=5,
        )

        class _DummyPolicy:
            epsilon = 0.42
            _options = {"o0": object(), "o1": object()}

        class _DummyAgent:
            reactive_policy = _DummyPolicy()

        agent = _DummyAgent()
        for episode, reward in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            recorder.log_episode(
                episode,
                reward,
                avg_reward=float(sum(range(1, episode + 2)) / (episode + 1)),
                agent=agent,
            )
        recorder.save()

        reward_svg = Path(tmpdir) / "test_curve_reward_curve.svg"
        reward_png = Path(tmpdir) / "test_curve_reward_curve.png"
        state_svg = Path(tmpdir) / "test_curve_training_state.svg"
        state_png = Path(tmpdir) / "test_curve_training_state.png"
        history_json = Path(tmpdir) / "test_curve_reward_history.json"

        assert reward_svg.exists()
        assert reward_png.exists()
        assert state_svg.exists()
        assert state_png.exists()
        assert history_json.exists()
        reward_svg_text = reward_svg.read_text()
        assert ">0</text>" in reward_svg_text
        assert ">4</text>" in reward_svg_text
        payload = json.loads(history_json.read_text())
        assert payload["episodes"] == 5
        assert payload["average_window"] == 5
        print("PASS: training curve recorder writes SVG and JSON artifacts")


def test_training_curve_recorder_writes_loss_metric_plots() -> None:
    """TrainingCurveRecorder should persist raw + averaged learner-loss plots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = TrainingCurveRecorder(
            output_dir=Path(tmpdir),
            prefix="test_curve",
            average_window=3,
        )

        class _DummyPerception:
            def training_metrics(self) -> dict[str, float]:
                return {}

        class _DummyValueFunction:
            def __init__(self) -> None:
                self._values = iter([0.9, 0.7, 0.5, 0.3])

            def training_metrics(self) -> dict[str, float]:
                return {"value_q_omega_loss": next(self._values)}

        class _DummyPolicy:
            epsilon = 0.42
            _options = {"o0": object(), "o1": object()}

            def training_metrics(self) -> dict[str, float]:
                return {"policy_q_loss": 0.25}

        class _DummyTransitionModel:
            def training_metrics(self) -> dict[str, float]:
                return {"model_loss": 0.1, "model_done_loss": 0.05}

        class _DummyAgent:
            perception = _DummyPerception()
            value_function = _DummyValueFunction()
            reactive_policy = _DummyPolicy()
            transition_model = _DummyTransitionModel()

        agent = _DummyAgent()
        for episode, reward in enumerate([1.0, 2.0, 3.0, 4.0]):
            recorder.log_episode(
                episode,
                reward,
                avg_reward=float(reward),
                agent=agent,
            )
        recorder.save()

        assert (Path(tmpdir) / "test_curve_value_q_omega_loss.svg").exists()
        assert (Path(tmpdir) / "test_curve_value_q_omega_loss.png").exists()
        assert (Path(tmpdir) / "test_curve_policy_q_loss.svg").exists()
        assert (Path(tmpdir) / "test_curve_policy_q_loss.png").exists()
        assert (Path(tmpdir) / "test_curve_model_loss.svg").exists()
        assert (Path(tmpdir) / "test_curve_model_loss.png").exists()
        payload = json.loads((Path(tmpdir) / "test_curve_reward_history.json").read_text())
        assert "value_q_omega_loss" in payload["metric_histories"]
        print("PASS: training curve recorder writes learner-loss plots")


def test_curve_recorder_from_env_uses_defaults_and_opt_out() -> None:
    """Main example runners should auto-enable curve capture with safe defaults."""
    env_keys = (
        "OAK_EXAMPLE_DISABLE_CURVES",
        "OAK_EXAMPLE_CURVE_DIR",
        "OAK_EXAMPLE_PLOT_DIR",
    )
    previous = {key: os.environ.get(key) for key in env_keys}
    try:
        for key in env_keys:
            os.environ.pop(key, None)

        recorder = curve_recorder_from_env("cartpole_v1_described", average_window=100)
        assert recorder is not None
        assert recorder.output_dir == Path("tests/results/training_curves/cartpole_v1_described")

        os.environ["OAK_EXAMPLE_DISABLE_CURVES"] = "1"
        assert curve_recorder_from_env("cartpole_v1_described", average_window=100) is None
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("PASS: curve recorder defaults are automatic and can be disabled")


def test_llm_plan_adds_missing_full_raw_state_view() -> None:
    """LLM plans with only partial raw slices should gain a safe full-state view."""
    world = DescribedGymWorld("CartPole-v1")
    result = {
        "default_tensor_view": "cart_motion_view",
        "tensor_views": [
            {
                "view_id": "cart_motion_view",
                "source_channel": "main",
                "encoder_type": "identity",
                "latent_dim": 2,
                "description": "Cart-only state slice.",
                "selector_names": ["cart_position", "cart_velocity"],
            },
            {
                "view_id": "pole_balance_view",
                "source_channel": "main",
                "encoder_type": "identity",
                "latent_dim": 2,
                "description": "Pole-only state slice.",
                "selector_names": ["pole_angle", "pole_angular_velocity"],
            },
        ],
        "feature_groups": [
            {
                "field_id": "cart_motion",
                "name": "Cart motion",
                "source_channel": "main",
                "description": "Cart coordinates.",
                "selector_names": ["cart_position", "cart_velocity"],
                "selector_indices": [],
            }
        ],
        "notes": "Planner output missing the complete state view.",
    }

    plan = _plan_from_llm_result(world.description, result)
    world.close()

    full_views = [
        view
        for view in plan.tensor_views
        if view.source_channel == "main" and int(view.input_dim or 0) == 4
    ]
    assert full_views, "Expected the planner sanitizer to add a full-state raw view"
    assert plan.default_tensor_view == full_views[0].view_id
    print("PASS: missing full raw-state views are synthesized automatically")


def test_build_agent_uses_full_raw_state_for_control() -> None:
    """build_agent() should not wire a subset raw-value view as the control state."""
    world = DescribedGymWorld("CartPole-v1")
    unsafe_plan = PerceptionPlan(
        world_description=world.description,
        feature_groups=world.description.feature_hints,
        tensor_views=(
            TensorViewPlan(
                view_id="cart_motion_view",
                source_channel="main",
                encoder_type="identity",
                input_shape=(2,),
                input_dim=2,
                latent_dim=2,
                selector_names=("cart_position", "cart_velocity"),
                description="Unsafe subset control view.",
            ),
            TensorViewPlan(
                view_id="pole_balance_view",
                source_channel="main",
                encoder_type="identity",
                input_shape=(2,),
                input_dim=2,
                latent_dim=2,
                selector_names=("pole_angle", "pole_angular_velocity"),
                description="Unsafe subset control view.",
            ),
        ),
        default_tensor_view="cart_motion_view",
        llm_used=True,
    )

    agent = build_agent(ExampleAgentSpec(perception_plan=unsafe_plan, source="test"))
    world.close()

    assert agent.value_function._q_net.net[0].in_features == 4
    assert agent.value_function._state_adapter.view_name != "cart_motion_view"
    print("PASS: build_agent promotes raw control to the full state width")


def test_acrobot_policy_uses_episode_based_epsilon_decay() -> None:
    """Acrobot exploration should not collapse just because episodes are long."""
    world = DescribedGymWorld("Acrobot-v1")
    agent = build_agent(world.description.to_config())
    world.close()

    policy = agent.reactive_policy
    policy._step_count = 20_000
    policy._episode_count = 20

    assert policy.epsilon > 0.9
    print("PASS: Acrobot epsilon decays by episode, not raw step count")


def test_acrobot_described_features() -> None:
    """Acrobot described mode should expose grouped semantic features."""
    world = DescribedGymWorld("Acrobot-v1")
    agent = build_agent(world.description.to_config())
    world.close()

    feature_ids = [feature.feature_id for feature in agent.perception.list_features()]
    assert feature_ids == ["link1_state", "link2_state"]
    print("PASS: Acrobot grouped feature hints preserved")


def test_run_training_acrobot_embedded() -> None:
    """run_training() works with an embedded Acrobot world."""
    world = DescribedGymWorld("Acrobot-v1")
    num_episodes = 3
    rewards = run_training(world, num_episodes=num_episodes, verbose=False)

    assert len(rewards) == num_episodes, f"Expected {num_episodes} episodes, got {len(rewards)}"
    print(f"PASS: run_training(Acrobot embedded) ran {len(rewards)} episodes")


def test_run_training_acrobot_discovery() -> None:
    """run_training() works with an Acrobot discovery world."""
    world = GymWorld("Acrobot-v1")
    original_analyze_world = example_01_runner.analyze_world
    example_01_runner.analyze_world = lambda *_args, **_kwargs: None
    try:
        rewards = run_training(world, num_episodes=5, verbose=False)
    finally:
        example_01_runner.analyze_world = original_analyze_world

    assert len(rewards) == 5, f"Expected 5 episodes, got {len(rewards)}"
    print(f"PASS: run_training(Acrobot discovery) ran {len(rewards)} episodes")


def test_raw_acrobot_world_preserves_gym_observation() -> None:
    """Discovery Acrobot worlds should expose the raw Gym observation unchanged."""
    world = GymWorld("Acrobot-v1")
    ts = world.reset()
    world.close()

    assert isinstance(ts.observation, np.ndarray)
    assert tuple(ts.observation.shape) == (6,)
    print("PASS: raw Acrobot discovery world preserves Gym observation")


def test_pixel_cartpole_world_protocol() -> None:
    """Pixel CartPole worlds satisfy the World protocol."""
    for env_id in ("PixelCartPole-v1", "MultiModalCartPole-v1"):
        w = GymWorld(env_id)
        assert isinstance(w, World), f"GymWorld({env_id!r}) does not satisfy World"
        w.close()
        wd = DescribedGymWorld(env_id)
        assert isinstance(wd, World), f"DescribedGymWorld({env_id!r}) does not satisfy World"
        wd.close()
        print(f"  {env_id}: OK")
    print("PASS: pixel CartPole worlds satisfy the protocol")


def test_pixel_cartpole_observation_shape() -> None:
    """PixelCartPole returns (84,84,3) uint8 observations."""
    w = GymWorld("PixelCartPole-v1")
    ts = w.reset()
    w.close()

    assert isinstance(ts.observation, np.ndarray)
    assert ts.observation.shape == (84, 84, 3)
    assert ts.observation.dtype == np.uint8
    print("PASS: pixel CartPole observation shape correct")


def test_multimodal_cartpole_observation_shape() -> None:
    """MultiModalCartPole returns dict with pixels and state."""
    w = GymWorld("MultiModalCartPole-v1")
    ts = w.reset()
    w.close()

    assert isinstance(ts.observation, dict)
    assert "pixels" in ts.observation and "state" in ts.observation
    assert ts.observation["pixels"].shape == (84, 84, 3)
    assert ts.observation["state"].shape == (4,)
    print("PASS: multi-modal CartPole observation shape correct")


def test_pixel_cartpole_described_training() -> None:
    """run_training() works with described pixel CartPole."""
    world = DescribedGymWorld("PixelCartPole-v1")
    rewards = run_training(
        world, num_episodes=3, verbose=False, ollama_model="__nonexistent__"
    )

    assert len(rewards) == 3
    print(f"PASS: run_training(PixelCartPole described) ran {len(rewards)} episodes")


def test_multimodal_cartpole_described_training() -> None:
    """run_training() works with described multi-modal CartPole."""
    world = DescribedGymWorld("MultiModalCartPole-v1")
    rewards = run_training(
        world, num_episodes=3, verbose=False, ollama_model="__nonexistent__"
    )

    assert len(rewards) == 3
    print(f"PASS: run_training(MultiModalCartPole described) ran {len(rewards)} episodes")


def test_multiview_state_adapter() -> None:
    """StateTensorAdapter concatenates multiple views correctly."""
    plan = PerceptionPlan(
        world_description=MULTIMODAL_CARTPOLE_WORLD_DESCRIPTION,
        feature_groups=MULTIMODAL_CARTPOLE_WORLD_DESCRIPTION.feature_hints,
        tensor_views=(
            TensorViewPlan(
                view_id="pixels",
                source_channel="pixels",
                encoder_type="cnn",
                input_shape=(84, 84, 3),
                input_channels=3,
                latent_dim=128,
            ),
            TensorViewPlan(
                view_id="state",
                source_channel="state",
                encoder_type="identity",
                input_shape=(4,),
                input_dim=4,
                latent_dim=4,
            ),
        ),
        default_tensor_view="pixels",
    )
    adapter = StateTensorAdapter(view_names=("pixels", "state"))
    assert adapter.state_dim(plan) == 132

    state = ExampleSubjectiveState(
        agent_observation=cast(Any, None),
        named_fields={},
        tensor_views={
            "pixels": torch.ones(128),
            "state": torch.zeros(4),
        },
        default_tensor_view="pixels",
    )
    tensor = adapter.tensor(state)
    assert tensor.shape == (132,)
    assert tensor[:128].sum() == 128.0
    assert tensor[128:].sum() == 0.0
    print("PASS: multi-view state adapter concatenation correct")


def test_smoke_minimal() -> None:
    """Minimal smoke test still passes."""
    trace = run_minimal_smoke(horizon=5)
    assert len(trace) > 0, "Smoke run produced empty trace"
    assert trace[0]["created_subtasks"] == ["subtask:observation"]
    print("PASS: minimal smoke test")


def test_smoke_fine_grained() -> None:
    """Fine-grained smoke test still passes."""
    trace = run_minimal_fine_grained(horizon=5)
    assert len(trace) > 0, "Fine-grained smoke run produced empty trace"
    print("PASS: fine-grained smoke test")


def test_example_imports() -> None:
    """Example 01 exports its direct public API."""
    assert callable(build_agent)
    assert callable(run_training)
    print("PASS: direct example imports")


def main() -> None:
    tests = [
        test_world_protocol,
        test_build_agent,
        test_grouped_feature_hints,
        test_perception_prioritizes_uncreated_features,
        test_perception_bootstraps_multiple_subtasks_when_budget_allows,
        test_perception_normalizes_raw_tensor_views,
        test_agent_train_embedded,
        test_run_training_embedded,
        test_run_training_discovery,
        test_episode_trace_logger_can_access_world_and_frames,
        test_raw_world_preserves_gym_observation,
        test_dyna_planning_skips_terminal_samples,
        test_run_training_supports_split_identity_llm_views,
        test_llm_plan_prefers_full_raw_state_as_default_tensor_view,
        test_llm_plan_adds_identity_full_raw_control_view,
        test_llm_plan_preserves_curated_raw_feature_groups,
        test_episode_animation_recorder_writes_gif,
        test_training_curve_recorder_writes_svg_and_json,
        test_llm_plan_adds_missing_full_raw_state_view,
        test_build_agent_uses_full_raw_state_for_control,
        test_acrobot_policy_uses_episode_based_epsilon_decay,
        test_acrobot_described_features,
        test_run_training_acrobot_embedded,
        test_run_training_acrobot_discovery,
        test_raw_acrobot_world_preserves_gym_observation,
        test_pixel_cartpole_world_protocol,
        test_pixel_cartpole_observation_shape,
        test_multimodal_cartpole_observation_shape,
        test_pixel_cartpole_described_training,
        test_multimodal_cartpole_described_training,
        test_multiview_state_adapter,
        test_smoke_minimal,
        test_smoke_fine_grained,
        test_example_imports,
    ]

    results: dict[str, bool] = {}
    for test in tests:
        name = test.__name__
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        try:
            test()
            results[name] = True
        except Exception as exc:
            print(f"FAIL: {exc}")
            results[name] = False

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
