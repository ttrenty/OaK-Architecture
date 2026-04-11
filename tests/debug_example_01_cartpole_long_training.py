"""Long-horizon diagnostics for Example 01 CartPole components.

This suite complements ``tests.debug_example_01_cartpole`` by running
longer learning checks on:

- trainable perception / encoder adaptation
- value-function learning
- reactive-policy learning
- transition-model learning
- value + policy together
- value + policy + model together

Artifacts are written under ``tests/results/debug_example_01_cartpole_long/``.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

from examples.example_01.perception import AdaptivePerception
from examples.example_01.reactive_policy import OptionCriticPolicy
from examples.example_01.schema import ExampleSubjectiveState
from examples.example_01.transition_model import DynaTransitionModel
from examples.example_01.value_function import OptionValueFunction
from examples.example_01.world_embedded import CARTPOLE_WORLD_DESCRIPTION
from oak import OaKAgent
from oak.types import SubtaskSpec, Transition
from tests.debug_example_01_cartpole import (
    ACTION_DELTAS,
    ACTION_REWARDS,
    DEFAULT_ENCODER_IMAGE_SIZE,
    DEFAULT_SEED,
    PLOT_COLORS,
    REFERENCE_STATE,
    _build_cartpole_agent,
    _build_trainable_agent,
    _cartpole_observation_to_image,
    _collect_cartpole_observations,
    _device,
    _encoder_parameter_delta,
    _encoder_parameter_snapshot,
    _policy_snapshot,
    _set_seed,
    _state_variant,
    _subjective_state,
    _train_encoder_via_perception,
    _trainable_encoder_for,
    _unwrap_modules,
    _write_json,
    _write_line_plot_svg,
)


DEFAULT_LONG_PERCEPTION_SAMPLES = int(
    os.environ.get("OAK_LONG_PERCEPTION_SAMPLES", "64")
)
DEFAULT_LONG_ENCODER_EPOCHS = int(
    os.environ.get("OAK_LONG_ENCODER_EPOCHS", "12")
)
DEFAULT_LONG_VALUE_UPDATES = int(os.environ.get("OAK_LONG_VALUE_UPDATES", "800"))
DEFAULT_LONG_POLICY_UPDATES = int(os.environ.get("OAK_LONG_POLICY_UPDATES", "800"))
DEFAULT_LONG_MODEL_UPDATES = int(os.environ.get("OAK_LONG_MODEL_UPDATES", "1200"))
DEFAULT_LONG_PAIR_UPDATES = int(os.environ.get("OAK_LONG_PAIR_UPDATES", "900"))
DEFAULT_LONG_TRIPLET_UPDATES = int(
    os.environ.get("OAK_LONG_TRIPLET_UPDATES", "1200")
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "OAK_LONG_OUTPUT_DIR",
        "tests/results/debug_example_01_cartpole_long",
    )
)


def _module_dir(name: str) -> Path:
    return DEFAULT_OUTPUT_DIR / name


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _window_mean(values: list[float], start: int, end: int) -> float:
    return _mean(values[start:end])


def _finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _discrete_action(action: Any) -> int:
    if isinstance(action, bool):
        return int(action)
    if isinstance(action, (int, float)):
        return int(action)
    raise TypeError(f"Expected scalar discrete action, got {type(action)!r}")


def _greedy_option_action_gap(
    reactive_policy: OptionCriticPolicy,
    option_id: str,
    subjective_state: Any,
) -> tuple[float, float, float]:
    q_values, stop_prob = _policy_snapshot(reactive_policy, option_id, subjective_state)
    action0 = float(q_values[0])
    action1 = float(q_values[1])
    return action0, action1, stop_prob


def test_perception_long_horizon_trainable_mlp() -> None:
    """Perception should stay stable and keep improving across longer training."""
    _set_seed(DEFAULT_SEED)
    device = _device()
    observations = _collect_cartpole_observations(
        DEFAULT_SEED,
        DEFAULT_LONG_PERCEPTION_SAMPLES,
    )

    config = dict(CARTPOLE_WORLD_DESCRIPTION.to_config())
    config["encoder_type"] = "mlp"
    config["latent_dim"] = 16
    agent = _build_trainable_agent(config, train_encoder=True, device=device)
    perception, _, _, _ = _unwrap_modules(agent)

    if not isinstance(perception, AdaptivePerception):
        raise TypeError(f"Unexpected perception type: {type(perception)!r}")

    encoder = _trainable_encoder_for(perception)
    before = _encoder_parameter_snapshot(encoder)
    latent, loss_history = _train_encoder_via_perception(
        perception,
        observations,
        epochs=DEFAULT_LONG_ENCODER_EPOCHS,
    )
    parameter_delta = _encoder_parameter_delta(before, encoder)

    module_dir = _module_dir("perception_mlp")
    _write_json(
        module_dir / "summary.json",
        {
            "epochs": DEFAULT_LONG_ENCODER_EPOCHS,
            "sample_count": len(observations),
            "initial_loss": loss_history[0],
            "final_loss": loss_history[-1],
            "parameter_delta": parameter_delta,
            "latent_shape": list(latent.shape),
            "latent_norm": float(torch.norm(latent).item()),
            "features": [feature.feature_id for feature in perception.list_features()],
        },
    )
    _write_line_plot_svg(
        module_dir / "loss.svg",
        title="Long-Horizon Perception MLP Loss",
        x_label="Epoch",
        y_label="Mean MSE",
        series=[("mlp_loss", loss_history, PLOT_COLORS[0])],
    )

    assert perception._train_encoder is True
    assert tuple(latent.shape) == (16,)
    assert _finite_tensor(latent)
    assert parameter_delta > 0.0
    assert loss_history[-1] < loss_history[0] * 0.4

    print(f"PASS: long-horizon perception diagnostics saved to {module_dir}")


def test_value_function_long_horizon() -> None:
    """ValueFunction should continue improving over a longer replay horizon."""
    _set_seed(DEFAULT_SEED)
    agent = _build_cartpole_agent(device=_device())
    _, _, value_function, _ = _unwrap_modules(agent)

    option_id = "option:long_value"
    gvf_id = "long_value"
    option_idx = value_function.register_option(option_id)
    value_function.add_gvf(gvf_id)

    q_history: list[float] = []
    gvf_history: list[float] = []
    advantage_history: list[float] = []

    initial_predictions = value_function.predict(_subjective_state(REFERENCE_STATE))

    for step in range(DEFAULT_LONG_VALUE_UPDATES):
        state = _state_variant(step)
        next_state = _state_variant(step + 1)
        value_function.set_last_option_idx(option_idx)
        td_errors = value_function.update(
            Transition(
                subjective_state=_subjective_state(state),
                action=1,
                reward=1.0,
                next_subjective_state=_subjective_state(next_state),
                terminated=((step + 1) % 40 == 0),
                option_id=option_id,
            )
        )

        predictions = value_function.predict(_subjective_state(REFERENCE_STATE))
        q_history.append(float(predictions[f"q_{option_id}"]))
        gvf_history.append(float(predictions[f"gvf_{gvf_id}"]))
        advantage_history.append(float(td_errors["advantage"]))

    early_q = _window_mean(q_history, 0, 100)
    late_q = _window_mean(q_history, len(q_history) - 100, len(q_history))
    early_gvf = _window_mean(gvf_history, 0, 100)
    late_gvf = _window_mean(gvf_history, len(gvf_history) - 100, len(gvf_history))

    module_dir = _module_dir("value_function")
    _write_json(
        module_dir / "summary.json",
        {
            "updates": DEFAULT_LONG_VALUE_UPDATES,
            "initial_q": float(initial_predictions[f"q_{option_id}"]),
            "final_q": q_history[-1],
            "early_q_mean": early_q,
            "late_q_mean": late_q,
            "initial_gvf": float(initial_predictions[f"gvf_{gvf_id}"]),
            "final_gvf": gvf_history[-1],
            "early_gvf_mean": early_gvf,
            "late_gvf_mean": late_gvf,
        },
    )
    _write_line_plot_svg(
        module_dir / "predictions.svg",
        title="Long-Horizon Value Learning",
        x_label="Update",
        y_label="Prediction",
        series=[
            ("q_omega", q_history, PLOT_COLORS[0]),
            ("gvf", gvf_history, PLOT_COLORS[1]),
            ("advantage", advantage_history, PLOT_COLORS[4]),
        ],
    )

    assert len(value_function._buffer) == DEFAULT_LONG_VALUE_UPDATES
    assert late_q > early_q + 0.4
    assert late_gvf > early_gvf + 0.1
    assert q_history[-1] > float(initial_predictions[f"q_{option_id}"]) + 0.5

    print(f"PASS: long-horizon value-function diagnostics saved to {module_dir}")


def test_reactive_policy_long_horizon() -> None:
    """ReactivePolicy should prefer the better primitive action over time."""
    _set_seed(DEFAULT_SEED)
    agent = _build_cartpole_agent(device=_device())
    _, reactive_policy, _, _ = _unwrap_modules(agent)

    subtask = SubtaskSpec(
        subtask_id="long_policy",
        name="Long policy option",
        feature_id="pole_angle",
    )
    reactive_policy.ingest_subtasks([subtask])
    option_id = "option:long_policy"
    reactive_policy._epsilon_decay_steps = max(DEFAULT_LONG_POLICY_UPDATES // 2, 1)

    reward_history: list[float] = []
    action_gap_history: list[float] = []
    stop_history: list[float] = []

    for step in range(DEFAULT_LONG_POLICY_UPDATES):
        subjective_state = _subjective_state(_state_variant(step))
        action, active_option_id = reactive_policy.select_action(
            subjective_state,
            option_stop_threshold=0.5,
        )
        assert active_option_id == option_id

        action_id = _discrete_action(action)
        reward = ACTION_REWARDS[action_id]
        transition: Transition[Any, ExampleSubjectiveState, dict[str, Any]] = Transition(
            subjective_state=subjective_state,
            action=action_id,
            reward=reward,
            next_subjective_state=_subjective_state(_state_variant(step + 1)),
            terminated=False,
            option_id=option_id,
        )
        reactive_policy.update(transition, {"advantage": reward})

        action0, action1, stop_prob = _greedy_option_action_gap(
            reactive_policy,
            option_id,
            _subjective_state(REFERENCE_STATE),
        )
        reward_history.append(reward)
        action_gap_history.append(action1 - action0)
        stop_history.append(stop_prob)

    early_reward = _window_mean(reward_history, 0, 100)
    late_reward = _window_mean(
        reward_history,
        len(reward_history) - 100,
        len(reward_history),
    )
    early_gap = _window_mean(action_gap_history, 0, 100)
    late_gap = _window_mean(
        action_gap_history,
        len(action_gap_history) - 100,
        len(action_gap_history),
    )
    final_action0, final_action1, final_stop = _greedy_option_action_gap(
        reactive_policy,
        option_id,
        _subjective_state(REFERENCE_STATE),
    )

    module_dir = _module_dir("reactive_policy")
    _write_json(
        module_dir / "summary.json",
        {
            "updates": DEFAULT_LONG_POLICY_UPDATES,
            "early_reward_mean": early_reward,
            "late_reward_mean": late_reward,
            "early_action_gap_mean": early_gap,
            "late_action_gap_mean": late_gap,
            "final_action0_q": final_action0,
            "final_action1_q": final_action1,
            "final_stop_prob": final_stop,
        },
    )
    _write_line_plot_svg(
        module_dir / "learning.svg",
        title="Long-Horizon Reactive Policy Learning",
        x_label="Update",
        y_label="Value",
        series=[
            ("reward", reward_history, PLOT_COLORS[2]),
            ("action_gap", action_gap_history, PLOT_COLORS[1]),
            ("stop_prob", stop_history, PLOT_COLORS[3]),
        ],
    )

    assert len(reactive_policy._buffer) == DEFAULT_LONG_POLICY_UPDATES
    assert late_reward > early_reward + 0.2
    assert late_gap > early_gap + 0.2
    assert final_action1 > final_action0 + 0.2
    assert 0.0 <= final_stop <= 1.0

    print(f"PASS: long-horizon reactive-policy diagnostics saved to {module_dir}")


def test_transition_model_long_horizon() -> None:
    """TransitionModel should fit longer dynamics and classify terminal outcomes."""
    _set_seed(DEFAULT_SEED)
    agent = _build_cartpole_agent(
        device=_device(),
        planning_budget=5,
        planning_warmup_steps=1,
    )
    _, _, value_function, transition_model = _unwrap_modules(agent)

    option_id = "option:long_model"
    option_idx = value_function.register_option(option_id)
    value_function.set_last_option_idx(option_idx)

    loss_history: list[float] = []
    for step in range(DEFAULT_LONG_MODEL_UPDATES):
        action = step % 2
        state = REFERENCE_STATE.clone()
        next_state = state + ACTION_DELTAS[action]
        terminated = action == 1
        transition_model.update(
            Transition(
                subjective_state=_subjective_state(state),
                action=action,
                reward=ACTION_REWARDS[action],
                next_subjective_state=_subjective_state(next_state),
                terminated=terminated,
                option_id=option_id,
            )
        )
        loss_history.append(float(transition_model._model_loss))

    delta_errors: list[float] = []
    reward_errors: list[float] = []
    done_probs: list[float] = []
    device_state = REFERENCE_STATE.to(transition_model._device)
    for action in (0, 1):
        with torch.no_grad():
            delta_pred, reward_pred, done_logit = transition_model._model(
                device_state,
                torch.tensor(action, device=transition_model._device),
            )
        delta_target = ACTION_DELTAS[action].to(transition_model._device)
        delta_errors.append(
            float(torch.mean((delta_pred.squeeze(0) - delta_target) ** 2).item())
        )
        reward_errors.append(
            float((reward_pred.squeeze() - ACTION_REWARDS[action]) ** 2)
        )
        done_probs.append(float(torch.sigmoid(done_logit).item()))

    q_before = float(
        value_function.predict(_subjective_state(REFERENCE_STATE))[f"q_{option_id}"]
    )
    planning = transition_model.plan(
        _subjective_state(REFERENCE_STATE),
        value_function,
        budget=5,
    )
    q_after = float(
        value_function.predict(_subjective_state(REFERENCE_STATE))[f"q_{option_id}"]
    )

    module_dir = _module_dir("transition_model")
    _write_json(
        module_dir / "summary.json",
        {
            "updates": DEFAULT_LONG_MODEL_UPDATES,
            "initial_loss": loss_history[0],
            "final_loss": loss_history[-1],
            "delta_mse": delta_errors,
            "reward_mse": reward_errors,
            "done_probabilities": done_probs,
            "planning_steps": planning.search_statistics.get("planning_steps", 0),
            "q_before_planning": q_before,
            "q_after_planning": q_after,
        },
    )
    _write_line_plot_svg(
        module_dir / "loss.svg",
        title="Long-Horizon Transition Model Loss",
        x_label="Update",
        y_label="Loss",
        series=[("model_loss", loss_history, PLOT_COLORS[4])],
    )

    assert len(transition_model._buffer) == DEFAULT_LONG_MODEL_UPDATES
    assert loss_history[-1] < _window_mean(loss_history, 0, 100)
    assert max(delta_errors) < 1e-3
    assert max(reward_errors) < 1e-3
    assert done_probs[0] < 0.1
    assert done_probs[1] > 0.9
    assert planning.search_statistics.get("planning_steps", 0) == 5
    assert abs(q_after - q_before) > 1e-6

    print(f"PASS: long-horizon transition-model diagnostics saved to {module_dir}")


def test_value_policy_pair_long_horizon() -> None:
    """ValueFunction and ReactivePolicy should improve together."""
    _set_seed(DEFAULT_SEED)
    agent = _build_cartpole_agent(device=_device())
    _, reactive_policy, value_function, _ = _unwrap_modules(agent)

    subtask = SubtaskSpec(
        subtask_id="long_pair",
        name="Long pair option",
        feature_id="pole_balance",
    )
    reactive_policy.ingest_subtasks([subtask])
    option_id = "option:long_pair"
    reactive_policy._epsilon_decay_steps = max(DEFAULT_LONG_PAIR_UPDATES // 3, 1)

    reward_history: list[float] = []
    option_q_history: list[float] = []
    action_gap_history: list[float] = []

    for step in range(DEFAULT_LONG_PAIR_UPDATES):
        subjective_state = _subjective_state(_state_variant(step))
        action, active_option_id = reactive_policy.select_action(
            subjective_state,
            option_stop_threshold=0.5,
        )
        assert active_option_id == option_id

        action_id = _discrete_action(action)
        reward = ACTION_REWARDS[action_id]
        transition: Transition[Any, ExampleSubjectiveState, dict[str, Any]] = Transition(
            subjective_state=subjective_state,
            action=action_id,
            reward=reward,
            next_subjective_state=_subjective_state(_state_variant(step + 1)),
            terminated=((step + 1) % 50 == 0),
            option_id=option_id,
        )
        td_errors = value_function.update(transition)
        reactive_policy.update(transition, td_errors)

        predictions = value_function.predict(_subjective_state(REFERENCE_STATE))
        action0, action1, _ = _greedy_option_action_gap(
            reactive_policy,
            option_id,
            _subjective_state(REFERENCE_STATE),
        )
        reward_history.append(reward)
        option_q_history.append(float(predictions[f"q_{option_id}"]))
        action_gap_history.append(action1 - action0)

    early_reward = _window_mean(reward_history, 0, 100)
    late_reward = _window_mean(
        reward_history,
        len(reward_history) - 100,
        len(reward_history),
    )
    early_option_q = _window_mean(option_q_history, 0, 100)
    late_option_q = _window_mean(
        option_q_history,
        len(option_q_history) - 100,
        len(option_q_history),
    )
    final_action0, final_action1, _ = _greedy_option_action_gap(
        reactive_policy,
        option_id,
        _subjective_state(REFERENCE_STATE),
    )

    module_dir = _module_dir("value_policy_pair")
    _write_json(
        module_dir / "summary.json",
        {
            "updates": DEFAULT_LONG_PAIR_UPDATES,
            "early_reward_mean": early_reward,
            "late_reward_mean": late_reward,
            "early_option_q_mean": early_option_q,
            "late_option_q_mean": late_option_q,
            "final_action0_q": final_action0,
            "final_action1_q": final_action1,
        },
    )
    _write_line_plot_svg(
        module_dir / "pair_learning.svg",
        title="Long-Horizon Value + Policy Learning",
        x_label="Update",
        y_label="Value",
        series=[
            ("reward", reward_history, PLOT_COLORS[2]),
            ("option_q", option_q_history, PLOT_COLORS[0]),
            ("action_gap", action_gap_history, PLOT_COLORS[1]),
        ],
    )

    assert late_reward > early_reward + 0.15
    assert late_option_q > early_option_q + 0.2
    assert final_action1 > final_action0 + 0.2

    print(f"PASS: long-horizon value+policy diagnostics saved to {module_dir}")


def test_value_policy_model_triplet_long_horizon() -> None:
    """ValueFunction, ReactivePolicy, and TransitionModel should co-adapt."""
    _set_seed(DEFAULT_SEED)
    agent = _build_cartpole_agent(
        device=_device(),
        planning_budget=1,
        planning_warmup_steps=64,
    )
    _, reactive_policy, value_function, transition_model = _unwrap_modules(agent)

    subtask = SubtaskSpec(
        subtask_id="long_triplet",
        name="Long triplet option",
        feature_id="cart_motion",
    )
    reactive_policy.ingest_subtasks([subtask])
    option_id = "option:long_triplet"
    reactive_policy._epsilon_decay_steps = max(DEFAULT_LONG_TRIPLET_UPDATES // 3, 1)

    reward_history: list[float] = []
    option_q_history: list[float] = []
    model_loss_history: list[float] = []
    planning_steps_history: list[int] = []
    action_gap_history: list[float] = []

    for step in range(DEFAULT_LONG_TRIPLET_UPDATES):
        subjective_state = _subjective_state(_state_variant(step))
        action, active_option_id = reactive_policy.select_action(
            subjective_state,
            option_stop_threshold=0.5,
        )
        assert active_option_id == option_id

        action_id = _discrete_action(action)
        next_tensor = _state_variant(step) + ACTION_DELTAS[action_id]
        reward = ACTION_REWARDS[action_id]
        transition: Transition[Any, ExampleSubjectiveState, dict[str, Any]] = Transition(
            subjective_state=subjective_state,
            action=action_id,
            reward=reward,
            next_subjective_state=_subjective_state(next_tensor),
            terminated=False,
            option_id=option_id,
        )
        td_errors = value_function.update(transition)
        reactive_policy.update(transition, td_errors)
        transition_model.update(transition)
        planning = transition_model.plan(subjective_state, value_function, budget=1)

        predictions = value_function.predict(_subjective_state(REFERENCE_STATE))
        action0, action1, _ = _greedy_option_action_gap(
            reactive_policy,
            option_id,
            _subjective_state(REFERENCE_STATE),
        )
        reward_history.append(reward)
        option_q_history.append(float(predictions[f"q_{option_id}"]))
        model_loss_history.append(float(transition_model._model_loss))
        planning_steps_history.append(
            _discrete_action(planning.search_statistics.get("planning_steps", 0))
        )
        action_gap_history.append(action1 - action0)

    early_reward = _window_mean(reward_history, 0, 100)
    late_reward = _window_mean(
        reward_history,
        len(reward_history) - 100,
        len(reward_history),
    )
    early_option_q = _window_mean(option_q_history, 0, 100)
    late_option_q = _window_mean(
        option_q_history,
        len(option_q_history) - 100,
        len(option_q_history),
    )
    final_action0, final_action1, _ = _greedy_option_action_gap(
        reactive_policy,
        option_id,
        _subjective_state(REFERENCE_STATE),
    )

    module_dir = _module_dir("value_policy_model_triplet")
    _write_json(
        module_dir / "summary.json",
        {
            "updates": DEFAULT_LONG_TRIPLET_UPDATES,
            "early_reward_mean": early_reward,
            "late_reward_mean": late_reward,
            "early_option_q_mean": early_option_q,
            "late_option_q_mean": late_option_q,
            "initial_model_loss": model_loss_history[0],
            "final_model_loss": model_loss_history[-1],
            "planning_steps_total": sum(planning_steps_history),
            "final_action0_q": final_action0,
            "final_action1_q": final_action1,
        },
    )
    _write_line_plot_svg(
        module_dir / "triplet_learning.svg",
        title="Long-Horizon Value + Policy + Model Learning",
        x_label="Update",
        y_label="Value",
        series=[
            ("reward", reward_history, PLOT_COLORS[2]),
            ("option_q", option_q_history, PLOT_COLORS[0]),
            ("model_loss", model_loss_history, PLOT_COLORS[4]),
            ("action_gap", action_gap_history, PLOT_COLORS[1]),
        ],
    )

    assert sum(planning_steps_history) > 0
    assert late_reward > early_reward + 0.15
    assert late_option_q > early_option_q + 0.25
    assert model_loss_history[-1] < _window_mean(model_loss_history, 0, 100)
    assert final_action1 > final_action0 + 0.2

    print(f"PASS: long-horizon triplet diagnostics saved to {module_dir}")


def main() -> None:
    tests = [
        test_perception_long_horizon_trainable_mlp,
        test_value_function_long_horizon,
        test_reactive_policy_long_horizon,
        test_transition_model_long_horizon,
        test_value_policy_pair_long_horizon,
        test_value_policy_model_triplet_long_horizon,
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
            message = str(exc) or exc.__class__.__name__
            print(f"FAIL: {message}")
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
