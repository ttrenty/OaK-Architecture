"""Module-isolated diagnostics for Example 01 on CartPole.

This file does not train the full agent end to end. Instead, it checks that
each real Example 01 module can work on its own when given sensible inputs:

- ``AdaptivePerception`` on real CartPole observations
- ``OptionValueFunction`` on fixed subjective-state transitions
- ``OptionCriticPolicy`` on fixed option-learning transitions
- ``DynaTransitionModel`` on fixed latent dynamics plus a small planning handoff

Each test saves JSON and SVG artifacts under
``results/debug_example_01_cartpole/<module>/``.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from html import escape
from pathlib import Path

import numpy as np
import torch
from typing import Any, cast

from examples.example_01.encoders import CNNEncoder, MLPEncoder
from examples.example_01.perception import AdaptivePerception
from examples.example_01.reactive_policy import OptionCriticPolicy
from examples.example_01.runner import build_agent
from examples.example_01.schema import ExampleSubjectiveState, subjective_state_from_tensor
from examples.example_01.transition_model import DynaTransitionModel
from examples.example_01.value_function import OptionValueFunction
from examples.example_01.world_embedded import (
    CARTPOLE_WORLD_DESCRIPTION,
    DescribedGymWorld,
)
from oak import OaKAgent
from oak.types import (
    ComponentKind,
    SubtaskSpec,
    Transition,
    UsageRecord,
    UtilityRecord,
)

DEFAULT_SEED = int(os.environ.get("OAK_DEBUG_SEED", "7"))
DEFAULT_DEVICE_NAME = os.environ.get("OAK_DEBUG_DEVICE", "cpu")
DEFAULT_PERCEPTION_SAMPLES = int(os.environ.get("OAK_DEBUG_PERCEPTION_SAMPLES", "48"))
DEFAULT_ENCODER_TRAIN_SAMPLES = int(
    os.environ.get("OAK_DEBUG_ENCODER_TRAIN_SAMPLES", "24")
)
DEFAULT_ENCODER_TRAIN_EPOCHS = int(
    os.environ.get("OAK_DEBUG_ENCODER_TRAIN_EPOCHS", "6")
)
DEFAULT_ENCODER_IMAGE_SIZE = int(os.environ.get("OAK_DEBUG_ENCODER_IMAGE_SIZE", "64"))
DEFAULT_VALUE_UPDATES = int(os.environ.get("OAK_DEBUG_VALUE_UPDATES", "160"))
DEFAULT_POLICY_UPDATES = int(os.environ.get("OAK_DEBUG_POLICY_UPDATES", "160"))
DEFAULT_MODEL_UPDATES = int(os.environ.get("OAK_DEBUG_MODEL_UPDATES", "560"))
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("OAK_DEBUG_OUTPUT_DIR", "tests/results/debug_example_01_cartpole")
)
PLOT_COLORS = (
    "#264653",
    "#2a9d8f",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#457b9d",
)
REFERENCE_STATE = torch.tensor([0.2, -0.1, 0.05, 0.15], dtype=torch.float32)
ACTION_DELTAS = {
    0: torch.tensor([0.05, -0.02, 0.00, 0.01], dtype=torch.float32),
    1: torch.tensor([-0.04, 0.03, 0.02, -0.01], dtype=torch.float32),
}
ACTION_REWARDS = {
    0: 0.25,
    1: 1.0,
}


def _device() -> torch.device:
    requested = DEFAULT_DEVICE_NAME.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _module_dir(name: str) -> Path:
    return DEFAULT_OUTPUT_DIR / name


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_line_plot_svg(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[float], str]],
) -> None:
    width = 960
    height = 540
    left = 70
    right = 24
    top = 52
    bottom = 58
    plot_width = width - left - right
    plot_height = height - top - bottom

    values = [
        float(value)
        for _, seq, _ in series
        for value in seq
        if math.isfinite(float(value))
    ]
    if not values:
        values = [0.0, 1.0]

    y_min = min(values)
    y_max = max(values)
    if math.isclose(y_min, y_max):
        pad = max(1.0, abs(y_min) * 0.1 + 1.0)
        y_min -= pad
        y_max += pad

    x_max = max((len(seq) - 1 for _, seq, _ in series if seq), default=1)
    x_max = max(x_max, 1)

    def x_pos(index: int) -> float:
        return left + (index / x_max) * plot_width

    def y_pos(value: float) -> float:
        norm = (value - y_min) / (y_max - y_min)
        return top + (1.0 - norm) * plot_height

    grid_lines: list[str] = []
    tick_labels: list[str] = []
    for tick in range(6):
        frac = tick / 5
        y_value = y_max - frac * (y_max - y_min)
        y = top + frac * plot_height
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#d7dde5" stroke-width="1" />'
        )
        tick_labels.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            'font-size="12" fill="#334155">'
            f"{y_value:.2f}</text>"
        )

    polylines: list[str] = []
    legend_items: list[str] = []
    for index, (label, seq, color) in enumerate(series):
        points = " ".join(
            f"{x_pos(i):.2f},{y_pos(float(value)):.2f}"
            for i, value in enumerate(seq)
            if math.isfinite(float(value))
        )
        if points:
            polylines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
                f'stroke-linejoin="round" stroke-linecap="round" points="{points}" />'
            )
        legend_y = top + 16 + 20 * index
        legend_items.append(
            f'<line x1="{width - 230}" y1="{legend_y:.2f}" '
            f'x2="{width - 210}" y2="{legend_y:.2f}" '
            f'stroke="{color}" stroke-width="3" />'
            f'<text x="{width - 202}" y="{legend_y + 4:.2f}" font-size="12" '
            f'fill="#0f172a">{escape(label)}</text>'
        )

    x_ticks: list[str] = []
    for tick in range(6):
        index = round((tick / 5) * x_max)
        x = x_pos(index)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{height - bottom}" x2="{x:.2f}" '
            f'y2="{height - bottom + 6}" stroke="#475569" stroke-width="1" />'
            f'<text x="{x:.2f}" y="{height - bottom + 22}" text-anchor="middle" '
            f'font-size="12" fill="#334155">{index}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc" />
  <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="10" />
  {''.join(grid_lines)}
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#475569" stroke-width="1.5" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#475569" stroke-width="1.5" />
  {''.join(polylines)}
  {''.join(tick_labels)}
  {''.join(x_ticks)}
  <text x="{width / 2:.2f}" y="28" text-anchor="middle" font-size="20" fill="#0f172a">{escape(title)}</text>
  <text x="{width / 2:.2f}" y="{height - 16}" text-anchor="middle" font-size="13" fill="#334155">{escape(x_label)}</text>
  <text x="20" y="{height / 2:.2f}" text-anchor="middle" font-size="13" fill="#334155" transform="rotate(-90 20 {height / 2:.2f})">{escape(y_label)}</text>
  {''.join(legend_items)}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def _make_world(seed: int) -> DescribedGymWorld:
    world: DescribedGymWorld[Any, Any] = DescribedGymWorld("CartPole-v1")
    world.env.reset(seed=seed)
    if hasattr(world.env.action_space, "seed"):
        world.env.action_space.seed(seed)
    if hasattr(world.env.observation_space, "seed"):
        world.env.observation_space.seed(seed)
    return world


def _build_cartpole_agent(device: torch.device | None = None) -> OaKAgent:
    return build_agent(CARTPOLE_WORLD_DESCRIPTION.to_config(), device=device)


def _build_trainable_agent(
    config: dict[str, object],
    *,
    train_encoder: bool,
    device: torch.device | None = None,
) -> OaKAgent:
    return build_agent(config, train_encoder=train_encoder, device=device)


def _unwrap_modules(
    agent: OaKAgent,
) -> tuple[
    AdaptivePerception,
    OptionCriticPolicy,
    OptionValueFunction,
    DynaTransitionModel,
]:
    perception = agent.perception
    reactive_policy = agent.reactive_policy
    value_function = agent.value_function
    transition_model = agent.transition_model

    if not isinstance(perception, AdaptivePerception):
        raise TypeError(f"Unexpected perception type: {type(perception)!r}")
    if not isinstance(reactive_policy, OptionCriticPolicy):
        raise TypeError(
            f"Unexpected reactive_policy type: {type(reactive_policy)!r}"
        )
    if not isinstance(value_function, OptionValueFunction):
        raise TypeError(f"Unexpected value_function type: {type(value_function)!r}")
    if not isinstance(transition_model, DynaTransitionModel):
        raise TypeError(
            f"Unexpected transition_model type: {type(transition_model)!r}"
        )

    return perception, reactive_policy, value_function, transition_model


def _collect_cartpole_observations(seed: int, sample_count: int) -> list[np.ndarray]:
    world = _make_world(seed)
    observations: list[np.ndarray] = []

    try:
        time_step = world.reset()
        next_action = 0
        while len(observations) < sample_count:
            obs = np.asarray(time_step.observation, dtype=np.float32).copy()
            observations.append(obs)

            if time_step.terminated or time_step.truncated:
                time_step = world.reset()
                continue

            time_step = world.step(next_action)
            next_action = 1 - next_action
    finally:
        world.close()

    return observations


def _cartpole_observation_to_image(
    observation: np.ndarray,
    *,
    image_size: int = DEFAULT_ENCODER_IMAGE_SIZE,
) -> np.ndarray:
    values = np.asarray(observation, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, image_size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, image_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    cart_position, cart_velocity, pole_angle, pole_velocity = values.tolist()
    channel0 = 0.5 + 0.5 * np.tanh(3.0 * (cart_position * xx + pole_angle * yy))
    channel1 = 0.5 + 0.5 * np.tanh(3.0 * (cart_velocity * yy - pole_velocity * xx))
    wave = (1.5 + cart_position + pole_angle) * xx + (
        1.5 + cart_velocity + pole_velocity
    ) * yy
    channel2 = 0.5 + 0.5 * np.sin(4.0 * wave)
    image = np.stack([channel0, channel1, channel2], axis=-1)
    return image.astype(np.float32)


def _encoder_parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in module.named_parameters()
    }


def _encoder_parameter_delta(
    before: dict[str, torch.Tensor], module: torch.nn.Module
) -> float:
    delta = 0.0
    current = dict(module.named_parameters())
    for name, previous in before.items():
        delta += float(torch.norm(current[name].detach().cpu() - previous).item())
    return delta


def _trainable_encoder_for(
    perception: AdaptivePerception,
) -> MLPEncoder | CNNEncoder:
    encoder = perception._encoder
    if isinstance(encoder, (MLPEncoder, CNNEncoder)):
        return encoder
    raise TypeError(f"Expected trainable MLP/CNN encoder, got {type(encoder)!r}")


def _mean_reconstruction_loss(
    encoder: MLPEncoder | CNNEncoder,
    observations: list[np.ndarray],
) -> float:
    losses: list[float] = []
    if isinstance(encoder, MLPEncoder):
        if encoder.decoder is None:
            raise AssertionError("MLP encoder decoder was not initialised")
        with torch.no_grad():
            for observation in observations:
                x = encoder._to_tensor(observation)
                z = encoder.encoder(x)
                x_hat = encoder.decoder(z)
                losses.append(float(torch.nn.functional.mse_loss(x_hat, x).item()))
        return sum(losses) / len(losses)

    encoder.encode(observations[0])
    if encoder._decoder is None:
        raise AssertionError("CNN encoder decoder was not initialised")
    with torch.no_grad():
        for observation in observations:
            x = encoder._to_image_tensor(observation)
            z = encoder._encode_batch(x)
            x_hat = encoder._decoder(z)
            if x_hat.shape[-2:] != x.shape[-2:]:
                x_hat = torch.nn.functional.interpolate(
                    x_hat,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            losses.append(float(torch.nn.functional.mse_loss(x_hat, x).item()))
    return sum(losses) / len(losses)


def _train_encoder_via_perception(
    perception: AdaptivePerception,
    observations: list[np.ndarray],
    *,
    epochs: int,
) -> tuple[torch.Tensor, list[float]]:
    encoder = _trainable_encoder_for(perception)
    loss_history = [_mean_reconstruction_loss(encoder, observations)]
    last_latent = torch.zeros(0)

    for epoch in range(epochs):
        for index, observation in enumerate(observations):
            last_action = (index + epoch) % 2
            state = perception.update(
                observation=observation,
                reward=float(last_action),
                last_action=last_action,
            )
            last_latent = state.tensor_view()
        loss_history.append(_mean_reconstruction_loss(encoder, observations))

    return last_latent, loss_history


def _state_variant(step: int) -> torch.Tensor:
    offset = torch.tensor(
        [
            0.03 * math.sin(step / 7.0),
            0.02 * math.cos(step / 5.0),
            0.015 * math.sin(step / 11.0),
            0.01 * math.cos(step / 13.0),
        ],
        dtype=torch.float32,
    )
    return REFERENCE_STATE + offset


def _subjective_state(tensor: torch.Tensor) -> ExampleSubjectiveState:
    return subjective_state_from_tensor(tensor, view_name="main")


def _policy_snapshot(
    policy: OptionCriticPolicy,
    option_id: str,
    subjective_state: ExampleSubjectiveState,
) -> tuple[list[float], float]:
    nets, _, _ = policy._options[option_id]
    device_state = policy._state_adapter.tensor(subjective_state).to(policy._device)
    with torch.no_grad():
        q_values = nets.q_values(device_state).detach().cpu().tolist()
    return [float(v) for v in q_values], float(nets.stop_prob(device_state))


def test_build_agent_components() -> None:
    """build_agent() should still construct the real CartPole modules."""
    agent = _build_cartpole_agent(device=_device())
    perception, reactive_policy, value_function, transition_model = _unwrap_modules(
        agent
    )

    assert isinstance(agent, OaKAgent)
    assert len(perception.list_features()) == 2
    assert reactive_policy._num_actions == 2
    assert value_function._state_dim == 4
    assert transition_model._state_dim == 4

    print("PASS: build_agent() returns the real Example 01 modules")


def test_perception_module() -> None:
    """Perception should encode observations, rank features, and emit subtasks."""
    _set_seed(DEFAULT_SEED)
    agent = _build_cartpole_agent(device=_device())
    perception, _, _, _ = _unwrap_modules(agent)

    observations = _collect_cartpole_observations(
        DEFAULT_SEED, DEFAULT_PERCEPTION_SAMPLES
    )
    latent_series: list[list[float]] = [[], [], [], []]
    last_latent: torch.Tensor | None = None
    last_state: ExampleSubjectiveState | None = None

    for index, observation in enumerate(observations):
        last_state = perception.update(
            observation=observation,
            reward=float(index % 2),
            last_action=(index % 2) if index else None,
        )
        last_latent = last_state.tensor_view()
        latent_values = [float(v) for v in last_latent.detach().cpu().tolist()]
        for dim, value in enumerate(latent_values):
            latent_series[dim].append(value)

    if last_latent is None or last_state is None:
        raise AssertionError("No observations were collected")

    utility_scores = (
        UtilityRecord(ComponentKind.FEATURE, "pole_balance", 4.0),
        UtilityRecord(ComponentKind.FEATURE, "cart_motion", 2.0),
    )
    ranked = list(
        perception.discover_and_rank_features(
            perception.current_subjective_state(),
            utility_scores,
            feature_budget=2,
        )
    )
    first_subtasks = list(perception.generate_subtasks(ranked))
    second_subtasks = list(perception.generate_subtasks(ranked))

    module_dir = _module_dir("perception")
    _write_json(
        module_dir / "summary.json",
        {
            "sample_count": len(observations),
            "latent_shape": list(last_latent.shape),
            "named_fields": sorted(last_state.named_fields),
            "ranked_features": ranked,
            "created_subtasks": [subtask.subtask_id for subtask in first_subtasks],
            "repeated_subtasks": [subtask.subtask_id for subtask in second_subtasks],
        },
    )
    _write_line_plot_svg(
        module_dir / "latent_dimensions.svg",
        title="Perception Latent Dimensions",
        x_label="Observation Index",
        y_label="Latent Value",
        series=[
            ("dim_0", latent_series[0], PLOT_COLORS[0]),
            ("dim_1", latent_series[1], PLOT_COLORS[1]),
            ("dim_2", latent_series[2], PLOT_COLORS[2]),
            ("dim_3", latent_series[3], PLOT_COLORS[3]),
        ],
    )

    assert tuple(last_latent.shape) == (4,)
    assert torch.allclose(perception.current_subjective_state().tensor_view(), last_latent)
    assert ranked == ["pole_balance", "cart_motion"]
    assert [subtask.subtask_id for subtask in first_subtasks] == [
        "subtask:pole_balance",
        "subtask:cart_motion",
    ]
    assert second_subtasks == []

    print(f"PASS: perception diagnostics saved to {module_dir}")


def test_trainable_encoders_learn_reconstruction() -> None:
    """train_encoder=True should update MLP/CNN encoder weights and lower loss."""
    _set_seed(DEFAULT_SEED)
    device = _device()
    base_observations = _collect_cartpole_observations(
        DEFAULT_SEED, DEFAULT_ENCODER_TRAIN_SAMPLES
    )

    mlp_config = dict(CARTPOLE_WORLD_DESCRIPTION.to_config())
    mlp_config["encoder_type"] = "mlp"
    mlp_config["latent_dim"] = 16
    mlp_agent = _build_trainable_agent(mlp_config, train_encoder=True, device=device)
    mlp_perception, _, _, _ = _unwrap_modules(mlp_agent)
    mlp_encoder = _trainable_encoder_for(mlp_perception)
    mlp_before = _encoder_parameter_snapshot(mlp_encoder)
    mlp_latent, mlp_loss_history = _train_encoder_via_perception(
        mlp_perception,
        base_observations,
        epochs=DEFAULT_ENCODER_TRAIN_EPOCHS,
    )
    mlp_delta = _encoder_parameter_delta(mlp_before, mlp_encoder)

    cnn_config: dict[str, object] = {
        "obs_shape": (DEFAULT_ENCODER_IMAGE_SIZE, DEFAULT_ENCODER_IMAGE_SIZE, 3),
        "action_n": 2,
        "encoder_type": "cnn",
        "latent_dim": 32,
        "features": [],
    }
    cnn_agent = _build_trainable_agent(cnn_config, train_encoder=True, device=device)
    cnn_perception, _, _, _ = _unwrap_modules(cnn_agent)
    cnn_encoder = _trainable_encoder_for(cnn_perception)
    image_observations = [
        _cartpole_observation_to_image(observation) for observation in base_observations
    ]
    cnn_encoder.encode(image_observations[0])
    cnn_before = _encoder_parameter_snapshot(cnn_encoder)
    cnn_latent, cnn_loss_history = _train_encoder_via_perception(
        cnn_perception,
        image_observations,
        epochs=DEFAULT_ENCODER_TRAIN_EPOCHS,
    )
    cnn_delta = _encoder_parameter_delta(cnn_before, cnn_encoder)

    module_dir = _module_dir("encoder_training")
    _write_json(
        module_dir / "summary.json",
        {
            "epochs": DEFAULT_ENCODER_TRAIN_EPOCHS,
            "sample_count": len(base_observations),
            "mlp": {
                "latent_shape": list(mlp_latent.shape),
                "initial_loss": mlp_loss_history[0],
                "final_loss": mlp_loss_history[-1],
                "parameter_delta": mlp_delta,
            },
            "cnn": {
                "latent_shape": list(cnn_latent.shape),
                "initial_loss": cnn_loss_history[0],
                "final_loss": cnn_loss_history[-1],
                "parameter_delta": cnn_delta,
            },
        },
    )
    _write_line_plot_svg(
        module_dir / "mlp_loss.svg",
        title="MLP Encoder Reconstruction Loss",
        x_label="Epoch",
        y_label="Mean MSE",
        series=[("mlp_loss", mlp_loss_history, PLOT_COLORS[0])],
    )
    _write_line_plot_svg(
        module_dir / "cnn_loss.svg",
        title="CNN Encoder Reconstruction Loss",
        x_label="Epoch",
        y_label="Mean MSE",
        series=[("cnn_loss", cnn_loss_history, PLOT_COLORS[1])],
    )

    assert mlp_perception._train_encoder is True
    assert cnn_perception._train_encoder is True
    assert tuple(mlp_latent.shape) == (16,)
    assert tuple(cnn_latent.shape) == (32,)
    assert mlp_delta > 0.0
    assert cnn_delta > 0.0
    assert mlp_loss_history[-1] < mlp_loss_history[0]
    assert cnn_loss_history[-1] < cnn_loss_history[0]

    print(f"PASS: encoder-training diagnostics saved to {module_dir}")


def test_value_function_module() -> None:
    """ValueFunction should learn option values and GVFs from fixed transitions."""
    _set_seed(DEFAULT_SEED)
    agent = _build_cartpole_agent(device=_device())
    _, _, value_function, _ = _unwrap_modules(agent)

    option_id = "option:debug_value"
    gvf_id = "debug_value"
    option_idx = value_function.register_option(option_id)
    value_function.add_gvf(gvf_id)

    q_history: list[float] = []
    gvf_history: list[float] = []
    advantage_history: list[float] = []

    initial_predictions = value_function.predict(_subjective_state(REFERENCE_STATE))
    initial_q = float(initial_predictions[f"q_{option_id}"])
    initial_gvf = float(initial_predictions[f"gvf_{gvf_id}"])

    for step in range(DEFAULT_VALUE_UPDATES):
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
            )
        )

        predictions = value_function.predict(_subjective_state(REFERENCE_STATE))
        q_history.append(float(predictions[f"q_{option_id}"]))
        gvf_history.append(float(predictions[f"gvf_{gvf_id}"]))
        advantage_history.append(float(td_errors["advantage"]))

    value_function.observe_usage(
        [
            UsageRecord(ComponentKind.OPTION, option_id),
            UsageRecord(ComponentKind.FEATURE, gvf_id),
        ]
    )
    utility_ids = [record.component_id for record in value_function.utility_scores()]

    module_dir = _module_dir("value_function")
    _write_json(
        module_dir / "summary.json",
        {
            "updates": DEFAULT_VALUE_UPDATES,
            "buffer_size": len(value_function._buffer),
            "initial_q": initial_q,
            "final_q": q_history[-1],
            "initial_gvf": initial_gvf,
            "final_gvf": gvf_history[-1],
            "utility_components": utility_ids,
        },
    )
    _write_line_plot_svg(
        module_dir / "value_learning.svg",
        title="ValueFunction Learning",
        x_label="Update",
        y_label="Prediction",
        series=[
            ("q_omega", q_history, PLOT_COLORS[0]),
            ("gvf", gvf_history, PLOT_COLORS[1]),
        ],
    )
    _write_line_plot_svg(
        module_dir / "advantage.svg",
        title="ValueFunction Advantage Signal",
        x_label="Update",
        y_label="Advantage",
        series=[("advantage", advantage_history, PLOT_COLORS[4])],
    )

    assert len(value_function._buffer) == DEFAULT_VALUE_UPDATES
    assert q_history[-1] > initial_q + 0.05
    assert gvf_history[-1] > initial_gvf + 0.05
    assert option_id in utility_ids
    assert gvf_id in utility_ids

    print(f"PASS: value-function diagnostics saved to {module_dir}")


def test_reactive_policy_module() -> None:
    """ReactivePolicy should learn a preferred primitive action for one option."""
    _set_seed(DEFAULT_SEED)
    agent = _build_cartpole_agent(device=_device())
    _, reactive_policy, value_function, _ = _unwrap_modules(agent)

    subtask = SubtaskSpec(
        subtask_id="debug_policy",
        name="Debug policy option",
        feature_id="cart_position",
    )
    reactive_policy.ingest_subtasks([subtask])
    option_id = "option:debug_policy"

    action0_history: list[float] = []
    action1_history: list[float] = []
    stop_history: list[float] = []

    initial_q_values, initial_stop_prob = _policy_snapshot(
        reactive_policy,
        option_id,
        _subjective_state(REFERENCE_STATE),
    )

    for step in range(DEFAULT_POLICY_UPDATES):
        state = _state_variant(step)
        next_state = _state_variant(step + 1)
        _, active_option_id = reactive_policy.select_action(
            _subjective_state(state),
            option_stop_threshold=0.5,
        )
        assert active_option_id == option_id

        action = 1 if step % 4 else 0
        reward = 1.0 if action == 1 else 0.0
        reactive_policy.update(
            Transition(
                subjective_state=_subjective_state(state),
                action=action,
                reward=reward,
                next_subjective_state=_subjective_state(next_state),
                terminated=False,
            ),
            {"advantage": reward},
        )

        q_values, stop_prob = _policy_snapshot(
            reactive_policy,
            option_id,
            _subjective_state(REFERENCE_STATE),
        )
        action0_history.append(q_values[0])
        action1_history.append(q_values[1])
        stop_history.append(stop_prob)

    module_dir = _module_dir("reactive_policy")
    _write_json(
        module_dir / "summary.json",
        {
            "updates": DEFAULT_POLICY_UPDATES,
            "buffer_size": len(reactive_policy._buffer),
            "initial_q_values": initial_q_values,
            "final_q_values": [action0_history[-1], action1_history[-1]],
            "initial_stop_prob": initial_stop_prob,
            "final_stop_prob": stop_history[-1],
            "registered_options": sorted(reactive_policy._options),
            "value_function_option_ids": value_function.active_option_ids,
        },
    )
    _write_line_plot_svg(
        module_dir / "action_values.svg",
        title="ReactivePolicy Action Values",
        x_label="Update",
        y_label="Q(s, a)",
        series=[
            ("action_0", action0_history, PLOT_COLORS[0]),
            ("action_1", action1_history, PLOT_COLORS[1]),
        ],
    )
    _write_line_plot_svg(
        module_dir / "stop_probability.svg",
        title="ReactivePolicy Stop Probability",
        x_label="Update",
        y_label="stop_prob",
        series=[("stop_prob", stop_history, PLOT_COLORS[3])],
    )

    assert len(reactive_policy._buffer) == DEFAULT_POLICY_UPDATES
    assert action1_history[-1] > initial_q_values[1] + 0.05
    assert action1_history[-1] > action0_history[-1] + 0.05
    assert 0.0 <= stop_history[-1] <= 1.0

    print(f"PASS: reactive-policy diagnostics saved to {module_dir}")


def test_transition_model_module() -> None:
    """TransitionModel should fit fixed dynamics and produce planning updates."""
    _set_seed(DEFAULT_SEED)
    agent = _build_cartpole_agent(device=_device())
    _, _, value_function, transition_model = _unwrap_modules(agent)

    option_id = "option:debug_model"
    option_idx = value_function.register_option(option_id)
    value_function.set_last_option_idx(option_idx)

    loss_history: list[float] = []
    for step in range(DEFAULT_MODEL_UPDATES):
        action = step % 2
        state = REFERENCE_STATE.clone()
        next_state = state + ACTION_DELTAS[action]
        transition_model.update(
            Transition(
                subjective_state=_subjective_state(state),
                action=action,
                reward=ACTION_REWARDS[action],
                next_subjective_state=_subjective_state(next_state),
                terminated=False,
                option_id=option_id,
            )
        )
        loss_history.append(float(transition_model._model_loss))

    delta_errors: list[float] = []
    reward_errors: list[float] = []
    device_state = REFERENCE_STATE.to(transition_model._device)
    for action in (0, 1):
        with torch.no_grad():
            delta_pred, reward_pred = transition_model._model(
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

    q_before = float(value_function.predict(_subjective_state(REFERENCE_STATE))[f"q_{option_id}"])
    planning = transition_model.plan(_subjective_state(REFERENCE_STATE), value_function, budget=5)
    q_after = float(value_function.predict(_subjective_state(REFERENCE_STATE))[f"q_{option_id}"])

    module_dir = _module_dir("transition_model")
    _write_json(
        module_dir / "summary.json",
        {
            "updates": DEFAULT_MODEL_UPDATES,
            "buffer_size": len(transition_model._buffer),
            "final_model_loss": loss_history[-1],
            "delta_mse": delta_errors,
            "reward_mse": reward_errors,
            "planning_steps": planning.search_statistics.get("planning_steps", 0),
            "q_before_planning": q_before,
            "q_after_planning": q_after,
        },
    )
    _write_line_plot_svg(
        module_dir / "model_loss.svg",
        title="TransitionModel Loss",
        x_label="Update",
        y_label="Loss",
        series=[("model_loss", loss_history, PLOT_COLORS[4])],
    )

    assert len(transition_model._buffer) == DEFAULT_MODEL_UPDATES
    assert loss_history[-1] < max(loss_history)
    assert max(delta_errors) < 1e-3
    assert max(reward_errors) < 1e-3
    if DEFAULT_MODEL_UPDATES >= 500:
        assert planning.search_statistics.get("planning_steps", 0) == 5
        assert abs(q_after - q_before) > 1e-6

    print(f"PASS: transition-model diagnostics saved to {module_dir}")


def test_planner_value_handoff_respects_option_identity() -> None:
    """Planning should update the option that generated the sampled transition."""
    _set_seed(DEFAULT_SEED)
    agent = _build_cartpole_agent(device=_device())
    _, _, value_function, transition_model = _unwrap_modules(agent)

    source_option_id = "option:planner_source"
    stale_option_id = "option:planner_stale"
    source_option_idx = value_function.register_option(source_option_id)
    stale_option_idx = value_function.register_option(stale_option_id)

    warmup_updates = max(DEFAULT_MODEL_UPDATES, 560)
    for _ in range(warmup_updates):
        transition_model.update(
            Transition(
                subjective_state=_subjective_state(REFERENCE_STATE),
                action=1,
                reward=ACTION_REWARDS[1],
                next_subjective_state=_subjective_state(REFERENCE_STATE + ACTION_DELTAS[1]),
                terminated=False,
                option_id=source_option_id,
            )
        )

    output_head = cast(torch.nn.Linear, value_function._q_net.net[-1])
    before_weight = output_head.weight.detach().clone()
    before_bias = output_head.bias.detach().clone()

    # Simulate a stale "last selected option" that does not match the sampled model data.
    value_function.set_last_option_idx(stale_option_idx)
    planning = transition_model.plan(_subjective_state(REFERENCE_STATE), value_function, budget=1)

    after_weight = output_head.weight.detach().clone()
    after_bias = output_head.bias.detach().clone()

    source_row_change = float(
        torch.norm(after_weight[source_option_idx] - before_weight[source_option_idx])
        + torch.abs(after_bias[source_option_idx] - before_bias[source_option_idx])
    )
    stale_row_change = float(
        torch.norm(after_weight[stale_option_idx] - before_weight[stale_option_idx])
        + torch.abs(after_bias[stale_option_idx] - before_bias[stale_option_idx])
    )

    module_dir = _module_dir("planner_value_handoff")
    _write_json(
        module_dir / "summary.json",
        {
            "warmup_updates": warmup_updates,
            "planning_steps": planning.search_statistics.get("planning_steps", 0),
            "source_option_id": source_option_id,
            "stale_option_id": stale_option_id,
            "source_row_change": source_row_change,
            "stale_row_change": stale_row_change,
        },
    )

    assert planning.search_statistics.get("planning_steps", 0) == 1
    assert source_row_change > 0.0
    assert stale_row_change < 1e-9

    print(f"PASS: planner/value handoff diagnostics saved to {module_dir}")


def main() -> None:
    tests = [
        test_build_agent_components,
        test_perception_module,
        test_trainable_encoders_learn_reconstruction,
        test_value_function_module,
        test_reactive_policy_module,
        test_transition_model_module,
        test_planner_value_handoff_respects_option_identity,
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
