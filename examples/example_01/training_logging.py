"""Training-time logging helpers for Example 01."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from oak.types import EpisodeTrace

_DEFAULT_ANIMATION_OUTPUT_DIR = Path("tests/results/animations")
_DEFAULT_ANIMATION_EVERY = 100
_DEFAULT_ANIMATION_LAST = 1
_DEFAULT_ANIMATION_FPS = 30
_DEFAULT_ANIMATION_MAX_FRAMES = 400
_DEFAULT_CURVE_OUTPUT_DIR = Path("tests/results/training_curves")
_DEFAULT_CURVE_CHECKPOINT_EVERY = 10
_DEFAULT_SERIES_OPACITY = 1.0
_RAW_SERIES_OPACITY = 0.25
_RAW_SERIES_WIDTH = 1.5
_AVG_SERIES_WIDTH = 2.75


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(window, 1)
    averaged: list[float] = []
    for index, value in enumerate(values):
        del value
        start = max(0, index - window + 1)
        window_values = [
            float(candidate)
            for candidate in values[start : index + 1]
            if math.isfinite(float(candidate))
        ]
        if not window_values:
            averaged.append(float("nan"))
        else:
            averaged.append(sum(window_values) / len(window_values))
    return averaged


def _episode_metric_snapshot(agent: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for component_name in (
        "perception",
        "value_function",
        "reactive_policy",
        "transition_model",
    ):
        component = getattr(agent, component_name, None)
        metric_fn = getattr(component, "training_metrics", None)
        if callable(metric_fn):
            for key, value in dict(metric_fn()).items():
                if math.isfinite(float(value)):
                    metrics[key] = float(value)
    return metrics


@dataclass(slots=True, frozen=True)
class EpisodeCaptureSchedule:
    """Select which training episodes should produce artifacts."""

    episode_indices: tuple[int, ...] = ()
    every_n_episodes: int | None = None
    last_n_episodes: int = 0

    def should_capture(self, episode: int, total_episodes: int) -> bool:
        if episode in self.episode_indices:
            return True
        if self.every_n_episodes is not None and self.every_n_episodes > 0:
            if episode % self.every_n_episodes == 0:
                return True
        if self.last_n_episodes > 0 and episode >= max(total_episodes - self.last_n_episodes, 0):
            return True
        return False


@dataclass(slots=True)
class EpisodeAnimationRecorder:
    """Save traced episode frames as GIF animations plus compact metadata."""

    output_dir: Path
    schedule: EpisodeCaptureSchedule = field(default_factory=EpisodeCaptureSchedule)
    prefix: str = "training"
    fps: int = 30
    max_frames: int | None = None
    save_metadata: bool = True

    def should_capture(self, episode: int, total_episodes: int) -> bool:
        return self.schedule.should_capture(episode, total_episodes)

    def __call__(self, trace: EpisodeTrace[Any, Any, Any, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self._episode_stem(trace)
        metadata_path = self.output_dir / f"{stem}.json"
        gif_path = self.output_dir / f"{stem}.gif"

        frames = list(trace.frames)
        if self.max_frames is not None and self.max_frames > 0:
            frames = frames[: self.max_frames]
        if not frames:
            if self.save_metadata:
                metadata = self._trace_metadata(trace)
                metadata["warning"] = (
                    "No frames were captured for this episode. Ensure the world was "
                    "created with render_mode='rgb_array' and the environment's "
                    "rendering dependencies are installed."
                )
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            return

        pil_frames = [self._to_pil_image(frame) for frame in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=max(int(1000 / max(self.fps, 1)), 1),
            loop=0,
        )

        if self.save_metadata:
            metadata_path.write_text(json.dumps(self._trace_metadata(trace), indent=2) + "\n")

    def _episode_stem(self, trace: EpisodeTrace[Any, Any, Any, Any]) -> str:
        return (
            f"{self.prefix}_episode_{trace.episode:04d}"
            f"_reward_{int(round(trace.episode_reward))}"
            f"_avg_{int(round(trace.avg_reward))}"
        )

    def _trace_metadata(
        self,
        trace: EpisodeTrace[Any, Any, Any, Any],
    ) -> dict[str, Any]:
        return {
            "episode": trace.episode,
            "episode_reward": trace.episode_reward,
            "avg_reward": trace.avg_reward,
            "step_count": trace.step_count,
            "solved": trace.solved,
            "frame_count": len(trace.frames),
            "actions": [step.action for step in trace.steps],
            "active_option_ids": [step.active_option_id for step in trace.steps],
            "metadata": dict(trace.metadata),
        }

    def _to_pil_image(self, frame: object) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        array = np.asarray(frame)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")


def animation_recorder_from_env(mode: str) -> EpisodeAnimationRecorder | None:
    """Build an animation recorder from env vars with sensible defaults.

    Animation capture is enabled by default for the example training entry
    points so that long runs automatically leave behind visual artifacts.
    Set ``OAK_EXAMPLE_DISABLE_ANIMATIONS=1`` to turn it off.
    """

    disable_token = os.environ.get("OAK_EXAMPLE_DISABLE_ANIMATIONS", "").strip().lower()
    if disable_token in {"1", "true", "yes", "on"}:
        return None

    output_dir = (
        os.environ.get("OAK_EXAMPLE_ANIMATION_DIR", "").strip()
        or str(_DEFAULT_ANIMATION_OUTPUT_DIR)
    )
    episode_tokens = os.environ.get("OAK_EXAMPLE_ANIMATION_EPISODES", "").split(",")
    episode_indices = tuple(
        int(token.strip())
        for token in episode_tokens
        if token.strip()
    )
    every_n = int(
        os.environ.get(
            "OAK_EXAMPLE_ANIMATION_EVERY",
            str(_DEFAULT_ANIMATION_EVERY),
        )
        or _DEFAULT_ANIMATION_EVERY
    )
    last_n = int(
        os.environ.get(
            "OAK_EXAMPLE_ANIMATION_LAST",
            str(_DEFAULT_ANIMATION_LAST),
        )
        or _DEFAULT_ANIMATION_LAST
    )
    fps = int(
        os.environ.get(
            "OAK_EXAMPLE_ANIMATION_FPS",
            str(_DEFAULT_ANIMATION_FPS),
        )
        or _DEFAULT_ANIMATION_FPS
    )
    max_frames = int(
        os.environ.get(
            "OAK_EXAMPLE_ANIMATION_MAX_FRAMES",
            str(_DEFAULT_ANIMATION_MAX_FRAMES),
        )
        or _DEFAULT_ANIMATION_MAX_FRAMES
    )

    return EpisodeAnimationRecorder(
        output_dir=Path(output_dir) / mode,
        schedule=EpisodeCaptureSchedule(
            episode_indices=episode_indices,
            every_n_episodes=every_n or None,
            last_n_episodes=last_n,
        ),
        prefix=mode,
        fps=fps,
        max_frames=max_frames or None,
    )


def curve_recorder_from_env(
    mode: str,
    *,
    average_window: int,
) -> TrainingCurveRecorder | None:
    """Build a curve recorder from env vars with sensible defaults.

    Curve capture is enabled by default so short training runs still write the
    SVG/JSON artifacts used in the report. Set
    ``OAK_EXAMPLE_DISABLE_CURVES=1`` to turn it off.
    """

    disable_token = os.environ.get("OAK_EXAMPLE_DISABLE_CURVES", "").strip().lower()
    if disable_token in {"1", "true", "yes", "on"}:
        return None

    output_dir = (
        os.environ.get("OAK_EXAMPLE_CURVE_DIR", "").strip()
        or os.environ.get("OAK_EXAMPLE_PLOT_DIR", "").strip()
        or str(_DEFAULT_CURVE_OUTPUT_DIR)
    )
    checkpoint_every = int(
        os.environ.get(
            "OAK_EXAMPLE_CURVE_CHECKPOINT_EVERY",
            str(_DEFAULT_CURVE_CHECKPOINT_EVERY),
        )
        or _DEFAULT_CURVE_CHECKPOINT_EVERY
    )
    return TrainingCurveRecorder(
        output_dir=Path(output_dir) / mode,
        prefix=mode,
        average_window=average_window,
        checkpoint_every=checkpoint_every,
    )


def _write_line_plot_svg(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[Any, ...]],
) -> None:
    """Write a lightweight SVG line chart without external plotting deps."""
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
        for entry in series
        for value in entry[1]
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

    x_max = max((len(entry[1]) - 1 for entry in series if entry[1]), default=1)
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

    x_ticks = sorted({round(tick * x_max / 5) for tick in range(6)})
    x_tick_labels: list[str] = []
    for x_tick in x_ticks:
        x = x_pos(int(x_tick))
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height:.2f}" '
            'stroke="#eef2f7" stroke-width="1" />'
        )
        x_tick_labels.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 20:.2f}" text-anchor="middle" '
            'font-size="12" fill="#334155">'
            f"{int(x_tick)}</text>"
        )

    polylines: list[str] = []
    legend_items: list[str] = []
    for index, entry in enumerate(series):
        label = str(entry[0])
        seq = entry[1]
        color = str(entry[2])
        opacity = float(entry[3]) if len(entry) >= 4 else _DEFAULT_SERIES_OPACITY
        stroke_width = float(entry[4]) if len(entry) >= 5 else 2.5
        points = " ".join(
            f"{x_pos(i):.2f},{y_pos(float(value)):.2f}"
            for i, value in enumerate(seq)
            if math.isfinite(float(value))
        )
        if points:
            polylines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" '
                f'stroke-opacity="{opacity:.3f}" '
                f'stroke-linejoin="round" stroke-linecap="round" points="{points}" />'
            )
        legend_y = top + 16 + 20 * index
        legend_items.append(
            f'<rect x="{width - right - 180}" y="{legend_y - 10}" width="14" height="14" '
            f'fill="{color}" fill-opacity="{max(opacity, 0.35):.3f}" rx="3" />'
            f'<text x="{width - right - 160}" y="{legend_y + 1}" font-size="13" '
            'fill="#0f172a">'
            f"{escape(label)}</text>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
  <rect width="100%" height="100%" fill="#f8fafc" />
  <text x="{left}" y="28" font-size="22" font-weight="700" fill="#0f172a">{escape(title)}</text>
  <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="10" />
  {''.join(grid_lines)}
  {''.join(tick_labels)}
  {''.join(x_tick_labels)}
  {''.join(polylines)}
  {''.join(legend_items)}
  <text x="{left + plot_width / 2:.2f}" y="{height - 16}" text-anchor="middle" font-size="14" fill="#334155">{escape(x_label)}</text>
  <text x="20" y="{top + plot_height / 2:.2f}" text-anchor="middle" font-size="14" fill="#334155" transform="rotate(-90 20 {top + plot_height / 2:.2f})">{escape(y_label)}</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def _hex_to_rgba(color: str, alpha: float = 1.0) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    if len(color) != 6:
        return (107, 114, 128, int(255 * alpha))
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return (red, green, blue, max(0, min(255, int(round(255 * alpha)))))


def _write_line_plot_png(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[Any, ...]],
) -> None:
    """Write a lightweight PNG line chart without plotting dependencies."""
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
        for entry in series
        for value in entry[1]
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

    x_max = max((len(entry[1]) - 1 for entry in series if entry[1]), default=1)
    x_max = max(x_max, 1)

    def x_pos(index: int) -> float:
        return left + (index / x_max) * plot_width

    def y_pos(value: float) -> float:
        norm = (value - y_min) / (y_max - y_min)
        return top + (1.0 - norm) * plot_height

    image = Image.new("RGBA", (width, height), (248, 250, 252, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    draw.rounded_rectangle(
        (left, top, left + plot_width, top + plot_height),
        radius=10,
        fill=(255, 255, 255, 255),
        outline=(203, 213, 225, 255),
        width=2,
    )
    draw.text((left, 16), title, fill=(15, 23, 42, 255), font=font)

    for tick in range(6):
        frac = tick / 5
        y_value = y_max - frac * (y_max - y_min)
        y = top + frac * plot_height
        draw.line(
            ((left, y), (width - right, y)),
            fill=(215, 221, 229, 255),
            width=1,
        )
        draw.text(
            (left - 52, y - 6),
            f"{y_value:.2f}",
            fill=(51, 65, 85, 255),
            font=font,
        )

    x_ticks = sorted({round(tick * x_max / 5) for tick in range(6)})
    for x_tick in x_ticks:
        x = x_pos(int(x_tick))
        draw.line(
            ((x, top), (x, top + plot_height)),
            fill=(238, 242, 247, 255),
            width=1,
        )
        draw.text(
            (x - 8, top + plot_height + 8),
            str(int(x_tick)),
            fill=(51, 65, 85, 255),
            font=font,
        )

    for entry in series:
        seq = entry[1]
        color = str(entry[2])
        opacity = float(entry[3]) if len(entry) >= 4 else _DEFAULT_SERIES_OPACITY
        stroke_width = max(1, int(round(float(entry[4]) if len(entry) >= 5 else 2.5)))
        points = [
            (x_pos(i), y_pos(float(value)))
            for i, value in enumerate(seq)
            if math.isfinite(float(value))
        ]
        if len(points) >= 2:
            draw.line(
                points,
                fill=_hex_to_rgba(color, opacity),
                width=stroke_width,
                joint="curve",
            )

    legend_x = width - right - 180
    for index, entry in enumerate(series):
        label = str(entry[0])
        color = str(entry[2])
        opacity = float(entry[3]) if len(entry) >= 4 else _DEFAULT_SERIES_OPACITY
        legend_y = top + 6 + 20 * index
        draw.rounded_rectangle(
            (legend_x, legend_y, legend_x + 14, legend_y + 14),
            radius=3,
            fill=_hex_to_rgba(color, max(opacity, 0.35)),
        )
        draw.text(
            (legend_x + 20, legend_y),
            label,
            fill=(15, 23, 42, 255),
            font=font,
        )

    draw.text(
        (left + plot_width / 2 - 25, height - 24),
        x_label,
        fill=(51, 65, 85, 255),
        font=font,
    )
    draw.text((8, top + plot_height / 2), y_label, fill=(51, 65, 85, 255), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG")


def _write_line_plot_assets(
    stem: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[Any, ...]],
) -> None:
    _write_line_plot_svg(
        stem.with_suffix(".svg"),
        title=title,
        x_label=x_label,
        y_label=y_label,
        series=series,
    )
    _write_line_plot_png(
        stem.with_suffix(".png"),
        title=title,
        x_label=x_label,
        y_label=y_label,
        series=series,
    )


@dataclass(slots=True)
class TrainingCurveRecorder:
    """Persist reward and moving-average curves for a full training run."""

    output_dir: Path
    prefix: str = "training"
    average_window: int = 100
    reward_color: str = "#264653"
    average_color: str = "#2a9d8f"
    epsilon_color: str = "#e76f51"
    option_color: str = "#457b9d"
    reward_history: list[float] = field(default_factory=list)
    average_history: list[float] = field(default_factory=list)
    epsilon_history: list[float] = field(default_factory=list)
    option_history: list[float] = field(default_factory=list)
    metric_histories: dict[str, list[float]] = field(default_factory=dict)
    checkpoint_every: int = 0
    metric_colors: dict[str, str] = field(
        default_factory=lambda: {
            "perception_encoder_loss": "#8a5cf6",
            "value_q_omega_loss": "#c0392b",
            "value_gvf_loss": "#d35400",
            "policy_q_loss": "#2980b9",
            "policy_termination_loss": "#16a085",
            "model_loss": "#7f8c8d",
            "model_done_loss": "#34495e",
        }
    )

    def log_episode(
        self,
        episode: int,
        reward: float,
        avg_reward: float,
        agent: Any,
    ) -> None:
        """Record one episode's aggregate metrics."""
        current_episode = int(episode)
        self.reward_history.append(float(reward))
        self.average_history.append(float(avg_reward))
        self.epsilon_history.append(
            float(getattr(agent.reactive_policy, "epsilon", 0.0))
        )
        self.option_history.append(
            float(len(getattr(agent.reactive_policy, "_options", {})))
        )
        current_metrics = _episode_metric_snapshot(agent)
        episode_count = len(self.reward_history)
        for metric_name in current_metrics:
            self.metric_histories.setdefault(
                metric_name,
                [float("nan")] * (episode_count - 1),
            )
        for metric_name, history in self.metric_histories.items():
            history.append(float(current_metrics.get(metric_name, float("nan"))))
        if self.checkpoint_every > 0 and (current_episode + 1) % self.checkpoint_every == 0:
            self.save_checkpoint()

    def _history_payload(self) -> dict[str, Any]:
        return {
            "average_window": self.average_window,
            "episodes": len(self.reward_history),
            "reward_history": self.reward_history,
            "average_history": self.average_history,
            "epsilon_history": self.epsilon_history,
            "option_history": self.option_history,
            "metric_histories": self.metric_histories,
            "best_reward": max(self.reward_history),
            "final_reward": self.reward_history[-1],
            "final_average": self.average_history[-1],
        }

    def save_checkpoint(self) -> None:
        """Write the raw training histories without rendering plots."""
        if not self.reward_history:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.output_dir / self.prefix
        payload = self._history_payload()
        payload["checkpoint"] = True
        stem.with_name(f"{self.prefix}_reward_history_checkpoint.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )

    def save(self) -> None:
        """Write SVG plots and the raw history JSON."""
        if not self.reward_history:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.output_dir / self.prefix

        _write_line_plot_assets(
            stem.with_name(f"{self.prefix}_reward_curve"),
            title=f"{self.prefix} reward over episodes",
            x_label="Episode",
            y_label="Reward",
            series=[
                (
                    "reward",
                    self.reward_history,
                    self.reward_color,
                    _RAW_SERIES_OPACITY,
                    _RAW_SERIES_WIDTH,
                ),
                (
                    f"avg{self.average_window}",
                    _moving_average(self.reward_history, self.average_window),
                    self.average_color,
                    _DEFAULT_SERIES_OPACITY,
                    _AVG_SERIES_WIDTH,
                ),
            ],
        )
        _write_line_plot_assets(
            stem.with_name(f"{self.prefix}_training_state"),
            title=f"{self.prefix} exploration and option count",
            x_label="Episode",
            y_label="Value",
            series=[
                ("epsilon", self.epsilon_history, self.epsilon_color),
                ("options", self.option_history, self.option_color),
            ],
        )

        for metric_name, history in sorted(self.metric_histories.items()):
            if not history:
                continue
            metric_label = metric_name.replace("_", " ")
            color = self.metric_colors.get(metric_name, "#6b7280")
            smoothed = _moving_average(history, self.average_window)
            _write_line_plot_assets(
                stem.with_name(f"{self.prefix}_{metric_name}"),
                title=f"{self.prefix} {metric_label} over episodes",
                x_label="Episode",
                y_label=metric_label,
                series=[
                    (
                        f"{metric_label} raw",
                        history,
                        color,
                        _RAW_SERIES_OPACITY,
                        _RAW_SERIES_WIDTH,
                    ),
                    (
                        f"{metric_label} avg{self.average_window}",
                        smoothed,
                        color,
                        _DEFAULT_SERIES_OPACITY,
                        _AVG_SERIES_WIDTH,
                    ),
                ],
            )

        payload = self._history_payload()
        stem.with_name(f"{self.prefix}_reward_history.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
