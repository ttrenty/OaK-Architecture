"""Reconstruct Example 01 training curves from sampled console logs.

This is useful when a long run finishes training but crashes during shutdown
before ``TrainingCurveRecorder.save()`` executes. The script rebuilds the
report-facing reward and training-state plots from the logged episode samples.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from examples.example_01.training_logging import (
    _AVG_SERIES_WIDTH,
    _DEFAULT_SERIES_OPACITY,
    _RAW_SERIES_OPACITY,
    _RAW_SERIES_WIDTH,
    _write_line_plot_assets,
)

_EPISODE_PATTERN = re.compile(
    r"Episode\s+(?P<episode>\d+)\s+\|\s+"
    r"Reward:\s+(?P<reward>-?\d+(?:\.\d+)?)\s+\|\s+"
    r"Avg\((?P<window>\d+)\):\s+(?P<avg>-?\d+(?:\.\d+)?)\s+\|\s+"
    r"Eps:\s+(?P<eps>-?\d+(?:\.\d+)?)\s+\|\s+"
    r"Options:\s+(?P<options>\d+)"
)


def _parse_console_samples(log_path: Path) -> tuple[int, list[dict[str, float]]]:
    average_window: int | None = None
    samples: list[dict[str, float]] = []
    for line in log_path.read_text().splitlines():
        match = _EPISODE_PATTERN.search(line)
        if match is None:
            continue
        window = int(match.group("window"))
        if average_window is None:
            average_window = window
        elif average_window != window:
            raise ValueError(
                f"Inconsistent average window in {log_path}: "
                f"{average_window} then {window}"
            )
        samples.append(
            {
                "episode": float(match.group("episode")),
                "reward": float(match.group("reward")),
                "avg_reward": float(match.group("avg")),
                "epsilon": float(match.group("eps")),
                "options": float(match.group("options")),
                "source": "console_log",
            }
        )
    if average_window is None or not samples:
        raise ValueError(f"No episode samples found in {log_path}")
    return average_window, samples


def _append_trace_sample(
    samples: list[dict[str, float]],
    trace_json_path: Path | None,
) -> list[dict[str, float]]:
    if trace_json_path is None:
        return samples
    payload = json.loads(trace_json_path.read_text())
    episode = float(payload["episode"])
    trace_sample = {
        "episode": episode,
        "reward": float(payload["episode_reward"]),
        "avg_reward": float(payload["avg_reward"]),
        "epsilon": float("nan"),
        "options": float("nan"),
        "source": str(trace_json_path),
    }
    by_episode = {sample["episode"]: sample for sample in samples}
    by_episode[episode] = trace_sample
    return [by_episode[index] for index in sorted(by_episode)]


def _dense_history(
    total_episodes: int,
    samples: list[dict[str, float]],
    key: str,
) -> list[float]:
    dense = [float("nan")] * total_episodes
    for sample in samples:
        episode = int(sample["episode"])
        if 0 <= episode < total_episodes:
            dense[episode] = float(sample[key])
    return dense


def _sampled_history(samples: list[dict[str, float]], key: str) -> list[float]:
    return [float(sample[key]) for sample in samples]


def _series_name(prefix: str) -> str:
    return prefix.replace("_", " ")


def reconstruct_curves(
    *,
    console_log_path: Path,
    output_dir: Path,
    prefix: str,
    total_episodes: int,
    trace_json_path: Path | None = None,
) -> dict[str, Any]:
    average_window, samples = _parse_console_samples(console_log_path)
    samples = _append_trace_sample(samples, trace_json_path)

    dense_rewards = _dense_history(total_episodes, samples, "reward")
    dense_averages = _dense_history(total_episodes, samples, "avg_reward")
    dense_epsilons = _dense_history(total_episodes, samples, "epsilon")
    dense_options = _dense_history(total_episodes, samples, "options")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_prefix = output_dir / prefix
    display_name = _series_name(prefix)

    _write_line_plot_assets(
        plot_prefix.with_name(f"{prefix}_reward_curve"),
        title=f"{display_name} reward over episodes (reconstructed)",
        x_label="Episode",
        y_label="Reward",
        series=[
            (
                "reward (sampled)",
                dense_rewards,
                "#264653",
                _RAW_SERIES_OPACITY,
                _RAW_SERIES_WIDTH,
            ),
            (
                f"avg{average_window}",
                dense_averages,
                "#2a9d8f",
                _DEFAULT_SERIES_OPACITY,
                _AVG_SERIES_WIDTH,
            ),
        ],
    )
    _write_line_plot_assets(
        plot_prefix.with_name(f"{prefix}_training_state"),
        title=f"{display_name} exploration and option count (reconstructed)",
        x_label="Episode",
        y_label="Value",
        series=[
            (
                "epsilon",
                dense_epsilons,
                "#e76f51",
                _DEFAULT_SERIES_OPACITY,
                _AVG_SERIES_WIDTH,
            ),
            (
                "options",
                dense_options,
                "#457b9d",
                _DEFAULT_SERIES_OPACITY,
                _AVG_SERIES_WIDTH,
            ),
        ],
    )

    finite_rewards = [
        reward for reward in _sampled_history(samples, "reward") if math.isfinite(reward)
    ]
    payload: dict[str, Any] = {
        "reconstructed": True,
        "reconstruction_source": str(console_log_path),
        "trace_json_source": str(trace_json_path) if trace_json_path is not None else None,
        "average_window": average_window,
        "episodes": total_episodes,
        "sample_count": len(samples),
        "sampled_episodes": [int(sample["episode"]) for sample in samples],
        "reward_history": _sampled_history(samples, "reward"),
        "average_history": _sampled_history(samples, "avg_reward"),
        "epsilon_history": _sampled_history(samples, "epsilon"),
        "option_history": _sampled_history(samples, "options"),
        "metric_histories": {},
        "best_reward": max(finite_rewards) if finite_rewards else None,
        "final_reward": float(samples[-1]["reward"]),
        "final_average": float(samples[-1]["avg_reward"]),
        "note": (
            "Recovered from sampled console output because the full recorder did not "
            "finish saving after training."
        ),
        "samples": samples,
    }
    plot_prefix.with_name(f"{prefix}_reward_history.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct Example 01 reward/training plots from console output."
    )
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--trace-json", type=Path)
    args = parser.parse_args()

    reconstruct_curves(
        console_log_path=args.console_log,
        output_dir=args.output_dir,
        prefix=args.prefix,
        total_episodes=args.episodes,
        trace_json_path=args.trace_json,
    )


if __name__ == "__main__":
    main()
