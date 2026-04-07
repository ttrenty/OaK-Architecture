"""ARC-AGI-3 benchmark for the OaK example_01 agent.

Runs the example_01 agent on the public ARC-AGI-3 demo environments and
computes RHAE (Relative Human Action Efficiency) scores.

Environment variables
---------------------
OAK_ARC_MAX_ENVS : int
    Maximum number of environments to benchmark.  Default: all keyboard envs.
OAK_ARC_PLANNING_BUDGET : int
    Planning rollouts per agent step.  Default: 5.
OAK_ARC_DEVICE : str
    Torch device.  Default: ``"cpu"``.
OAK_ARC_OPERATION_MODE : str
    ARC toolkit mode: ``offline``, ``normal``, or ``online``.
    Default: ``"offline"`` for reliable local benchmarking.
OAK_ARC_API_KEY / ARC_API_KEY : str
    ARC Prize API key used for ``normal`` / ``online`` modes.
OAK_ARC_PRETRAIN_EPISODES : int
    Number of local offline warmup episodes before the scored run.
    Default: ``0``.
OAK_ARC_TRAIN_ENCODER : 0 | 1
    Whether to train the CNN encoder reconstruction objective. Default: ``1``.
OAK_ARC_PLANNING_WARMUP : int
    Number of real transitions before Dyna planning activates. Default: ``32``.
OAK_ARC_GREEDY_EVAL : 0 | 1
    Run the scored evaluation greedily after pretraining. Default: ``1``.
OAK_ARC_OUTPUT_DIR : str
    Directory for result files.  Default: ``"tests/results/benchmark_arc_agi"``.
OAK_ARC_VERBOSE : 0 | 1
    Print per-step logs.  Default: 1.

Usage
-----
    pixi run benchmark_arc_agi
    # or directly:
    env PYTHONPATH=src python -m tests.benchmark_arc_agi
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias, cast

import arc_agi
import numpy as np
import torch

from examples.example_01.runner import build_agent
from examples.example_01.world_arc import ArcWorld
from oak.agent import OaKAgent

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

MAX_ENVS = int(os.environ.get("OAK_ARC_MAX_ENVS", "0"))  # 0 = all
PLANNING_BUDGET = int(os.environ.get("OAK_ARC_PLANNING_BUDGET", "5"))
PLANNING_WARMUP = int(os.environ.get("OAK_ARC_PLANNING_WARMUP", "32"))
DEVICE = os.environ.get("OAK_ARC_DEVICE", "cpu")
OPERATION_MODE = os.environ.get(
    "OAK_ARC_OPERATION_MODE",
    os.environ.get("OPERATION_MODE", "offline"),
).strip().lower()
ARC_API_KEY = os.environ.get("OAK_ARC_API_KEY", os.environ.get("ARC_API_KEY", ""))
PRETRAIN_EPISODES = int(os.environ.get("OAK_ARC_PRETRAIN_EPISODES", "0"))
TRAIN_ENCODER = bool(int(os.environ.get("OAK_ARC_TRAIN_ENCODER", "1")))
GREEDY_EVAL = bool(int(os.environ.get("OAK_ARC_GREEDY_EVAL", "1")))
OUTPUT_DIR = os.environ.get(
    "OAK_ARC_OUTPUT_DIR", "tests/results/benchmark_arc_agi"
)
VERBOSE = bool(int(os.environ.get("OAK_ARC_VERBOSE", "1")))

ArcInfo: TypeAlias = dict[str, Any]
ArcAgent: TypeAlias = OaKAgent[np.ndarray, int, torch.Tensor, ArcInfo]


# ---------------------------------------------------------------------------
# RHAE scoring (matches ARC-AGI-3 specification)
# ---------------------------------------------------------------------------


def rhae_level_score(
    actions_taken: int, baseline_actions: int, completed: bool
) -> float:
    """Compute RHAE score for a single level.

    S = min(1.0, baseline / actions)^2   if completed
    S = 0.0                               if not completed
    """
    if not completed or actions_taken <= 0:
        return 0.0
    raw = min(1.0, baseline_actions / actions_taken)
    return raw**2


def rhae_environment_score(level_scores: list[float]) -> float:
    """Weighted average across levels (later levels count more).

    E = sum(l * S_l for l in 1..n) / (n*(n+1)/2)
    """
    n = len(level_scores)
    if n == 0:
        return 0.0
    weighted = sum((i + 1) * s for i, s in enumerate(level_scores))
    normalizer = n * (n + 1) / 2
    return weighted / normalizer


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class LevelResult:
    level_index: int
    completed: bool
    actions_taken: int
    baseline_actions: int
    score: float


@dataclass
class EnvironmentResult:
    game_id: str
    title: str
    tags: list[str]
    levels: list[LevelResult] = field(default_factory=list)
    score: float = 0.0
    total_actions: int = 0
    levels_completed: int = 0
    elapsed_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class BenchmarkResult:
    environments: list[EnvironmentResult] = field(default_factory=list)
    total_score: float = 0.0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# ARC helpers
# ---------------------------------------------------------------------------


def _set_greedy_eval(
    agent: ArcAgent, enabled: bool
) -> tuple[float, float, int] | None:
    """Temporarily disable epsilon exploration for evaluation."""
    if not enabled:
        return None

    policy = cast(Any, agent.reactive_policy)
    if not hasattr(policy, "_epsilon_start"):
        return None

    previous = (
        float(policy._epsilon_start),
        float(policy._epsilon_end),
        int(policy._step_count),
    )
    policy._epsilon_start = 0.0
    policy._epsilon_end = 0.0
    policy._step_count = max(policy._step_count, 1_000_000_000)
    return previous


def _restore_greedy_eval(
    agent: ArcAgent, previous: tuple[float, float, int] | None
) -> None:
    if previous is None:
        return

    policy = cast(Any, agent.reactive_policy)
    policy._epsilon_start, policy._epsilon_end, policy._step_count = previous


def _run_arc_agent(agent: ArcAgent, world: ArcWorld, *, greedy_eval: bool) -> None:
    """Run one full ARC game with OaK's continuous level-transition logic."""
    previous_greedy_state = _set_greedy_eval(agent, greedy_eval)
    try:
        time_step = world.reset()
        agent.reset()
        prev_levels = 0

        while not world.is_game_over:
            step_result = agent.step(time_step)

            if time_step.terminated or time_step.truncated:
                agent.reactive_policy.clear_active_option()
                agent.last_action = None
                agent.last_subjective_state = None

                if world.is_game_over:
                    break

                if VERBOSE and world.levels_completed > prev_levels:
                    print(
                        f"    Level {prev_levels} WON at step {world.total_actions}"
                    )
                    prev_levels = world.levels_completed

                time_step = world.reset()
                continue

            time_step = world.step(step_result.action)
    finally:
        _restore_greedy_eval(agent, previous_greedy_state)


def _lookup_env_info(
    arcade: arc_agi.Arcade, game_id: str
) -> arc_agi.EnvironmentInfo | None:
    for env_info in arcade.get_environments():
        if env_info.game_id == game_id:
            return env_info
    return None


def _make_arcade(
    operation_mode: arc_agi.OperationMode, arc_api_key: str
) -> arc_agi.Arcade:
    if operation_mode == arc_agi.OperationMode.OFFLINE:
        return arc_agi.Arcade(operation_mode=operation_mode)
    return arc_agi.Arcade(operation_mode=operation_mode, arc_api_key=arc_api_key)


def _parse_operation_mode(mode_name: str) -> arc_agi.OperationMode:
    lookup = {
        "offline": arc_agi.OperationMode.OFFLINE,
        "normal": arc_agi.OperationMode.NORMAL,
        "online": arc_agi.OperationMode.ONLINE,
        "competition": arc_agi.OperationMode.COMPETITION,
    }
    try:
        return lookup[mode_name]
    except KeyError as exc:
        valid = ", ".join(sorted(lookup))
        raise ValueError(
            f"OAK_ARC_OPERATION_MODE={mode_name!r} is invalid; expected one of: {valid}"
        ) from exc


def _build_arc_agent(
    world: ArcWorld,
    device: torch.device,
) -> ArcAgent:
    config = world.description.to_config()
    return cast(
        ArcAgent,
        build_agent(
            config,
            train_encoder=TRAIN_ENCODER,
            planning_budget=PLANNING_BUDGET,
            planning_warmup_steps=PLANNING_WARMUP,
            device=device,
        ),
    )


def _pretrain_agent_if_requested(
    env_info: arc_agi.EnvironmentInfo,
    *,
    device: torch.device,
) -> ArcAgent | None:
    if PRETRAIN_EPISODES <= 0:
        return None

    local_arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    local_env_info = _lookup_env_info(local_arcade, env_info.game_id)
    if local_env_info is None:
        if VERBOSE:
            print("  Pretraining skipped: no matching local environment copy")
        return None

    env = local_arcade.make(local_env_info.game_id)
    if env is None:
        if VERBOSE:
            print("  Pretraining skipped: local environment creation failed")
        return None

    agent = _build_arc_agent(ArcWorld(env, local_env_info), device=device)
    best_agent: ArcAgent | None = None
    best_key = (-1, float("-inf"))

    if VERBOSE:
        print(
            f"  Pretraining offline for {PRETRAIN_EPISODES} episode(s) "
            f"(train_encoder={int(TRAIN_ENCODER)}, planning_warmup={PLANNING_WARMUP})"
        )

    for episode_idx in range(PRETRAIN_EPISODES):
        episode_env = local_arcade.make(local_env_info.game_id)
        if episode_env is None:
            break
        episode_world = ArcWorld(episode_env, local_env_info)
        _run_arc_agent(agent, episode_world, greedy_eval=False)

        metric = (episode_world.levels_completed, -episode_world.total_actions)
        if VERBOSE:
            print(
                f"    warmup {episode_idx + 1:02d}/{PRETRAIN_EPISODES}: "
                f"levels={episode_world.levels_completed} "
                f"actions={episode_world.total_actions}"
            )
        if metric > best_key:
            best_key = metric
            best_agent = copy.deepcopy(agent)

    local_arcade.close_scorecard()
    return best_agent or agent


# ---------------------------------------------------------------------------
# Single-environment benchmark
# ---------------------------------------------------------------------------


def _benchmark_one_environment(
    arcade: arc_agi.Arcade,
    env_info: arc_agi.EnvironmentInfo,
    device: torch.device,
) -> EnvironmentResult:
    """Run the OaK agent on one ARC-AGI-3 environment."""

    result = EnvironmentResult(
        game_id=env_info.game_id,
        title=env_info.title,
        tags=list(env_info.tags or []),
    )

    t0 = time.monotonic()

    # Create environment.
    try:
        env = arcade.make(env_info.game_id)
    except Exception as exc:
        result.skipped = True
        result.skip_reason = f"Failed to create environment: {exc}"
        result.elapsed_seconds = time.monotonic() - t0
        return result

    # Wrap in OaK World.
    try:
        world = ArcWorld(env, env_info)
    except ValueError as exc:
        result.skipped = True
        result.skip_reason = str(exc)
        result.elapsed_seconds = time.monotonic() - t0
        return result

    agent = _pretrain_agent_if_requested(env_info, device=device) or _build_arc_agent(
        world, device=device
    )

    if VERBOSE:
        print(
            f"  Agent built: action_n={world.action_n}, "
            f"levels={world.win_levels}, "
            f"baselines={world.baseline_actions}"
        )

    _run_arc_agent(agent, world, greedy_eval=GREEDY_EVAL)

    if VERBOSE:
        print(
            f"    Game ended: {world.levels_completed}/{world.win_levels} "
            f"levels, {world.total_actions} total actions"
        )

    # ------------------------------------------------------------------
    # Read per-level results from the arcade's scorecard.
    # ------------------------------------------------------------------
    result.total_actions = world.total_actions
    result.levels_completed = world.levels_completed

    scorecard = arcade.get_scorecard()
    baselines = world.baseline_actions

    # Parse per-level data from the arcade's scorecard (Pydantic model).
    level_scores: list[float] = []

    # Convert scorecard to dict for safe traversal.
    sc_data = (
        scorecard.model_dump()
        if hasattr(scorecard, "model_dump")
        else scorecard
    )
    sc_envs = sc_data.get("environments", []) if isinstance(sc_data, dict) else []

    # Find our environment in the scorecard.
    sc_env = None
    for se in sc_envs:
        if isinstance(se, dict) and se.get("id") == env_info.game_id:
            sc_env = se
            break
    if sc_env is None:
        game_prefix = env_info.game_id.split("-")[0]
        for se in sc_envs:
            if isinstance(se, dict) and se.get("id", "").startswith(game_prefix):
                sc_env = se
                break

    level_actions_list: list[int] = []
    level_scores_list: list[float] = []

    if sc_env and sc_env.get("runs"):
        latest_run = sc_env["runs"][-1]
        level_actions_list = latest_run.get("level_actions", [])
        level_scores_list = latest_run.get("level_scores", [])

    for level_idx in range(world.win_levels):
        actions = (
            level_actions_list[level_idx]
            if level_idx < len(level_actions_list)
            else 0
        )
        sc_score = (
            level_scores_list[level_idx]
            if level_idx < len(level_scores_list)
            else 0.0
        )
        bl = baselines[level_idx] if level_idx < len(baselines) else 100
        completed = actions > 0 and sc_score > 0
        score = rhae_level_score(actions, bl, completed)
        level_scores.append(score)
        result.levels.append(
            LevelResult(
                level_index=level_idx,
                completed=completed,
                actions_taken=actions,
                baseline_actions=bl,
                score=score,
            )
        )

    result.score = rhae_environment_score(level_scores)
    result.elapsed_seconds = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> BenchmarkResult:
    """Run the full ARC-AGI-3 benchmark on keyboard-only environments."""
    print("=" * 70)
    print("ARC-AGI-3 Benchmark: OaK example_01 agent")
    print("=" * 70)

    device = torch.device(DEVICE)
    operation_mode = _parse_operation_mode(OPERATION_MODE)

    print(f"Device: {device}")
    print(f"Planning budget: {PLANNING_BUDGET}")
    print(f"Planning warmup: {PLANNING_WARMUP}")
    print(f"ARC mode: {operation_mode.value}")
    print(f"ARC API key: {'set' if ARC_API_KEY else 'not set'}")
    print(f"Train encoder: {TRAIN_ENCODER}")
    print(f"Pretrain episodes: {PRETRAIN_EPISODES}")
    print(f"Greedy eval: {GREEDY_EVAL}")
    print()

    # Initialize arcade.
    arcade = _make_arcade(operation_mode, ARC_API_KEY)
    all_envs = arcade.get_environments()

    # Filter to keyboard-only environments (no coordinate click required).
    keyboard_envs = [
        e
        for e in all_envs
        if e.tags and "keyboard" in e.tags and "keyboard_click" not in e.tags
    ]

    if MAX_ENVS > 0:
        keyboard_envs = keyboard_envs[:MAX_ENVS]

    print(
        f"Environments: {len(keyboard_envs)} keyboard-only "
        f"(out of {len(all_envs)} total)"
    )
    print()

    # Run benchmark.
    benchmark = BenchmarkResult()
    t0 = time.monotonic()

    for i, env_info in enumerate(keyboard_envs):
        print(
            f"[{i + 1}/{len(keyboard_envs)}] "
            f"{env_info.title} ({env_info.game_id})"
        )

        env_result = _benchmark_one_environment(arcade, env_info, device)
        benchmark.environments.append(env_result)

        if env_result.skipped:
            print(f"  SKIPPED: {env_result.skip_reason}")
        else:
            completed = sum(1 for l in env_result.levels if l.completed)
            total = len(env_result.levels)
            print(
                f"  Score: {env_result.score:.4f} | "
                f"Levels: {completed}/{total} | "
                f"Actions: {env_result.total_actions} | "
                f"Time: {env_result.elapsed_seconds:.1f}s"
            )
        print()

    benchmark.elapsed_seconds = time.monotonic() - t0

    # Compute total score (mean of environment scores).
    scored_envs = [e for e in benchmark.environments if not e.skipped]
    if scored_envs:
        benchmark.total_score = (
            sum(e.score for e in scored_envs) / len(scored_envs)
        )

    # Summary.
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total environments: {len(scored_envs)}")
    print(
        f"Total RHAE score:   {benchmark.total_score:.4f} "
        f"({benchmark.total_score * 100:.2f}%)"
    )
    print(f"Total time:         {benchmark.elapsed_seconds:.1f}s")
    print()

    for env_result in benchmark.environments:
        status = "SKIP" if env_result.skipped else f"{env_result.score:.4f}"
        completed = sum(1 for l in env_result.levels if l.completed)
        total = len(env_result.levels)
        print(f"  {env_result.title:8s} {status:>8s}  levels={completed}/{total}")

    # Save results.
    _save_results(benchmark)

    arcade.close_scorecard()
    return benchmark


def _save_results(benchmark: BenchmarkResult) -> None:
    """Persist benchmark results as JSON."""
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "total_score": benchmark.total_score,
        "total_score_pct": round(benchmark.total_score * 100, 4),
        "elapsed_seconds": round(benchmark.elapsed_seconds, 1),
        "num_environments": len(benchmark.environments),
        "operation_mode": OPERATION_MODE,
        "pretrain_episodes": PRETRAIN_EPISODES,
        "train_encoder": TRAIN_ENCODER,
        "planning_budget": PLANNING_BUDGET,
        "planning_warmup": PLANNING_WARMUP,
        "greedy_eval": GREEDY_EVAL,
        "environments": [],
    }

    for env_result in benchmark.environments:
        env_data: dict[str, Any] = {
            "game_id": env_result.game_id,
            "title": env_result.title,
            "tags": env_result.tags,
            "score": round(env_result.score, 6),
            "total_actions": env_result.total_actions,
            "levels_completed": env_result.levels_completed,
            "elapsed_seconds": round(env_result.elapsed_seconds, 1),
            "skipped": env_result.skipped,
        }
        if env_result.skipped:
            env_data["skip_reason"] = env_result.skip_reason
        else:
            env_data["levels"] = [
                {
                    "level_index": l.level_index,
                    "completed": l.completed,
                    "actions_taken": l.actions_taken,
                    "baseline_actions": l.baseline_actions,
                    "score": round(l.score, 6),
                }
                for l in env_result.levels
            ]
        data["environments"].append(env_data)

    result_path = out_dir / "benchmark_results.json"
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {result_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    result = run_benchmark()
    sys.exit(0 if result.total_score >= 0 else 1)
