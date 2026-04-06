"""Integration tests for Example 01 and the smoke examples.

Verifies that:
- All World implementations satisfy the World protocol
- build_agent() produces a valid OaKAgent from a config dict
- agent.train(world) runs the episode loop correctly
- run_training() works in both discovery and embedded modes
- Smoke tests still pass after refactoring
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from examples.example_01 import runner as example_01_runner
from examples.example_01 import DescribedGymWorld, GymWorld, build_agent, run_training
from examples.smoke.minimal_oak import MinimalWorld
from examples.smoke.minimal_oak import run_minimal_episode as run_minimal_smoke
from examples.smoke.minimal_oak_fine_grained import run_minimal_episode as run_minimal_fine_grained
from oak import OaKAgent
from oak.interfaces import World


def test_world_protocol() -> None:
    """All World implementations satisfy the World protocol."""
    world_builders: list[tuple[str, Callable[[], object]]] = [
        ("GymWorld", lambda: GymWorld("CartPole-v1")),
        ("DescribedGymWorld", lambda: DescribedGymWorld("CartPole-v1")),
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
        test_agent_train_embedded,
        test_run_training_embedded,
        test_run_training_discovery,
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
