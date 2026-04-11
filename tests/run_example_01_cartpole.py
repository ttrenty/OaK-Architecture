"""Run Example 01 in discovery mode.

By default uses CartPole-v1, but supports any classic-control Gymnasium
environment via ``--env``::

    pixi run test_example_01_cartpole -- --env Acrobot-v1
"""

from __future__ import annotations

import argparse

from examples.example_01 import (
    GymWorld,
    animation_recorder_from_env,
    curve_recorder_from_env,
    run_training,
)
from oak import OaKAgent

# Per-environment solved thresholds (higher is better for CartPole,
# less negative is better for Acrobot).
_SOLVED_THRESHOLDS: dict[str, float] = {
    "CartPole-v1": 475.0,
    "Acrobot-v1": -100.0,
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Example 01 in discovery mode.")
    parser.add_argument(
        "--env",
        default="CartPole-v1",
        help="Gymnasium environment id (default: CartPole-v1)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes to run (default: 1000)",
    )
    args = parser.parse_args()
    env_id: str = args.env
    num_episodes: int = args.episodes

    # Ensure pixel/multimodal envs are registered with gymnasium.
    if "Pixel" in env_id or "MultiModal" in env_id:
        import examples.example_01.world_pixel  # noqa: F401

    solved_threshold = _SOLVED_THRESHOLDS.get(env_id)
    average_window = 100

    def log_episode(
        episode: int,
        reward: float,
        avg_reward: float,
        agent: OaKAgent,
    ) -> None:
        if episode % 10 != 0:
            return

        eps = getattr(agent.reactive_policy, "epsilon", 0.0)
        n_options = len(getattr(agent.reactive_policy, "_options", {}))
        print(
            f"  Episode {episode:4d} | "
            f"Reward: {reward:7.1f} | "
            f"Avg({average_window}): {avg_reward:7.1f} | "
            f"Eps: {eps:.3f} | "
            f"Options: {n_options}"
        )

    def combined_logger(
        episode: int,
        reward: float,
        avg_reward: float,
        agent: OaKAgent,
    ) -> None:
        if curve_recorder is not None:
            curve_recorder.log_episode(episode, reward, avg_reward, agent)
        log_episode(episode, reward, avg_reward, agent)

    env_slug = env_id.lower().replace("-", "_")
    recorder = animation_recorder_from_env(f"{env_slug}_discovery")
    curve_recorder = curve_recorder_from_env(
        f"{env_slug}_discovery",
        average_window=average_window,
    )
    if recorder is not None:
        print(f"Animations enabled: {recorder.output_dir}")
    if curve_recorder is not None:
        print(f"Training curves enabled: {curve_recorder.output_dir}")

    make_kwargs = {"render_mode": "rgb_array"} if recorder is not None else None
    world = GymWorld(env_id, make_kwargs=make_kwargs)

    reward_history = run_training(
        world,
        num_episodes=num_episodes,
        average_window=average_window,
        solved_threshold=solved_threshold,
        episode_logger=combined_logger,
        episode_trace_logger=recorder,
        trace_selector=recorder.should_capture if recorder is not None else None,
        capture_rendered_frames=recorder is not None,
        verbose=True,
    )

    if not reward_history:
        raise RuntimeError("Training produced no episodes")
    if curve_recorder is not None:
        curve_recorder.save()

    n = len(reward_history)
    if n >= 20:
        early_avg = sum(reward_history[:10]) / 10
        late_avg = sum(reward_history[-10:]) / 10
        print(f"\nEarly avg (first 10): {early_avg:.1f}")
        print(f"Late avg (last 10):   {late_avg:.1f}")
        if late_avg > early_avg:
            print("Learning detected: late performance > early performance")
        else:
            print("Warning: no clear learning detected (may need more episodes)")

    final_window = (
        reward_history[-average_window:] if n >= average_window else reward_history
    )
    final_avg = sum(final_window) / len(final_window)
    print(f"Final avg: {final_avg:.1f}")

    if solved_threshold is not None and final_avg >= solved_threshold:
        print("SOLVED!")
    else:
        print(f"Not solved (avg={final_avg:.1f}), but training ran successfully.")


if __name__ == "__main__":
    main()
