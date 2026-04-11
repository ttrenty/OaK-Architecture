"""Run Example 01 in described mode.

By default uses CartPole-v1, but supports any environment that has an
embedded ``WorldDescription`` via ``--env``::

    pixi run test_example_01_cartpole_described -- --env Acrobot-v1

Environment names worth distinguishing:

- ``PixelCartPole-v1``: image only, no raw state vector.
- ``MultiModalCartPole-v1``: image + raw state vector together.
"""

from __future__ import annotations

import argparse

from examples.example_01 import (
    DescribedGymWorld,
    animation_recorder_from_env,
    curve_recorder_from_env,
    run_training,
)
from examples.example_01.world_embedded import _KNOWN_WORLD_DESCRIPTIONS
from oak import OaKAgent

# Per-environment solved thresholds.
_SOLVED_THRESHOLDS: dict[str, float] = {
    "CartPole-v1": 475.0,
    "Acrobot-v1": -100.0,
}

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Example 01 in described mode. "
            "PixelCartPole-v1 is image-only; "
            "MultiModalCartPole-v1 is pixels plus raw state."
        )
    )
    parser.add_argument(
        "--env",
        default="CartPole-v1",
        choices=sorted(_KNOWN_WORLD_DESCRIPTIONS),
        help=(
            "Gymnasium environment id "
            "(PixelCartPole-v1=image only, "
            "MultiModalCartPole-v1=pixels + raw state)"
        ),
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

    # Show the embedded world description
    desc = _KNOWN_WORLD_DESCRIPTIONS[env_id]
    print("World description:")
    print(
        "  channels="
        + str(
            [
                {
                    "channel_id": channel.channel_id,
                    "kind": channel.kind,
                    "shape": channel.shape,
                }
                for channel in desc.observation_channels
            ]
        )
    )
    print(f"  action_type={desc.action_type}, action_n={desc.action_n}")
    print(f"  default_encoder_type={desc.encoder_type}")
    print(f"  feature_groups={[f['name'] for f in desc.features]}")
    print()

    env_slug = env_id.lower().replace("-", "_")
    recorder = animation_recorder_from_env(f"{env_slug}_described")
    curve_recorder = curve_recorder_from_env(
        f"{env_slug}_described",
        average_window=average_window,
    )
    if recorder is not None:
        print(f"Animations enabled: {recorder.output_dir}")
    if curve_recorder is not None:
        print(f"Training curves enabled: {curve_recorder.output_dir}")

    make_kwargs = {"render_mode": "rgb_array"} if recorder is not None else None
    world = DescribedGymWorld(env_id, make_kwargs=make_kwargs)
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
