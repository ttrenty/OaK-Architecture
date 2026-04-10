"""Run Example 01 on CartPole-v1 in described mode.

This script uses the same training pipeline as `run_example_01_cartpole.py` but passes
a `DescribedGymWorld("CartPole-v1")` whose `description` attribute provides
observation/action space metadata directly, so discovery is skipped entirely.

Compare the output of this script with `run_example_01_cartpole.py`
to see the effect of the two approaches on training.
"""

from __future__ import annotations

from examples.example_01 import (
    CARTPOLE_WORLD_DESCRIPTION,
    DescribedGymWorld,
    run_training,
)
from oak import OaKAgent


def main() -> None:
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
            f"Reward: {reward:6.1f} | "
            f"Avg({average_window}): {avg_reward:6.1f} | "
            f"Eps: {eps:.3f} | "
            f"Options: {n_options}"
        )

    # Show the embedded world description
    desc = CARTPOLE_WORLD_DESCRIPTION
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

    world = DescribedGymWorld("CartPole-v1")
    reward_history = run_training(
        world,
        num_episodes=1000,
        average_window=average_window,
        solved_threshold=475.0,
        episode_logger=log_episode,
        verbose=True,
    )

    if not reward_history:
        raise RuntimeError("Training produced no episodes")

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

    if final_avg >= 475.0:
        print("SOLVED!")
    else:
        print(f"Not solved (avg={final_avg:.1f}), but training ran successfully.")


if __name__ == "__main__":
    main()
