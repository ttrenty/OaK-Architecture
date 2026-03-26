from __future__ import annotations

from pathlib import Path
from pprint import pprint
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.smoke.minimal_oak import (
    build_minimal_agent,
    run_minimal_episode,
    run_minimal_training,
)
from oak_architecture import OaKAgent


def main() -> None:

    agent = build_minimal_agent()
    if not isinstance(agent, OaKAgent):
        raise TypeError("build_minimal_agent() did not return an OaKAgent")

    trace = run_minimal_episode(horizon=5)
    if not trace:
        raise RuntimeError("Smoke run produced an empty trace")

    first_step = trace[0]
    if first_step["created_subtasks"] != ["subtask:observation"]:
        raise RuntimeError(
            "Unexpected first-step subtasks: " f"{first_step['created_subtasks']!r}"
        )

    if any("action" not in step for step in trace):
        raise RuntimeError("Smoke run produced a step without an action")

    if any("subjective_state" not in step for step in trace):
        raise RuntimeError("Smoke run produced a step without a subjective_state")

    rewards = run_minimal_training(num_episodes=3, horizon=5)
    if rewards != [2.0, 2.0, 2.0]:
        raise RuntimeError(f"Unexpected training rewards: {rewards!r}")

    early_stop_rewards = run_minimal_training(
        num_episodes=5,
        horizon=5,
        average_window=2,
        solved_threshold=2.0,
    )
    if early_stop_rewards != [2.0, 2.0]:
        raise RuntimeError(
            "Unexpected early-stop training rewards: "
            f"{early_stop_rewards!r}"
        )

    print("Minimal OaK smoke run")
    pprint(
        {
            "trace": trace,
            "rewards": rewards,
            "early_stop_rewards": early_stop_rewards,
        }
    )


if __name__ == "__main__":
    main()
