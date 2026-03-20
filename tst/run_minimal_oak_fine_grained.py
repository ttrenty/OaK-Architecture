from __future__ import annotations

from pathlib import Path
from pprint import pprint
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.minimal_oak_fine_grained import build_minimal_agent, run_minimal_episode
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

    print("Minimal fine-grained OaK smoke run")
    pprint(trace)


if __name__ == "__main__":
    main()
