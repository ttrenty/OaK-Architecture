from __future__ import annotations

from pprint import pprint

from oak_architecture import OaKAgent
from oak_architecture.implementations.minimal_oak import (
    build_minimal_agent,
    run_minimal_episode,
)


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
            "Unexpected first-step subtasks: "
            f"{first_step['created_subtasks']!r}"
        )

    if any("action" not in step for step in trace):
        raise RuntimeError("Smoke run produced a step without an action")

    print("Minimal OaK smoke run")
    pprint(trace)


if __name__ == "__main__":
    main()
