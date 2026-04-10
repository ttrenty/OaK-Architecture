from __future__ import annotations

import json
import os

import numpy as np

from examples.example_01 import CARTPOLE_WORLD_DESCRIPTION
from examples.example_01.llm import _DEFAULT_MODEL, _get_ollama_url, analyze_world
from examples.example_01.schema import PerceptionPlan

_RUN_FLAG = "OAK_RUN_LLM_CONNECTION_TEST"
_MODEL_ENV = "OAK_LLM_MODEL"
_TIMEOUT_ENV = "OAK_LLM_TIMEOUT_SECONDS"


def _validate_response(result: PerceptionPlan) -> None:
    if not result.tensor_views:
        raise RuntimeError("LLM startup plan must include at least one tensor view")
    if not result.feature_groups:
        raise RuntimeError("LLM startup plan must include at least one feature group")
    if result.default_tensor_view not in {view.view_id for view in result.tensor_views}:
        raise RuntimeError("LLM startup plan returned an unknown default tensor view")


def main() -> None:
    if os.environ.get(_RUN_FLAG) != "1":
        print(
            "Skipping live LLM connection test. "
            f"Set {_RUN_FLAG}=1 or run `pixi run test_llm_connection`."
        )
        return

    model = os.environ.get(_MODEL_ENV, _DEFAULT_MODEL)
    timeout = float(os.environ.get(_TIMEOUT_ENV, "120"))
    target_url = _get_ollama_url()

    observation_samples = [
        np.array([0.02, -0.03, 0.01, 0.04], dtype=np.float32),
        np.array([0.05, -0.01, 0.03, 0.02], dtype=np.float32),
        np.array([-0.01, 0.02, -0.04, 0.01], dtype=np.float32),
    ]

    print(f"Testing LLM connection against {target_url}")
    print(f"Using model: {model}")

    result = analyze_world(
        CARTPOLE_WORLD_DESCRIPTION,
        observation_samples,
        model=model,
        timeout=timeout,
    )
    if result is None:
        raise RuntimeError(
            "LLM connection test failed: no parsed response was returned. "
            "Check that Ollama is running, the selected model is installed, "
            "and OLLAMA_HOST points at the correct server."
        )

    _validate_response(result)
    if not result.llm_used:
        raise RuntimeError("LLM connection test expected a live LLM-backed plan")

    print("LLM connection test passed.")
    print(json.dumps(result.to_config(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
