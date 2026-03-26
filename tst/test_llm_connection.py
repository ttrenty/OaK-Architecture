from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.cartpole.llm import _DEFAULT_MODEL, _get_ollama_url, analyze_world

_RUN_FLAG = "OAK_RUN_LLM_CONNECTION_TEST"
_MODEL_ENV = "OAK_LLM_MODEL"
_TIMEOUT_ENV = "OAK_LLM_TIMEOUT_SECONDS"


def _validate_response(result: dict[str, object]) -> None:
    expected_keys = {
        "observation_type",
        "dimensions",
        "encoder_type",
        "features",
        "notes",
    }
    missing = sorted(expected_keys.difference(result))
    if missing:
        raise RuntimeError(f"LLM response missing expected keys: {missing}")

    encoder_type = result["encoder_type"]
    if not isinstance(encoder_type, str) or not encoder_type.strip():
        raise TypeError("LLM response field 'encoder_type' must be a non-empty string")

    features = result["features"]
    if not isinstance(features, list):
        raise TypeError("LLM response field 'features' must be a list")

    notes = result["notes"]
    if not isinstance(notes, str):
        raise TypeError("LLM response field 'notes' must be a string")


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
    action_discovery = {
        "action_type": "discrete",
        "action_n": 2,
    }

    print(f"Testing LLM connection against {target_url}")
    print(f"Using model: {model}")

    result = analyze_world(
        observation_samples,
        action_discovery,
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

    print("LLM connection test passed.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
