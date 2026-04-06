"""Ollama REST API interface for LLM-augmented perception.

Calls a local ollama instance to analyze observation samples and
recommend encoder architectures, features, and configurations.
Falls back gracefully when ollama is not available.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "qwen3.5:9b"


def _get_ollama_url() -> str:
    """Determine the ollama API URL.

    On WSL2 the ollama server typically runs on the Windows host.
    We try the standard `OLLAMA_HOST` env-var first, then fall back
    to `localhost`.
    """
    host = os.environ.get("OLLAMA_HOST", "")
    if host:
        base = host.rstrip("/")
        if not base.startswith("http"):
            base = f"http://{base}"
        return f"{base}/api/chat"
    # On WSL2 try the Windows host gateway if localhost fails
    return "http://172.26.64.1:11434/api/chat"


_SYSTEM_PROMPT = """\
You are an AI assistant that analyzes raw sensor data from an unknown environment.
Your job is to determine:
1. What type of data the observations are (numeric vector, image, text, etc.)
2. What encoder architecture would best process them (mlp, cnn, identity)
3. What meaningful features or groupings exist in the data

You must respond with ONLY valid JSON, no other text. Use this exact schema:
{
  "observation_type": "numeric_vector" | "image" | "text" | "unknown",
  "dimensions": <int or null>,
  "encoder_type": "mlp" | "cnn" | "identity",
  "features": [
    {"id": "<string>", "name": "<string>", "description": "<string>", "indices": [<int>, ...]}
  ],
  "notes": "<brief analysis>"
}
"""


def analyze_world(
    observation_samples: list[Any],
    action_discovery: dict[str, Any],
    *,
    model: str = _DEFAULT_MODEL,
    timeout: float = 120.0,
) -> dict[str, Any] | None:
    """Send observation samples and action discovery results to the LLM.

    Returns the parsed JSON configuration dict, or `None` if ollama is
    unreachable or the response cannot be parsed.
    """
    user_content = _build_user_prompt(observation_samples, action_discovery)
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "format": "json",
        }
    ).encode()

    req = urllib.request.Request(
        _get_ollama_url(),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        content = body.get("message", {}).get("content", "")
        result = json.loads(content)
        logger.info(f"LLM analysis received: {result}")
        return result
    except (urllib.error.URLError, OSError) as exc:
        logger.warning(f"Ollama not reachable ({exc}), using heuristic fallback.")
        return None
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning(
            f"Failed to parse LLM response ({exc}), using heuristic fallback."
        )
        return None


def _build_user_prompt(
    observation_samples: list[Any],
    action_discovery: dict[str, Any],
) -> str:
    """Build the user message sent to the LLM."""
    # Serialize observation samples
    formatted_samples: list[str] = []
    for i, obs in enumerate(observation_samples[:10]):
        formatted_samples.append(f"  Sample {i}: {_serialize_observation(obs)}")

    sample_block = "\n".join(formatted_samples)

    return (
        f"I have an unknown environment. Here are sample observations:\n"
        f"{sample_block}\n\n"
        f"Python type of first observation: {type(observation_samples[0]).__name__}\n"
        f"Action discovery results: {json.dumps(action_discovery)}\n\n"
        f"Analyze these observations and recommend an encoder architecture "
        f"and meaningful features. Respond with JSON only."
    )


def _serialize_observation(obs: Any) -> str:
    """Convert an observation to a string representation for the LLM prompt."""
    if isinstance(obs, np.ndarray):
        return f"ndarray(shape={obs.shape}, dtype={obs.dtype}, values={obs.tolist()})"
    return repr(obs)
