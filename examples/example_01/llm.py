"""Startup-time LLM planner for Example 01 perception."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, cast

from .schema import (
    PerceptionPlan,
    SemanticFieldPlan,
    TensorViewPlan,
    WorldDescription,
)
from .startup import build_heuristic_perception_plan, serialize_observation_sample

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "qwen3.5:9b"


def _get_ollama_url() -> str:
    """Determine the ollama API URL."""
    host = os.environ.get("OLLAMA_HOST", "")
    if host:
        base = host.rstrip("/")
        if not base.startswith("http"):
            base = f"http://{base}"
        return f"{base}/api/chat"
    return "http://172.26.64.1:11434/api/chat"


_SYSTEM_PROMPT = """\
You are an AI assistant planning perception for a reinforcement-learning agent.

You receive:
- a structured world description with observation channels and actions
- optional raw observation samples

Return ONLY valid JSON using this schema:
{
  "default_tensor_view": "<string>",
  "tensor_views": [
    {
      "view_id": "<string>",
      "source_channel": "<string>",
      "encoder_type": "identity" | "mlp" | "cnn",
      "latent_dim": <int or null>,
      "description": "<string>"
    }
  ],
  "feature_groups": [
    {
      "field_id": "<string>",
      "name": "<string>",
      "source_channel": "<string>",
      "description": "<string>",
      "selector_names": ["<string>", ...],
      "selector_indices": [<int>, ...]
    }
  ],
  "notes": "<brief analysis>"
}
"""


def analyze_world(
    world_description: WorldDescription,
    observation_samples: list[Any] | None = None,
    *,
    model: str = _DEFAULT_MODEL,
    timeout: float = 120.0,
) -> PerceptionPlan | None:
    """Attempt an LLM startup plan. Returns `None` on transport/parse failure."""
    user_content = _build_user_prompt(world_description, observation_samples or [])
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
        plan = _plan_from_llm_result(world_description, result)
        logger.info("LLM startup plan received")
        return plan
    except (urllib.error.URLError, OSError) as exc:
        logger.warning(f"Ollama not reachable ({exc}), using heuristic startup plan.")
        return None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning(f"Failed to parse LLM startup plan ({exc}), using heuristic fallback.")
        return None


def plan_perception(
    world_description: WorldDescription,
    observation_samples: list[Any] | None = None,
    *,
    model: str = _DEFAULT_MODEL,
    timeout: float = 120.0,
) -> PerceptionPlan:
    """Build a startup-time perception plan with deterministic fallback."""
    llm_plan = analyze_world(
        world_description,
        observation_samples,
        model=model,
        timeout=timeout,
    )
    if llm_plan is not None:
        return llm_plan
    return build_heuristic_perception_plan(
        world_description,
        notes="Heuristic startup perception plan (LLM unavailable).",
        llm_used=False,
        metadata={"planner": "heuristic"},
    )


def _build_user_prompt(
    world_description: WorldDescription,
    observation_samples: list[Any],
) -> str:
    formatted_samples = [
        f"  Sample {index}: {serialize_observation_sample(sample)}"
        for index, sample in enumerate(observation_samples[:10])
    ]
    sample_block = "\n".join(formatted_samples) or "  <no samples provided>"

    return (
        "I have an RL world and need a startup-time perception plan.\n\n"
        f"World description JSON:\n{json.dumps(world_description.to_config(), indent=2)}\n\n"
        f"Observation samples:\n{sample_block}\n\n"
        "Please propose grouped semantic fields and tensor views for downstream learners."
    )


def _plan_from_llm_result(
    world_description: WorldDescription,
    result: dict[str, Any],
) -> PerceptionPlan:
    tensor_views_raw = result.get("tensor_views")
    feature_groups_raw = result.get("feature_groups")
    default_tensor_view = str(result.get("default_tensor_view", "")).strip()
    notes = str(result.get("notes", "")).strip()

    if not isinstance(tensor_views_raw, list) or not tensor_views_raw:
        raise ValueError("LLM response must include a non-empty tensor_views list")
    if not isinstance(feature_groups_raw, list) or not feature_groups_raw:
        raise ValueError("LLM response must include a non-empty feature_groups list")

    channel_ids = {channel.channel_id for channel in world_description.observation_channels}
    tensor_views: list[TensorViewPlan] = []
    for item in tensor_views_raw:
        if not isinstance(item, dict):
            continue
        source_channel = str(item.get("source_channel", "")).strip()
        if source_channel not in channel_ids:
            continue
        channel = next(
            channel
            for channel in world_description.observation_channels
            if channel.channel_id == source_channel
        )
        view_id = str(item.get("view_id", source_channel)).strip() or source_channel
        encoder_type = str(item.get("encoder_type", world_description.encoder_type)).strip()
        if encoder_type not in {"identity", "mlp", "cnn"}:
            continue
        latent_dim_raw = item.get("latent_dim")
        latent_dim = int(latent_dim_raw) if isinstance(latent_dim_raw, int) else None
        tensor_views.append(
            TensorViewPlan(
                view_id=view_id,
                source_channel=source_channel,
                encoder_type=cast(Any, encoder_type),
                input_shape=channel.shape,
                input_dim=channel.input_dim(),
                input_channels=(
                    int(channel.shape[-1])
                    if channel.shape is not None and len(channel.shape) >= 3
                    else 1
                ),
                latent_dim=latent_dim,
                description=str(item.get("description", "")).strip(),
            )
        )

    feature_groups: list[SemanticFieldPlan] = []
    for item in feature_groups_raw:
        if not isinstance(item, dict):
            continue
        source_channel = str(item.get("source_channel", "")).strip()
        if source_channel not in channel_ids:
            continue
        field_id = str(item.get("field_id", "")).strip()
        name = str(item.get("name", field_id)).strip()
        if not field_id or not name:
            continue
        selector_names_raw = item.get("selector_names", [])
        selector_indices_raw = item.get("selector_indices", [])
        selector_names = tuple(
            str(value)
            for value in selector_names_raw
            if isinstance(value, str) and value.strip()
        )
        selector_indices = tuple(
            int(value)
            for value in selector_indices_raw
            if isinstance(value, int)
        )
        feature_groups.append(
            SemanticFieldPlan(
                field_id=field_id,
                name=name,
                source_channel=source_channel,
                description=str(item.get("description", "")).strip(),
                selector_names=selector_names,
                selector_indices=selector_indices,
            )
        )

    if not tensor_views or not feature_groups:
        raise ValueError("LLM response did not yield any valid tensor views or feature groups")

    chosen_default = default_tensor_view if default_tensor_view else tensor_views[0].view_id
    if chosen_default not in {view.view_id for view in tensor_views}:
        chosen_default = tensor_views[0].view_id

    return PerceptionPlan(
        world_description=world_description,
        feature_groups=tuple(feature_groups),
        tensor_views=tuple(tensor_views),
        default_tensor_view=chosen_default,
        notes=notes,
        llm_used=True,
        metadata={"planner": "llm"},
    )
