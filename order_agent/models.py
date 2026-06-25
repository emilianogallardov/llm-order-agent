"""Model registry: the single source of truth for every LLM this app can call.

Pattern: a declarative registry (model id -> metadata) plus task-based routing.
Callers say *what they want to do* (a task), not *which model to use*. Picking
the model, the fallback, and any runtime override lives here, in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Pricing:
    """USD per 1M tokens."""

    input: float
    output: float
    cached: Optional[float] = None


@dataclass(frozen=True)
class Capabilities:
    json_mode: bool = True
    function_calling: bool = True
    vision: bool = False
    max_output_tokens: int = 4096
    context_window: int = 128_000


@dataclass(frozen=True)
class Performance:
    speed: str = "medium"        # fast | medium | slow
    quality: str = "good"        # basic | good | excellent
    typical_latency_ms: int = 2000


@dataclass(frozen=True)
class ModelConfig:
    id: str
    name: str
    provider: str                # openai | anthropic
    pricing: Pricing
    capabilities: Capabilities = field(default_factory=Capabilities)
    performance: Performance = field(default_factory=Performance)


# --- Registry -----------------------------------------------------------------
# Add a model here once; every task config and the LLM client can reference it.
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o-mini": ModelConfig(
        id="gpt-4o-mini",
        name="GPT-4o mini",
        provider="openai",
        pricing=Pricing(input=0.15, output=0.60, cached=0.075),
        capabilities=Capabilities(json_mode=True, max_output_tokens=16_384),
        performance=Performance(speed="fast", quality="good", typical_latency_ms=1200),
    ),
    "gpt-4o": ModelConfig(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        pricing=Pricing(input=2.50, output=10.0, cached=1.25),
        capabilities=Capabilities(json_mode=True, vision=True, max_output_tokens=16_384),
        performance=Performance(speed="medium", quality="excellent", typical_latency_ms=2800),
    ),
    "claude-haiku-4-5": ModelConfig(
        id="claude-haiku-4-5",
        name="Claude Haiku 4.5",
        provider="anthropic",
        pricing=Pricing(input=1.0, output=5.0),
        capabilities=Capabilities(json_mode=True, max_output_tokens=8192),
        performance=Performance(speed="fast", quality="good", typical_latency_ms=1500),
    ),
    "claude-sonnet-4-6": ModelConfig(
        id="claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        provider="anthropic",
        pricing=Pricing(input=3.0, output=15.0, cached=0.30),
        capabilities=Capabilities(json_mode=True, vision=True, max_output_tokens=8192),
        performance=Performance(speed="medium", quality="excellent", typical_latency_ms=3200),
    ),
}


def get_model(model_id: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError:
        raise ValueError(
            f"Unknown model '{model_id}'. Known: {', '.join(sorted(MODEL_REGISTRY))}"
        )


def has_model(model_id: str) -> bool:
    return model_id in MODEL_REGISTRY


def all_model_ids() -> list[str]:
    return sorted(MODEL_REGISTRY)
