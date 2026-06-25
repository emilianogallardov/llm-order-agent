"""The model selector: the one entry point a caller uses to resolve a task into
an executable config (model + its metadata + generation params + fallback).

    cfg = get_execution_config("order_extraction")
    cfg.model         -> "gpt-4o-mini" (or whatever the env override says)
    cfg.model_config  -> full ModelConfig (provider, pricing, capabilities)
    cfg.fallback      -> model id to retry with, or None

Routing decisions (which model, which fallback, env overrides) live in tasks.py
and models.py. This module just composes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import ModelConfig, get_model
from .tasks import get_task_config


@dataclass(frozen=True)
class ExecutionConfig:
    task: str
    model: str
    model_config: ModelConfig
    fallback: Optional[str]
    max_tokens: int
    temperature: float


def get_execution_config(task: str) -> ExecutionConfig:
    task_config = get_task_config(task)
    model_config = get_model(task_config.model)

    return ExecutionConfig(
        task=task,
        model=task_config.model,
        model_config=model_config,
        fallback=task_config.fallback,
        max_tokens=task_config.max_tokens,
        temperature=task_config.temperature,
    )


def estimate_cost(cfg: ExecutionConfig, input_tokens: int, output_tokens: int) -> float:
    """Rough USD cost for a single call, using registry pricing (per 1M tokens)."""
    p = cfg.model_config.pricing
    return round(
        (input_tokens / 1_000_000) * p.input + (output_tokens / 1_000_000) * p.output,
        6,
    )
