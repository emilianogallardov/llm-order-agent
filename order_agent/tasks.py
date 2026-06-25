"""Task -> model routing.

A task describes intent ("extract an order"). The config says which model runs
it by default, which model to fall back to, and which env var can override the
choice at runtime with no code change. This is how you swap models per
environment (cheap one in CI, strong one in prod) without touching call sites.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TaskConfig:
    model: str
    fallback: Optional[str]
    max_tokens: int
    temperature: float
    description: str
    env_override: Optional[str] = None


# Order extraction is a structured-output task: low temperature, JSON mode, and a
# fast-but-capable default. The strong model is the fallback for when the cheap
# one returns malformed or low-confidence output.
TASK_CONFIGS: dict[str, TaskConfig] = {
    "order_extraction": TaskConfig(
        model="gpt-4o-mini",
        fallback="gpt-4o",
        max_tokens=2000,
        temperature=0.0,
        description="Parse a natural-language order into structured, catalog-mapped line items.",
        env_override="MODEL_ORDER_EXTRACTION",
    ),
    "clarification_drafting": TaskConfig(
        model="gpt-4o-mini",
        fallback=None,
        max_tokens=400,
        temperature=0.3,
        description="Draft a short clarifying question when an order can't be resolved deterministically.",
        env_override="MODEL_CLARIFICATION",
    ),
}


def get_task_config(task: str) -> TaskConfig:
    config = TASK_CONFIGS.get(task)
    if config is None:
        raise ValueError(
            f"Unknown task '{task}'. Known: {', '.join(sorted(TASK_CONFIGS))}"
        )

    # Runtime override: env var wins over the declared default.
    if config.env_override and os.environ.get(config.env_override):
        override = os.environ[config.env_override]
        return TaskConfig(
            model=override,
            fallback=config.fallback,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            description=config.description,
            env_override=config.env_override,
        )

    return config


def all_tasks() -> list[str]:
    return sorted(TASK_CONFIGS)
