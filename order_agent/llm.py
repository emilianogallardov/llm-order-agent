"""LLM clients behind one interface, dispatched by the model selector.

`LLMClient.complete_json(execution_config, system, user)` returns parsed JSON.
The provider (OpenAI vs Anthropic) is read from the selected model's metadata, so
call sites never branch on provider. Provider SDKs are imported lazily, which
keeps the package importable (and the eval suite runnable) with zero installs.

`MockClient` returns canned, deterministic extractions keyed by the order text.
It exists so the eval suite asserts on the deterministic pipeline without network
flakiness or API spend. The exact same prompts run live through the real client.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

from .selector import ExecutionConfig


class LLMClient:
    """Routes a structured-output call to the right provider by model metadata."""

    def complete_json(self, cfg: ExecutionConfig, system: str, user: str) -> dict:
        provider = cfg.model_config.provider
        if provider == "openai":
            return self._openai(cfg, system, user)
        if provider == "anthropic":
            return self._anthropic(cfg, system, user)
        raise ValueError(f"No client wired for provider '{provider}'")

    def _openai(self, cfg: ExecutionConfig, system: str, user: str) -> dict:
        from openai import OpenAI  # lazy import

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return json.loads(resp.choices[0].message.content)

    def _anthropic(self, cfg: ExecutionConfig, system: str, user: str) -> dict:
        from anthropic import Anthropic  # lazy import

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": user + "\n\nReturn only the JSON object, nothing else.",
                }
            ],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return _extract_json(text)


class MockClient(LLMClient):
    """Deterministic stand-in for the model. Used by the eval suite and offline runs."""

    def __init__(self, responder: Callable[[str, str], dict]):
        self._responder = responder

    def complete_json(self, cfg: ExecutionConfig, system: str, user: str) -> dict:
        return self._responder(system, user)


def _extract_json(text: str) -> dict:
    """Best-effort JSON recovery if a model wraps output in prose or fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


_PROVIDER_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def provider_key_available(provider: str) -> bool:
    """True if the env key for this provider is set. Used to skip a selected model
    whose provider isn't configured (and fall back to one that is)."""
    env = _PROVIDER_KEY_ENV.get(provider)
    return bool(env and os.environ.get(env))


def default_client() -> Optional[LLMClient]:
    """A live client if any provider key is present, else None (use mock)."""
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        return LLMClient()
    return None
