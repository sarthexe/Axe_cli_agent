"""Multi-provider open-weight inference engine."""

from __future__ import annotations

from axe.llm.providers.ollama import OllamaProvider
from axe.llm.providers.open_compat import OpenCompatProvider
from axe.llm.providers.registry import (
    LOCAL_PROVIDERS,
    ProviderRegistry,
    ResolvedModel,
)

__all__ = [
    "LOCAL_PROVIDERS",
    "OllamaProvider",
    "OpenCompatProvider",
    "ProviderRegistry",
    "ResolvedModel",
]
