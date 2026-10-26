"""Dynamic provider resolution for model strings.

The registry turns a configured label such as ``deepseek/deepseek-v4-pro``,
``zhipu/glm-5.3``, ``ollama/qwen3-coder:14b``, or ``vllm/qwen3`` into a bound
provider client with the correct base URL and API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from axe.llm.provider import LLMProvider, UnknownProviderError
from axe.llm.providers.ollama import OllamaProvider
from axe.llm.providers.open_compat import OpenCompatProvider

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rich.console import Console

    from axe.config.settings import ProviderConfig, Settings

# Providers that run locally and need no API key.
LOCAL_PROVIDERS = {"ollama", "local_vllm", "llamacpp"}

# Vendor prefixes that route directly only when the matching key is present;
# otherwise they use the default OpenRouter slug.
_DIRECT_WHEN_KEYED = {"deepseek", "zhipu", "moonshot"}

_LOCAL_PREFIXES = {
    "ollama": "ollama",
    "vllm": "local_vllm",
    "local_vllm": "local_vllm",
    "llamacpp": "llamacpp",
    "llama.cpp": "llamacpp",
}

_STRIP_PREFIXES = {"openrouter", "openai", "fireworks", "zhipu", "moonshot"}


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A model label resolved to a provider client and API model ID."""

    requested: str
    provider_name: str
    model: str
    provider: LLMProvider

    @property
    def is_local(self) -> bool:
        return self.provider_name in LOCAL_PROVIDERS

    def availability(self) -> tuple[bool, str]:
        return self.provider.is_available()


class ProviderRegistry:
    """Resolve model labels to configured provider instances."""

    def __init__(self, settings: "Settings", console: "Console | None" = None) -> None:
        self.settings = settings
        self._console = console
        self._cache: dict[str, LLMProvider] = {}

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, model: str) -> ResolvedModel:
        """Resolve a model label into a bound provider client."""
        requested = model.strip()
        if not requested:
            raise UnknownProviderError("Model name cannot be empty.")

        provider_name, api_model = self._route(requested)
        cache_key = f"{provider_name}:{api_model}:{requested}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return ResolvedModel(requested, provider_name, api_model, cached)

        config = self.settings.providers.get(provider_name)
        if config is None:
            raise UnknownProviderError(
                f"Provider '{provider_name}' is not configured. "
                f"Add [providers.{provider_name}] to config.toml."
            )

        provider = self._build_provider(provider_name, api_model, requested, config)
        self._cache[cache_key] = provider
        return ResolvedModel(requested, provider_name, api_model, provider)

    def provider_for(self, model: str) -> LLMProvider:
        """Return just the provider instance for a model label."""
        return self.resolve(model).provider

    # Convenience aliases for callers that only need the bound client.
    def get(self, model: str) -> LLMProvider:
        return self.provider_for(model)

    def get_provider(self, model: str) -> LLMProvider:
        return self.provider_for(model)

    def resolve_provider(self, model: str) -> LLMProvider:
        return self.provider_for(model)

    def availability(self, model: str) -> tuple[bool, str]:
        """Check whether a model's provider is currently usable."""
        try:
            return self.resolve(model).availability()
        except Exception as exc:  # noqa: BLE001 - reported to the router
            return False, str(exc)

    def candidates(self, models: list[str]) -> list[ResolvedModel]:
        """Resolve every model in a tier queue, skipping unknown providers."""
        resolved: list[ResolvedModel] = []
        for model in models:
            try:
                resolved.append(self.resolve(model))
            except UnknownProviderError:
                continue
        return resolved

    # ------------------------------------------------------------------
    # Routing rules
    # ------------------------------------------------------------------

    def _route(self, model: str) -> tuple[str, str]:
        """Return ``(provider_name, api_model_id)`` for a model label."""
        lowered = model.lower()

        for prefix, provider_name in _LOCAL_PREFIXES.items():
            marker = f"{prefix}/"
            if lowered.startswith(marker):
                return provider_name, model[len(marker):]

        for prefix in _DIRECT_WHEN_KEYED:
            marker = f"{prefix}/"
            if lowered.startswith(marker):
                if self._provider_has_key(prefix):
                    return prefix, model[len(marker):]
                return self.settings.providers.default, model

        for prefix in _STRIP_PREFIXES:
            marker = f"{prefix}/"
            if lowered.startswith(marker):
                return prefix, model[len(marker):]

        # Explicit ``provider:model`` syntax (e.g. deepseek:deepseek-chat).
        # Slash-based Ollama tags are handled above, so a colon is safe here.
        if ":" in model:
            prefix, rest = model.split(":", 1)
            if rest and prefix in self.settings.providers.names():
                return prefix, rest

        # Any other configured custom provider prefix.
        if "/" in model:
            prefix, rest = model.split("/", 1)
            if rest and prefix in self.settings.providers.names():
                return prefix, rest

        return self.settings.providers.default, model

    def _provider_has_key(self, provider_name: str) -> bool:
        config = self.settings.providers.get(provider_name)
        return bool(config and config.resolved_api_key())

    # ------------------------------------------------------------------
    # Provider construction
    # ------------------------------------------------------------------

    def _build_provider(
        self,
        provider_name: str,
        api_model: str,
        requested: str,
        config: "ProviderConfig",
    ) -> LLMProvider:
        if provider_name == "ollama":
            return OllamaProvider(
                model=api_model,
                base_url=config.base_url,
                api_key=config.resolved_api_key(),
                temperature=config.temperature,
                timeout=config.timeout_seconds,
                requested_model=requested,
            )

        return OpenCompatProvider(
            name=provider_name,
            model=api_model,
            base_url=config.base_url,
            api_key=config.resolved_api_key(),
            api_key_env=config.api_key_env,
            default_headers=getattr(config, "default_headers", {}),
            temperature=config.temperature,
            timeout=config.timeout_seconds,
            requested_model=requested,
        )

    # ------------------------------------------------------------------
    # Local discovery
    # ------------------------------------------------------------------

    def list_ollama_models(self) -> list[str]:
        """Return installed Ollama models (empty if the daemon is down)."""
        config = self.settings.providers.get("ollama")
        if config is None:
            return []
        provider = OllamaProvider(
            model="",
            base_url=config.base_url,
            api_key=config.resolved_api_key(),
            timeout=config.timeout_seconds,
        )
        return provider.list_local_models(force=True)
