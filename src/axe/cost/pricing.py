"""Open-weight model pricing with KV-cache hit and cache-miss rates.

All rates are USD per 1,000,000 tokens. Local inference (Ollama, vLLM,
llama.cpp) is hard-coded to zero cost. Unknown cloud slugs use a conservative
fallback so cost tracking never crashes on a new OpenRouter model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Input, cached-input, and output rates for one model."""

    input_cost: float
    cached_input_cost: float
    output_cost: float


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Detailed cost decomposition for one model call."""

    model: str
    prompt_tokens: int
    cached_prompt_tokens: int
    uncached_prompt_tokens: int
    completion_tokens: int
    input_cost_usd: float
    cached_input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    cache_savings_usd: float


PRICING: dict[str, ModelPricing] = {
    "deepseek/deepseek-v4": ModelPricing(0.27, 0.07, 1.10),
    "deepseek/deepseek-v4-pro": ModelPricing(0.55, 0.14, 2.19),
    "zhipu/glm-5.3": ModelPricing(0.35, 0.09, 1.40),
    "zhipu/glm-5.3-flash": ModelPricing(0.06, 0.015, 0.24),
    "qwen/qwen3-coder-next": ModelPricing(0.12, 0.03, 0.36),
    "qwen/qwen3.8-27b": ModelPricing(0.15, 0.04, 0.45),
    "moonshot/kimi-k3": ModelPricing(0.50, 0.12, 2.00),
    "typesafe/jev-latest": ModelPricing(0.02, 0.005, 0.02),
}

# Conservative fallback for unlisted cloud models: never fail, never claim
# free inference for a hosted endpoint.
DEFAULT_PRICING = ModelPricing(0.50, 0.10, 2.00)
LOCAL_PRICING = ModelPricing(0.0, 0.0, 0.0)

LOCAL_PREFIXES = (
    "ollama/",
    "vllm/",
    "local_vllm/",
    "llamacpp/",
    "llama.cpp/",
)

# Direct provider API IDs (the registry strips vendor prefixes when routing
# directly) map back onto their canonical OpenRouter-style slugs.
_ALIASES: dict[str, str] = {
    "deepseek-v4": "deepseek/deepseek-v4",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "glm-5.3": "zhipu/glm-5.3",
    "glm-5.3-flash": "zhipu/glm-5.3-flash",
    "qwen3-coder-next": "qwen/qwen3-coder-next",
    "qwen3.8-27b": "qwen/qwen3.8-27b",
    "kimi-k3": "moonshot/kimi-k3",
    "jev-latest": "typesafe/jev-latest",
}

TIER_LABELS: dict[str, str] = {
    "qwen/qwen3-coder-next": "fast",
    "zhipu/glm-5.3-flash": "fast",
    "ollama/qwen3-coder:14b": "fast",
    "deepseek/deepseek-v4": "coder",
    "qwen/qwen3.8-27b": "coder",
    "deepseek/deepseek-v4-pro": "architect",
    "moonshot/kimi-k3": "architect",
    "zhipu/glm-5.3": "security",
    "typesafe/jev-latest": "jev",
    # Direct API IDs
    "deepseek-v4": "coder",
    "deepseek-v4-pro": "architect",
    "glm-5.3": "security",
    "glm-5.3-flash": "fast",
    "qwen3-coder-next": "fast",
    "qwen3.8-27b": "coder",
    "kimi-k3": "architect",
}


def _canonical_name(model: str) -> str:
    name = model.strip()
    if name.lower().startswith("openrouter/"):
        name = name[len("openrouter/"):]
    return name


def is_local_model(model: str) -> bool:
    """Return ``True`` for local-inference model labels."""
    return _canonical_name(model).lower().startswith(LOCAL_PREFIXES)


def get_pricing(model: str) -> ModelPricing:
    """Return the pricing row for a model, including local/fallback rates."""
    name = _canonical_name(model)
    if name.lower().startswith(LOCAL_PREFIXES):
        return LOCAL_PRICING

    pricing = PRICING.get(name)
    if pricing is not None:
        return pricing

    alias = _ALIASES.get(name)
    if alias is not None:
        return PRICING[alias]

    # Case-insensitive fallback for labels that differ only in casing.
    lowered = name.lower()
    for key, value in PRICING.items():
        if key.lower() == lowered:
            return value
    return DEFAULT_PRICING


def is_priced_model(model: str) -> bool:
    """Return ``True`` when a model has an explicit pricing row."""
    name = _canonical_name(model)
    if name.lower().startswith(LOCAL_PREFIXES):
        return True
    return name in PRICING or name in _ALIASES


def cost_breakdown(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> CostBreakdown:
    """Compute input, cached-input, output, and cache-savings costs."""
    prompt = max(int(prompt_tokens), 0)
    completion = max(int(completion_tokens), 0)
    cached = min(max(int(cached_prompt_tokens), 0), prompt)
    uncached = prompt - cached

    pricing = get_pricing(model)
    input_cost = (uncached * pricing.input_cost) / 1_000_000
    cached_input_cost = (cached * pricing.cached_input_cost) / 1_000_000
    output_cost = (completion * pricing.output_cost) / 1_000_000
    cache_savings = (
        cached * max(pricing.input_cost - pricing.cached_input_cost, 0.0)
    ) / 1_000_000

    return CostBreakdown(
        model=model,
        prompt_tokens=prompt,
        cached_prompt_tokens=cached,
        uncached_prompt_tokens=uncached,
        completion_tokens=completion,
        input_cost_usd=input_cost,
        cached_input_cost_usd=cached_input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + cached_input_cost + output_cost,
        cache_savings_usd=cache_savings,
    )


def cost_for(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> float:
    """Return the total USD cost for a single call."""
    return cost_breakdown(
        model,
        prompt_tokens,
        completion_tokens,
        cached_prompt_tokens,
    ).total_cost_usd
