"""Token pricing table for supported OpenAI models.

Each entry maps a model name to a (input_cost, output_cost) tuple
where costs are expressed in USD per 1,000,000 tokens.
"""

from __future__ import annotations

# (input $/1M tokens, output $/1M tokens)
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1":      (2.00, 8.00),
    "o3-mini":      (1.10, 4.40),
}

# Tier ordering used by the router for display purposes
TIER_LABELS: dict[str, str] = {
    "gpt-4.1-mini": "T1",
    "gpt-4.1":      "T2",
    "o3-mini":      "T3",
}


def cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the USD cost for a single call given token counts.

    Falls back to $0 for unknown models so cost tracking never hard-crashes.
    """
    rates = PRICING.get(model)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000
