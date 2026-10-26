"""Tests for open-weight pricing, KV-cache savings, and throughput tracking."""

from __future__ import annotations

import pytest

from axe.cost.pricing import (
    DEFAULT_PRICING,
    PRICING,
    cost_breakdown,
    cost_for,
    get_pricing,
    is_local_model,
)
from axe.cost.tracker import CostTracker
from axe.llm.provider import TokenUsage


def test_deepseek_v4_pro_cache_pricing_and_savings() -> None:
    breakdown = cost_breakdown(
        "deepseek/deepseek-v4-pro",
        prompt_tokens=1_000_000,
        completion_tokens=100_000,
        cached_prompt_tokens=400_000,
    )

    assert breakdown.uncached_prompt_tokens == 600_000
    assert breakdown.input_cost_usd == pytest.approx(0.33)
    assert breakdown.cached_input_cost_usd == pytest.approx(0.056)
    assert breakdown.output_cost_usd == pytest.approx(0.219)
    assert breakdown.total_cost_usd == pytest.approx(0.605)
    assert breakdown.cache_savings_usd == pytest.approx(0.164)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("deepseek/deepseek-v4", (0.27, 0.07, 1.10)),
        ("deepseek/deepseek-v4-pro", (0.55, 0.14, 2.19)),
        ("zhipu/glm-5.3", (0.35, 0.09, 1.40)),
        ("zhipu/glm-5.3-flash", (0.06, 0.015, 0.24)),
        ("qwen/qwen3-coder-next", (0.12, 0.03, 0.36)),
        ("qwen/qwen3.8-27b", (0.15, 0.04, 0.45)),
        ("moonshot/kimi-k3", (0.50, 0.12, 2.00)),
        ("typesafe/jev-latest", (0.02, 0.005, 0.02)),
    ],
)
def test_spec_pricing_rows(model: str, expected: tuple[float, float, float]) -> None:
    pricing = get_pricing(model)

    assert (
        pricing.input_cost,
        pricing.cached_input_cost,
        pricing.output_cost,
    ) == expected


def test_local_models_are_free_and_have_no_cache_savings() -> None:
    assert is_local_model("ollama/qwen3-coder:14b") is True
    assert is_local_model("vllm/qwen3.8-27b") is True
    assert cost_for("ollama/qwen3-coder:14b", 1_000_000, 1_000_000, 500_000) == 0.0

    breakdown = cost_breakdown(
        "vllm/qwen3.8-27b",
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        cached_prompt_tokens=500_000,
    )
    assert breakdown.total_cost_usd == 0.0
    assert breakdown.cache_savings_usd == 0.0


def test_unknown_cloud_model_uses_graceful_fallback_rate() -> None:
    breakdown = cost_breakdown(
        "vendor/brand-new-model",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        cached_prompt_tokens=500_000,
    )

    assert get_pricing("vendor/brand-new-model") is DEFAULT_PRICING
    assert breakdown.input_cost_usd == pytest.approx(0.25)
    assert breakdown.cached_input_cost_usd == pytest.approx(0.05)
    assert breakdown.output_cost_usd == pytest.approx(2.00)
    assert breakdown.total_cost_usd == pytest.approx(2.30)
    assert breakdown.cache_savings_usd == pytest.approx(0.20)


def test_direct_api_ids_alias_to_canonical_pricing() -> None:
    assert get_pricing("deepseek-v4-pro") is PRICING["deepseek/deepseek-v4-pro"]
    assert get_pricing("glm-5.3") is PRICING["zhipu/glm-5.3"]


def test_cached_tokens_are_clamped_to_prompt_tokens() -> None:
    breakdown = cost_breakdown(
        "deepseek/deepseek-v4",
        prompt_tokens=100,
        completion_tokens=0,
        cached_prompt_tokens=500,
    )

    assert breakdown.cached_prompt_tokens == 100
    assert breakdown.uncached_prompt_tokens == 0
    assert breakdown.input_cost_usd == 0.0


def test_tracker_accumulates_cache_reasoning_and_throughput() -> None:
    tracker = CostTracker(session_budget=10.0)
    tracker.record(
        "deepseek/deepseek-v4-pro",
        TokenUsage(
            prompt_tokens=1_000_000,
            completion_tokens=100_000,
            cached_prompt_tokens=400_000,
            reasoning_tokens=40_000,
            latency_ms=1000.0,
            total_ms=1200.0,
            ttft_ms=200.0,
            tokens_per_sec=100.0,
        ),
        tier="architect",
        provider="openrouter",
    )
    tracker.record(
        "ollama/qwen3-coder:14b",
        TokenUsage(
            prompt_tokens=1_000,
            completion_tokens=500,
            reasoning_tokens=10,
            latency_ms=500.0,
            total_ms=600.0,
            ttft_ms=100.0,
            tokens_per_sec=20.0,
        ),
        tier="fast",
        provider="ollama",
    )

    assert tracker.total_prompt_tokens == 1_001_000
    assert tracker.total_cached_tokens == 400_000
    assert tracker.total_completion_tokens == 100_500
    assert tracker.total_reasoning_tokens == 40_010
    assert tracker.total_cost() == pytest.approx(0.605)
    assert tracker.cache_savings_usd == pytest.approx(0.164)
    assert tracker.cache_hit_ratio == pytest.approx(400_000 / 1_001_000 * 100)
    assert tracker.avg_tokens_per_sec == pytest.approx(100_500 / 1025)
    assert tracker.per_tier_calls == {"architect": 1, "fast": 1}
    assert tracker.per_provider_calls == {"openrouter": 1, "ollama": 1}
    assert tracker.total_calls() == 2
    assert tracker._records[0].ttft_ms == 200.0


def test_tracker_accepts_deepseek_raw_cache_hit_field() -> None:
    tracker = CostTracker(session_budget=10.0)
    record = tracker.record(
        "deepseek/deepseek-v4",
        {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 0,
            "prompt_cache_hit_tokens": 1_000_000,
        },
        tier="coder",
    )

    assert record.cached_prompt_tokens == 1_000_000
    assert record.cost_usd == pytest.approx(0.07)
    assert record.cache_savings_usd == pytest.approx(0.20)


def test_tracker_extracts_reasoning_from_nested_usage_details() -> None:
    tracker = CostTracker()
    record = tracker.record(
        "zhipu/glm-5.3",
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 25},
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
        tier="security",
    )

    assert record.cached_prompt_tokens == 25
    assert record.reasoning_tokens == 7
    assert tracker.per_tier_calls == {"security": 1}


def test_summary_and_overview_expose_required_fields() -> None:
    tracker = CostTracker()
    tracker.record(
        "deepseek/deepseek-v4-pro",
        TokenUsage(
            prompt_tokens=1_000_000,
            completion_tokens=100_000,
            cached_prompt_tokens=400_000,
            reasoning_tokens=40_000,
            tokens_per_sec=100.0,
        ),
        tier="architect",
        provider="openrouter",
    )

    table = tracker.summary_table()
    overview = tracker.overview_text().plain

    assert table.row_count >= 2
    assert "Cache hit:" in overview
    assert "Saved:" in overview
    assert "Reasoning:" in overview
    assert "Avg throughput:" in overview
    assert "Tier calls:" in overview
    assert "Providers:" in overview
