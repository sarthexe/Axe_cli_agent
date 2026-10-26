"""Session cost, KV-cache, reasoning, and throughput tracking."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from axe.cost.pricing import TIER_LABELS, cost_breakdown
from axe.llm.provider import TokenUsage


def _nested_int(raw: Mapping[str, Any], parent: str, child: str) -> int:
    nested = raw.get(parent)
    if nested is not None and not isinstance(nested, Mapping):
        dump = getattr(nested, "model_dump", None)
        if callable(dump):
            try:
                nested = dump()
            except Exception:  # pragma: no cover - defensive
                nested = None
    if isinstance(nested, Mapping):
        try:
            return int(nested.get(child, 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _as_usage(usage: TokenUsage | Mapping[str, Any] | Any | None) -> TokenUsage:
    """Normalize a TokenUsage, raw usage dict, or usage-like object."""
    if isinstance(usage, TokenUsage):
        return usage
    if usage is None:
        return TokenUsage()

    if isinstance(usage, Mapping):
        raw = dict(usage)
    else:
        raw = {}
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "cached_prompt_tokens",
            "prompt_cache_hit_tokens",
            "cache_read_input_tokens",
            "reasoning_tokens",
            "latency_ms",
            "total_ms",
            "ttft_ms",
            "tokens_per_sec",
            "prompt_tokens_details",
            "completion_tokens_details",
        ):
            if hasattr(usage, name):
                raw[name] = getattr(usage, name)

    cached = int(
        raw.get("cached_prompt_tokens")
        or raw.get("prompt_cache_hit_tokens")
        or _nested_int(raw, "prompt_tokens_details", "cached_tokens")
        or 0
    )
    reasoning = int(
        raw.get("reasoning_tokens")
        or _nested_int(raw, "completion_tokens_details", "reasoning_tokens")
        or 0
    )
    return TokenUsage(
        prompt_tokens=int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0),
        completion_tokens=int(
            raw.get("completion_tokens") or raw.get("output_tokens") or 0
        ),
        cached_prompt_tokens=cached,
        reasoning_tokens=reasoning,
        latency_ms=float(raw.get("latency_ms") or 0.0),
        total_ms=float(raw.get("total_ms") or 0.0),
        ttft_ms=float(raw.get("ttft_ms") or 0.0),
        tokens_per_sec=float(raw.get("tokens_per_sec") or 0.0),
    )


@dataclass
class CallRecord:
    """A single LLM call's billing and throughput snapshot."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    tier: str = "unknown"
    provider: str = ""
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    cache_savings_usd: float = 0.0
    latency_ms: float = 0.0
    total_ms: float = 0.0
    ttft_ms: float = 0.0
    tokens_per_sec: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CostTracker:
    """Track token usage, KV-cache savings, reasoning, and throughput."""

    def __init__(
        self,
        session_budget: float = 1.00,
        alert_at_percent: int = 80,
        log_file: str | None = None,
    ) -> None:
        self.session_budget = session_budget
        self.alert_at_percent = alert_at_percent
        self._log_path: Path | None = Path(log_file).expanduser() if log_file else None
        self._records: list[CallRecord] = []
        self.tokens_saved_by_ast: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def add_ast_tokens_saved(self, tokens: int) -> None:
        """Record tokens avoided by AST/symbol context compression."""
        if tokens > 0:
            self.tokens_saved_by_ast += int(tokens)

    def record_ast_savings(self, tokens: int) -> None:
        """Backward-friendly alias for :meth:`add_ast_tokens_saved`."""
        self.add_ast_tokens_saved(tokens)

    def record(
        self,
        model: str,
        usage: TokenUsage | Mapping[str, Any] | Any | None = None,
        tier: str = "",
        provider: str = "",
    ) -> CallRecord:
        """Record one completed call and return its :class:`CallRecord`."""
        normalized = _as_usage(usage)
        breakdown = cost_breakdown(
            model,
            normalized.prompt_tokens,
            normalized.completion_tokens,
            normalized.cached_prompt_tokens,
        )
        resolved_tier = tier or TIER_LABELS.get(model, "unknown")
        record = CallRecord(
            model=model,
            prompt_tokens=normalized.prompt_tokens,
            completion_tokens=normalized.completion_tokens,
            cost_usd=breakdown.total_cost_usd,
            tier=resolved_tier,
            provider=provider,
            cached_prompt_tokens=normalized.cached_prompt_tokens,
            reasoning_tokens=normalized.reasoning_tokens,
            cache_savings_usd=breakdown.cache_savings_usd,
            latency_ms=normalized.latency_ms,
            total_ms=normalized.total_ms,
            ttft_ms=normalized.ttft_ms,
            tokens_per_sec=normalized.tokens_per_sec,
        )
        self._records.append(record)
        if self._log_path:
            self._append_log(record)
        return record

    # ------------------------------------------------------------------
    # Session totals
    # ------------------------------------------------------------------

    def total_cost(self) -> float:
        return sum(record.cost_usd for record in self._records)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(record.prompt_tokens for record in self._records)

    @property
    def total_cached_tokens(self) -> int:
        return sum(record.cached_prompt_tokens for record in self._records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(record.completion_tokens for record in self._records)

    @property
    def total_reasoning_tokens(self) -> int:
        return sum(record.reasoning_tokens for record in self._records)

    @property
    def cache_savings_usd(self) -> float:
        return sum(record.cache_savings_usd for record in self._records)

    @property
    def cache_hit_ratio(self) -> float:
        prompt = self.total_prompt_tokens
        if prompt <= 0:
            return 0.0
        return (self.total_cached_tokens / prompt) * 100.0

    @property
    def avg_tokens_per_sec(self) -> float:
        """Completion-token-weighted average throughput across calls."""
        weighted = [
            (record.completion_tokens, record.tokens_per_sec)
            for record in self._records
            if record.completion_tokens > 0 and record.tokens_per_sec > 0
        ]
        if not weighted:
            return 0.0
        total_tokens = sum(tokens for tokens, _ in weighted)
        total_seconds = sum(tokens / tps for tokens, tps in weighted)
        if total_seconds <= 0:
            return 0.0
        return total_tokens / total_seconds

    @property
    def per_tier_calls(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records:
            counts[record.tier] = counts.get(record.tier, 0) + 1
        return counts

    @property
    def per_provider_calls(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records:
            key = record.provider or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def avg_latency_ms(self) -> float:
        if not self._records:
            return 0.0
        return sum(record.latency_ms for record in self._records) / len(self._records)

    @property
    def avg_ttft_ms(self) -> float:
        ttfts = [record.ttft_ms for record in self._records if record.ttft_ms > 0]
        if not ttfts:
            return 0.0
        return sum(ttfts) / len(ttfts)

    def total_calls(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Budget checks
    # ------------------------------------------------------------------

    def budget_percent(self) -> float:
        if self.session_budget <= 0:
            return 100.0
        return (self.total_cost() / self.session_budget) * 100.0

    def over_budget(self) -> bool:
        return self.total_cost() >= self.session_budget

    def near_budget(self) -> bool:
        return self.budget_percent() >= self.alert_at_percent

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def per_model_stats(self) -> dict[str, dict[str, Any]]:
        """Aggregate stats by model, including tier, cache, and throughput."""
        stats: dict[str, dict[str, Any]] = {}
        for record in self._records:
            entry = stats.setdefault(
                record.model,
                {
                    "calls": 0,
                    "tier": record.tier,
                    "prompt_tokens": 0,
                    "cached_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "cost_usd": 0.0,
                    "cache_savings_usd": 0.0,
                    "latency_ms": 0.0,
                    "tokens_per_sec": 0.0,
                    "_tps_tokens": 0,
                },
            )
            if entry["tier"] != record.tier:
                entry["tier"] = "mixed"
            entry["calls"] += 1
            entry["prompt_tokens"] += record.prompt_tokens
            entry["cached_tokens"] += record.cached_prompt_tokens
            entry["completion_tokens"] += record.completion_tokens
            entry["reasoning_tokens"] += record.reasoning_tokens
            entry["cost_usd"] += record.cost_usd
            entry["cache_savings_usd"] += record.cache_savings_usd
            entry["latency_ms"] += record.latency_ms
            if record.tokens_per_sec > 0 and record.completion_tokens > 0:
                entry["tokens_per_sec"] += record.tokens_per_sec * record.completion_tokens
                entry["_tps_tokens"] += record.completion_tokens

        for entry in stats.values():
            tokens = entry.pop("_tps_tokens")
            entry["tokens_per_sec"] = (
                entry["tokens_per_sec"] / tokens if tokens > 0 else 0.0
            )
        return stats

    def per_model_counts(self) -> dict[str, int]:
        return {model: stats["calls"] for model, stats in self.per_model_stats().items()}

    def summary_table(self, console: Console | None = None) -> Table:
        """Build the per-model cost/cache/reasoning/throughput table."""
        del console  # Kept for API compatibility
        table = Table(
            title="Session Cost Summary",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=False,
        )
        table.add_column("Model", style="bold")
        table.add_column("Tier", justify="center")
        table.add_column("Calls", justify="right")
        table.add_column("Input", justify="right")
        table.add_column("Cached", justify="right")
        table.add_column("Reason", justify="right")
        table.add_column("Output", justify="right")
        table.add_column("Cost", justify="right", style="green")
        table.add_column("Saved", justify="right", style="green")
        table.add_column("Tok/s", justify="right")

        stats = self.per_model_stats()
        total_calls = 0
        total_input = 0
        total_cached = 0
        total_reasoning = 0
        total_output = 0
        total_cost = 0.0
        total_saved = 0.0

        for model, entry in stats.items():
            cached = entry["cached_tokens"]
            prompt = entry["prompt_tokens"]
            cached_pct = (cached / prompt * 100.0) if prompt else 0.0
            table.add_row(
                model,
                str(entry["tier"]),
                str(entry["calls"]),
                f"{prompt:,}",
                f"{cached:,} ({cached_pct:.0f}%)" if cached else "0",
                f"{entry['reasoning_tokens']:,}",
                f"{entry['completion_tokens']:,}",
                f"${entry['cost_usd']:.4f}",
                f"${entry['cache_savings_usd']:.4f}",
                f"{entry['tokens_per_sec']:.1f}",
            )
            total_calls += entry["calls"]
            total_input += prompt
            total_cached += cached
            total_reasoning += entry["reasoning_tokens"]
            total_output += entry["completion_tokens"]
            total_cost += entry["cost_usd"]
            total_saved += entry["cache_savings_usd"]

        total_cached_pct = (
            total_cached / total_input * 100.0 if total_input else 0.0
        )
        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            "",
            f"[bold]{total_calls}[/bold]",
            f"[bold]{total_input:,}[/bold]",
            (
                f"[bold]{total_cached:,} ({total_cached_pct:.0f}%)[/bold]"
                if total_cached
                else "[bold]0[/bold]"
            ),
            f"[bold]{total_reasoning:,}[/bold]",
            f"[bold]{total_output:,}[/bold]",
            f"[bold green]${total_cost:.4f}[/bold green]",
            f"[bold green]${total_saved:.4f}[/bold green]",
            f"[bold]{self.avg_tokens_per_sec:.1f}[/bold]",
        )
        return table

    def overview_text(self) -> Text:
        """One-line KV-cache / reasoning / throughput overview for ``/cost``."""
        text = Text()
        text.append("Cache hit: ", style="bold")
        text.append(
            f"{self.cache_hit_ratio:.1f}% "
            f"({self.total_cached_tokens:,} tok)",
            style="cyan",
        )
        text.append("  |  Saved: ", style="bold")
        text.append(f"${self.cache_savings_usd:.4f}", style="green")
        text.append("  |  Reasoning: ", style="bold")
        text.append(f"{self.total_reasoning_tokens:,} tok", style="magenta")
        text.append("  |  Avg throughput: ", style="bold")
        text.append(f"{self.avg_tokens_per_sec:.1f} tok/s", style="cyan")
        text.append("  |  Latency: ", style="bold")
        text.append(f"{self.avg_latency_ms:.0f} ms", style="cyan")
        if self.avg_ttft_ms > 0:
            text.append(f" (TTFT {self.avg_ttft_ms:.0f} ms)", style="dim")

        if self.per_tier_calls:
            breakdown = ", ".join(
                f"{tier} {count}"
                for tier, count in sorted(self.per_tier_calls.items())
            )
            text.append("\nTier calls: ", style="bold")
            text.append(breakdown, style="dim")
        if self.per_provider_calls:
            breakdown = ", ".join(
                f"{provider} {count}"
                for provider, count in sorted(self.per_provider_calls.items())
            )
            text.append("  |  Providers: ", style="bold")
            text.append(breakdown, style="dim")
        if self.tokens_saved_by_ast:
            text.append("  |  AST saved: ", style="bold")
            text.append(f"{self.tokens_saved_by_ast:,} tok", style="green")
        return text

    def budget_line(self) -> Text:
        """One-liner budget status for the footer."""
        pct = self.budget_percent()
        used = self.total_cost()
        style = (
            "bold red"
            if self.over_budget()
            else ("yellow" if self.near_budget() else "dim green")
        )
        text = Text()
        text.append(f"Budget: ${used:.4f} / ${self.session_budget:.2f} ", style=style)
        text.append(f"({pct:.1f}% used)", style=style)
        return text

    # ------------------------------------------------------------------
    # JSONL logging
    # ------------------------------------------------------------------

    def _append_log(self, record: CallRecord) -> None:
        try:
            assert self._log_path is not None
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record)) + "\n")
        except OSError:
            pass  # Never crash because of logging
