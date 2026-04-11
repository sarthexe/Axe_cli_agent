"""Session cost tracker.

Records every LLM call's token usage, computes USD cost from the pricing
table, enforces a session budget, and generates a Rich-formatted summary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from cost.pricing import PRICING, TIER_LABELS, cost_for


@dataclass
class CallRecord:
    """A single LLM call's billing snapshot."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CostTracker:
    """Tracks token usage and USD cost across a session.

    Usage::

        tracker = CostTracker(session_budget=1.00)
        tracker.record(model="gpt-4.1-mini", usage=response.raw.get("usage", {}))
        print(tracker.total_cost())
    """

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

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, model: str, usage: dict[str, Any] | None) -> CallRecord:
        """Record a completed LLM call and return the CallRecord.

        ``usage`` is the raw dict from ``response.raw.get("usage")``.
        """
        if not usage:
            usage = {}

        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        cost = cost_for(model, prompt_tokens, completion_tokens)

        rec = CallRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )
        self._records.append(rec)

        if self._log_path:
            self._append_log(rec)

        return rec

    # ------------------------------------------------------------------
    # Budget checks
    # ------------------------------------------------------------------

    def total_cost(self) -> float:
        """Total USD spent this session."""
        return sum(r.cost_usd for r in self._records)

    def budget_percent(self) -> float:
        """Fraction of budget consumed (0–100)."""
        if self.session_budget <= 0:
            return 100.0
        return (self.total_cost() / self.session_budget) * 100.0

    def over_budget(self) -> bool:
        """True if session cost has exceeded the configured budget."""
        return self.total_cost() >= self.session_budget

    def near_budget(self) -> bool:
        """True if spending has reached the alert threshold."""
        return self.budget_percent() >= self.alert_at_percent

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def per_model_stats(self) -> dict[str, dict[str, Any]]:
        """Aggregate stats by model: calls, tokens (in/out), cost."""
        stats: dict[str, dict[str, Any]] = {}
        for rec in self._records:
            entry = stats.setdefault(
                rec.model,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
            )
            entry["calls"] += 1
            entry["prompt_tokens"] += rec.prompt_tokens
            entry["completion_tokens"] += rec.completion_tokens
            entry["cost_usd"] += rec.cost_usd
        return stats

    def per_model_counts(self) -> dict[str, int]:
        """Simple {model: call_count} mapping."""
        return {m: s["calls"] for m, s in self.per_model_stats().items()}

    def total_calls(self) -> int:
        return len(self._records)

    def summary_table(self, console: Console | None = None) -> Table:
        """Build a Rich Table with per-model breakdown + totals."""
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
        table.add_column("Tokens In", justify="right")
        table.add_column("Tokens Out", justify="right")
        table.add_column("Cost", justify="right", style="green")

        stats = self.per_model_stats()
        total_calls = 0
        total_in = 0
        total_out = 0
        total_cost = 0.0

        # Emit rows in tier order
        for model in ["gpt-4.1-mini", "gpt-4.1", "o3-mini"]:
            if model not in stats:
                continue
            s = stats[model]
            tier = TIER_LABELS.get(model, "?")
            table.add_row(
                model,
                tier,
                str(s["calls"]),
                f"{s['prompt_tokens']:,}",
                f"{s['completion_tokens']:,}",
                f"${s['cost_usd']:.4f}",
            )
            total_calls += s["calls"]
            total_in += s["prompt_tokens"]
            total_out += s["completion_tokens"]
            total_cost += s["cost_usd"]

        # Totals row
        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            "",
            f"[bold]{total_calls}[/bold]",
            f"[bold]{total_in:,}[/bold]",
            f"[bold]{total_out:,}[/bold]",
            f"[bold green]${total_cost:.4f}[/bold green]",
        )

        return table

    def budget_line(self) -> Text:
        """One-liner budget status for the footer."""
        pct = self.budget_percent()
        used = self.total_cost()
        style = "bold red" if self.over_budget() else ("yellow" if self.near_budget() else "dim green")
        t = Text()
        t.append(f"Budget: ${used:.4f} / ${self.session_budget:.2f} ", style=style)
        t.append(f"({pct:.1f}% used)", style=style)
        return t

    # ------------------------------------------------------------------
    # JSONL logging
    # ------------------------------------------------------------------

    def _append_log(self, rec: CallRecord) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            with self._log_path.open("a", encoding="utf-8") as fh:  # type: ignore[union-attr]
                fh.write(json.dumps(rec.__dict__) + "\n")
        except OSError:
            pass  # Never crash because of logging
