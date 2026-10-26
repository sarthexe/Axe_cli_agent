"""Interactive REPL, slash commands, and single-shot execution.

This module owns everything that happens *after* the CLI has parsed
arguments and loaded settings: provider construction, the terminal loop,
session cost reporting, and the one-shot path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

try:  # Optional: nicer line editing on POSIX
    import readline  # noqa: F401
except ImportError:  # pragma: no cover - Windows / minimal environments
    pass

from rich.align import Align
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from axe import __version__
from axe.agent.context import ContextManager
from axe.agent.core import Agent
from axe.agent.snapshots import SnapshotManager
from axe.cli.security_prompt import airlock_status_markup, confirm_airlock_bypass
from axe.config.settings import Settings
from axe.cost.tracker import CostTracker
from axe.llm.providers import ProviderRegistry
from axe.llm.router import ModelRouter
from axe.mcp.client import MCPClientManager
from axe.mcp.default_servers import register_builtin_airlock_tool, setup_default_servers
from axe.project.memory import ProjectMemory
from axe.sandbox.permissions import CommandSafety
from axe.security.airlock_middleware import AirlockMiddleware
from axe.security.guided_fixer import GuidedSecurityFixer
from axe.tools.file_edit import FileEditTool
from axe.tools.file_read import FileReadTool
from axe.tools.file_write import FileWriteTool
from axe.tools.glob_search import GlobSearchTool
from axe.tools.grep_search import GrepSearchTool
from axe.tools.registry import ToolRegistry
from axe.tools.shell import ShellTool
from axe.utils.logger import SessionLogger, get_logger

# ---------------------------------------------------------------------------
# Rich console with custom theme
# ---------------------------------------------------------------------------

THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "tier.fast": "bold blue",
        "tier.coder": "bold cyan",
        "tier.architect": "bold magenta",
        "tier.security": "bold yellow",
        "cost": "dim green",
        "prompt": "bold cyan",
    }
)

console = Console(theme=THEME)

_PROMPT = "\x01\033[1;36m\x02❯ \x01\033[0m\x02"

# Providers that do not require a real API key for local use.
_BUILTIN_TIERS = {"fast", "coder", "architect", "security"}


def _tier_style(name: str) -> str:
    """Return a themed style for built-in tiers, falling back for custom ones."""
    return f"tier.{name}" if name in _BUILTIN_TIERS else "bold cyan"

REPL_COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/exit": "Exit the agent",
    "/quit": "Exit the agent",
    "/clear": "Clear conversation history (keeps memory)",
    "/cost": "Show current session cost",
    "/tier": "Show routing mode/decision, or switch with /tier auto|<tier>",
    "/airlock": "Show Airlock status, toggle on|off, or scan <path_or_text>",
    "/version": "Show version info",
    "/memory": "View/edit project memory (e.g., /memory set language=python)",
    "/rewind": "Rewind to a previous step (e.g., /rewind 3)",
    "/snapshots": "List turn snapshots and files touched",
    "/undo": "Restore files to before the last file-modifying turn",
    "/explain": "Preview plan without executing",
}


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """Everything needed to run one REPL or one-shot session."""

    settings: Settings
    router: ModelRouter
    tracker: CostTracker
    memory: ProjectMemory | None
    snapshots: SnapshotManager | None
    safety: CommandSafety
    session_logger: SessionLogger
    agent: Agent
    airlock: AirlockMiddleware | None = None
    mcp_manager: MCPClientManager | None = None
    guided_fixer: GuidedSecurityFixer | None = None


def build_router_and_tracker(
    settings: Settings,
    *,
    tier: str = "auto",
    model_override: str | None = None,
    airlock: AirlockMiddleware | None = None,
) -> tuple[ModelRouter, CostTracker]:
    """Create a ModelRouter + CostTracker pair from resolved settings.

    Provider construction is deferred to the registry, so the CLI can boot
    without any API key and report actionable fallback warnings per model.
    """
    registry = ProviderRegistry(settings, console=console)
    tracker = CostTracker(
        session_budget=settings.cost.session_budget,
        alert_at_percent=settings.cost.alert_at_percent,
        log_file=settings.cost.log_file,
    )
    router = ModelRouter(
        registry,
        settings,
        console,
        initial_tier=tier,
        model_override=model_override,
        tracker=tracker,
        airlock=airlock,
    )
    return router, tracker


def create_session(
    settings: Settings,
    *,
    tier: str = "auto",
    model_override: str | None = None,
    dry_run: bool = False,
    mode: str = "repl",
    assume_yes: bool = False,
) -> Session:
    """Construct the agent runtime for a REPL or single-shot session."""
    airlock = AirlockMiddleware(settings, console=console)
    guided_fixer = GuidedSecurityFixer(
        settings,
        airlock,
        repo_root=Path.cwd(),
        console=console,
        assume_yes=assume_yes,
    )
    router, tracker = build_router_and_tracker(
        settings,
        tier=tier,
        model_override=model_override,
        airlock=airlock,
    )

    memory: ProjectMemory | None = None
    project_context = ""
    if settings.features.project_memory:
        memory_path = Path(settings.general.project_memory_file).expanduser()
        memory = ProjectMemory(memory_dir=memory_path.parent, console=console)
        memory.load()
        if not memory.data:
            memory.auto_detect()
        project_context = memory.to_system_context()

    ctx_manager = ContextManager(settings.context, console=console)
    ctx_manager.set_provider(router)

    snapshots = (
        SnapshotManager(snapshot_dir=Path(".axe/snapshots"), console=console)
        if settings.features.session_snapshots
        else None
    )
    safety = CommandSafety(
        settings.sandbox,
        settings.agent,
        guided_fixer=guided_fixer,
    )
    session_logger = SessionLogger(path=".axe/logs/session.jsonl")
    session_logger.log_event("session_start", mode=mode, dry_run=dry_run)

    registry = ToolRegistry()
    registry.register(ShellTool(settings.sandbox, safety=safety))
    registry.register(FileReadTool(settings.context))
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GlobSearchTool())
    registry.register(GrepSearchTool())

    mcp_manager = setup_default_servers(
        settings,
        registry,
        airlock,
        console,
    )

    agent = Agent(
        router,
        registry,
        settings,
        console,
        tracker=tracker,
        context_manager=ctx_manager,
        snapshot_manager=snapshots,
        command_safety=safety,
        dry_run=dry_run,
        session_logger=session_logger,
        project_context=project_context,
        airlock=airlock,
        project_memory=memory,
        guided_fixer=guided_fixer,
    )

    return Session(
        settings=settings,
        router=router,
        tracker=tracker,
        memory=memory,
        snapshots=snapshots,
        safety=safety,
        session_logger=session_logger,
        agent=agent,
        airlock=airlock,
        mcp_manager=mcp_manager,
        guided_fixer=guided_fixer,
    )


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def show_banner(session: Session, dry_run: bool = False) -> None:
    """Display the welcome banner with routing and security status."""
    settings = session.settings
    ascii_axe = (
        "   █████████                       \n"
        "  ███▒▒▒▒▒███                      \n"
        " ▒███    ▒███  █████ █████  ██████ \n"
        " ▒███████████ ▒▒███ ▒▒███  ███▒▒███\n"
        " ▒███▒▒▒▒▒███  ▒▒▒█████▒  ▒███████ \n"
        " ▒███    ▒███   ███▒▒▒███ ▒███▒▒▒  \n"
        " █████   █████ █████ █████▒▒██████ \n"
        "▒▒▒▒▒   ▒▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒▒  ▒▒▒▒▒▒  \n\n"
    )

    banner = Text()
    banner.append(ascii_axe, style="bold cyan")
    banner.append(
        f"Autonomous coding agent with intelligent routing (v{__version__})\n",
        style="dim italic",
    )
    banner.append("─" * 58 + "\n\n", style="dim")

    banner.append("Active: ", style="bold")
    banner.append(
        f"[{session.router.current_tier}] {session.router.current_model}",
        style=_tier_style(session.router.current_tier),
    )
    banner.append("\nTiers: ", style="bold")
    for index, name in enumerate(("fast", "coder", "architect", "security")):
        tier = settings.router.tiers.get(name)
        if tier is None or not tier.models:
            continue
        if index:
            banner.append(" → ", style="dim")
        banner.append(f"{name}:{tier.models[0]}", style=_tier_style(name))
    banner.append("\nJev router: ", style="bold")
    banner.append(
        "enabled" if settings.router.jev.enabled else "disabled",
        style="success" if settings.router.jev.enabled else "warning",
    )
    banner.append(f"  ({settings.router.jev.model})", style="dim")
    banner.append("\nBudget: ", style="bold")
    banner.append(f"${settings.cost.session_budget:.2f}/session", style="cost")
    banner.append(f"  Max iterations: {settings.agent.max_iterations}", style="dim")
    banner.append("\n")
    banner.append_text(Text.from_markup(f"  {airlock_status_markup(settings)}"))
    if dry_run:
        banner.append("\nMode: ", style="bold")
        banner.append("DRY-RUN (no writes, no shell execution)", style="warning")

    console.print(
        Panel(
            Align.center(banner),
            border_style="bold magenta",
            title="[bold]🪓 Axe[/bold]",
            subtitle="[dim]Type /help for commands, /exit to quit[/dim]",
            padding=(1, 2),
        )
    )


# ---------------------------------------------------------------------------
# REPL commands
# ---------------------------------------------------------------------------


def _tier_table(session: Session) -> Table:
    table = Table(
        title="Routing Tiers",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=False,
    )
    table.add_column("", justify="center", width=2)
    table.add_column("Tier", style="bold")
    table.add_column("Thinking", justify="center")
    table.add_column("Max Tokens", justify="right")
    table.add_column("Models")

    for name in session.settings.router.tiers.names():
        tier = session.settings.router.tiers.get(name)
        if tier is None:
            continue
        marker = "→" if name == session.router.current_tier else ""
        table.add_row(
            marker,
            name,
            tier.thinking_mode,
            f"{tier.max_tokens:,}",
            ", ".join(tier.models),
        )
    return table


def _print_tier_status(session: Session) -> None:
    """Print routing mode and the latest TierDecision summary."""
    router = session.router
    mode = (
        "auto (Jev)"
        if router.requested_tier == "auto"
        else f"locked ({router.requested_tier})"
    )
    console.print(f"[bold]Routing mode:[/bold] {mode}")

    decision = router.last_decision
    if decision is None:
        console.print("[dim]No routing decision yet.[/dim]\n")
        return

    console.print(
        f"[bold]Selected:[/bold] {decision.tier}  "
        f"thinking={decision.thinking_mode}  "
        f"max_tokens={decision.max_tokens:,}"
    )
    console.print(f"[dim]Reason:[/dim] {escape(decision.reason)}")

    jev = decision.jev_eval
    if jev is not None:
        console.print(
            f"  Jev: intent={jev.task_intent} ({jev.intent_confidence:.2f})  "
            f"depth={jev.reasoning_level} ({jev.reasoning_score:.2f})"
        )
        console.print(
            f"  risk={jev.destructive_risk_prob:.2f}  "
            f"ast_sufficient={jev.ast_sufficient_prob:.2f}  "
            f"needs_thinking={jev.needs_thinking_prob:.2f}  "
            f"latency={jev.latency_ms:.0f}ms"
        )
        if jev.used_fallback:
            console.print(
                f"  [warning]Fallback: {escape(jev.fallback_reason or 'heuristic')}[/warning]"
            )
    else:
        console.print("  [dim]No Jev evaluation yet (startup or forced tier).[/dim]")

    console.print(
        f"  risk_flagged={decision.risk_flagged}  "
        f"ast_compression={decision.use_ast_compression}"
    )
    context_manager = getattr(getattr(session, "agent", None), "context_manager", None)
    ast_saved = int(getattr(context_manager, "tokens_saved_by_ast", 0) or 0)
    if ast_saved:
        console.print(f"  ast_saved_tokens={ast_saved:,}")
    console.print()


def _print_airlock_status(session: Session) -> None:
    """Display Airlock/MCP/vault status without exposing any raw values."""
    settings = session.settings
    airlock_settings = settings.security.airlock
    middleware = session.airlock
    stats = middleware.stats() if middleware is not None else {}

    mode = stats.get("mode", "disabled" if not airlock_settings.enabled else "builtin_cascade")
    mode_label = {
        "mcp_connected": "MCP connected (code-airlock)",
        "builtin_cascade": "built-in cascade fallback",
        "disabled": "disabled",
    }.get(str(mode), str(mode))

    vault_by_category = stats.get("by_category", {})
    category_text = (
        ", ".join(f"{category}={count}" for category, count in sorted(vault_by_category.items()))
        if isinstance(vault_by_category, dict) and vault_by_category
        else "none"
    )

    lines = [
        f"Enabled: {'yes' if airlock_settings.enabled else 'no'}",
        f"Mode: {mode_label}",
        f"Outbound redaction: {'on' if airlock_settings.redact_outbound_context else 'off'}",
        f"Restore on write/shell: {'on' if airlock_settings.restore_on_file_write else 'off'}",
        f"Guided remediation: {airlock_settings.guided_remediation_mode}",
        f"MCP status: {stats.get('mcp_status', 'not_configured')}",
        f"Total secrets redacted: {stats.get('total_secrets_redacted', 0)}",
        f"Total PII redacted: {stats.get('total_pii_redacted', 0)}",
        f"Active vault placeholders: {stats.get('active_placeholders', 0)}",
        f"Restored placeholders: {stats.get('restored_placeholders', 0)}",
        f"Scan passes: {stats.get('scan_count', 0)}",
        f"Categories: {category_text}",
    ]
    if stats.get("last_detector_source"):
        lines.append(f"Last detector: {stats['last_detector_source']}")
    if stats.get("mcp_error"):
        lines.append(f"MCP notice: {stats['mcp_error']}")

    console.print()
    console.print(Panel("\n".join(lines), title="Airlock", border_style="cyan"))
    console.print()


def _print_airlock_scan(session: Session, target: str) -> None:
    """Scan a path or inline text and print a raw-value-free summary."""
    middleware = session.airlock
    if middleware is None:
        console.print("[warning]Airlock middleware is not available.[/warning]")
        return

    findings = middleware.scan_path(target)
    if not findings:
        console.print("\n[success]No secrets or PII detected.[/success]\n")
        return

    table = Table(
        title="Airlock Scan",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=False,
    )
    table.add_column("Category", style="bold")
    table.add_column("Detector")
    table.add_column("Placeholder")
    table.add_column("Source")
    table.add_column("Count", justify="right")

    for finding in findings[:100]:
        table.add_row(
            finding.category,
            finding.detector_source,
            finding.placeholder or "-",
            escape(finding.source_hint or "inline_text"),
            str(finding.redacted_count),
        )

    console.print()
    console.print(table)
    if len(findings) > 100:
        console.print(f"[dim]Showing first 100 of {len(findings)} findings.[/dim]")
    console.print(
        f"[dim]{len(findings)} finding(s). Raw secret values are withheld.[/dim]\n"
    )


def handle_repl_command(command: str, session: Session) -> bool:
    """Handle a REPL slash-command.

    Returns ``True`` if the REPL should continue, ``False`` if it should exit.
    """
    parts = command.strip().split()
    cmd = parts[0].lower()
    args = parts[1:]
    settings = session.settings

    if cmd in ("/exit", "/quit"):
        _print_session_summary(session.tracker)
        return False

    if cmd == "/help":
        console.print("\n[bold]Available commands:[/bold]")
        for name, desc in REPL_COMMANDS.items():
            console.print(f"  [prompt]{name:<12}[/prompt] {desc}")
        console.print()

    elif cmd == "/version":
        console.print(f"\n[info]Axe v{__version__}[/info]")
        console.print(f"  Active model: {session.router.current_model}")
        console.print(f"  Tier: {session.router.current_tier}")
        console.print(f"  Python: {sys.version.split()[0]}\n")

    elif cmd == "/cost":
        if session.tracker.total_calls() == 0:
            console.print("\n[dim]No LLM calls recorded yet.[/dim]\n")
        else:
            console.print()
            console.print(session.tracker.summary_table())
            console.print(session.tracker.overview_text())
            console.print(session.tracker.budget_line())
            console.print()

    elif cmd == "/tier":
        if not args:
            console.print()
            _print_tier_status(session)
            console.print(_tier_table(session))
            console.print()
        else:
            requested = args[0].lower()
            try:
                session.router.set_tier(requested, announce=False)
            except KeyError as exc:
                console.print(f"[error]{exc.args[0]}[/error]")
            else:
                if requested == "auto":
                    console.print(
                        "[success]Auto-routing enabled (Jev evaluates each turn).[/success]"
                    )
                else:
                    console.print(
                        f"[success]Tier locked to {requested} "
                        f"({session.router.current_model})[/success]"
                    )
                _print_tier_status(session)

    elif cmd == "/airlock":
        airlock = settings.security.airlock
        if not args:
            _print_airlock_status(session)
        elif args[0].lower() in {"on", "off"}:
            turn_on = args[0].lower() == "on"
            if not turn_on and airlock.enabled:
                if not confirm_airlock_bypass(console):
                    console.print("[warning]Airlock left enabled.[/warning]\n")
                    return True
            airlock.enabled = turn_on
            airlock.redact_outbound_context = turn_on
            agent = getattr(session, "agent", None)
            registry = getattr(agent, "registry", None)
            if (
                turn_on
                and session.airlock is not None
                and registry is not None
                and not registry.has("airlock_scan")
            ):
                register_builtin_airlock_tool(
                    registry,
                    session.airlock,
                )
            state = "enabled" if turn_on else "disabled"
            console.print(f"[success]Airlock {state} for this session.[/success]")
            _print_airlock_status(session)
        elif args[0].lower() == "scan":
            parts = command.strip().split(None, 2)
            target = parts[2].strip() if len(parts) >= 3 else ""
            if not target:
                console.print("[error]Usage: /airlock scan <path_or_text>[/error]")
            else:
                _print_airlock_scan(session, target)
        else:
            console.print("[error]Usage: /airlock [on|off|scan <path_or_text>][/error]")

    elif cmd == "/clear":
        session.agent.clear_conversation()
        console.print("\n[info]Conversation cleared. Project memory preserved.[/info]\n")

    elif cmd == "/memory":
        memory = session.memory
        if memory is None:
            console.print("\n[warning]Project memory is disabled.[/warning]\n")
        elif args and args[0] == "set" and len(args) >= 2:
            kv = " ".join(args[1:])
            if "=" in kv:
                key, value = kv.split("=", 1)
                memory.update(key.strip(), value.strip())
                console.print(f"[success]Set {key.strip()} = {value.strip()}[/success]\n")
            else:
                console.print("[error]Usage: /memory set key=value[/error]")
        elif args and args[0] == "detect":
            memory.auto_detect()
        else:
            console.print()
            console.print(memory.display_table())
            console.print()

    elif cmd == "/rewind":
        snapshots = session.snapshots
        if snapshots is None:
            console.print("\n[warning]Snapshots are disabled.[/warning]\n")
        elif not args:
            console.print()
            console.print(snapshots.list_table())
            console.print()
        else:
            try:
                step = int(args[0])
            except ValueError:
                console.print("[error]Usage: /rewind <step_number>[/error]")
                return True
            restored = snapshots.rewind(step)
            if not restored:
                console.print(f"[warning]No snapshot found for step {step}[/warning]")

    elif cmd == "/snapshots":
        snapshots = session.snapshots
        if snapshots is None:
            console.print("\n[warning]Snapshots are disabled.[/warning]\n")
        else:
            console.print()
            console.print(snapshots.list_table())
            console.print()

    elif cmd == "/undo":
        snapshots = session.snapshots
        if snapshots is None:
            console.print("\n[warning]Snapshots are disabled.[/warning]\n")
        else:
            report = snapshots.rollback_last()
            if report.success:
                console.print(f"[success]{report.message}[/success]\n")
            else:
                console.print(f"[warning]{report.message}[/warning]\n")

    elif cmd == "/explain":
        if not args:
            console.print("\n[warning]Usage: /explain <task>[/warning]\n")
        else:
            explain_text = session.agent.explain(" ".join(args))
            console.print()
            console.print(Panel(explain_text, title="Execution Preview", border_style="cyan"))
            console.print()

    else:
        console.print(f"[error]Unknown command: {cmd}[/error]")
        console.print("[dim]Type /help for available commands.[/dim]")

    return True


# ---------------------------------------------------------------------------
# REPL loop
# ---------------------------------------------------------------------------


def _print_session_summary(tracker: CostTracker | None) -> None:
    """Print the end-of-session cost table (if any calls were made)."""
    if tracker is None or tracker.total_calls() == 0:
        console.print("\n[dim]Session ended. No LLM calls were made.[/dim]")
        return
    console.print()
    console.print(tracker.summary_table())
    console.print(tracker.overview_text())
    console.print(tracker.budget_line())
    console.print()


def start_repl(
    settings: Settings,
    *,
    tier: str = "auto",
    model_override: str | None = None,
    dry_run: bool = False,
) -> None:
    """Run the interactive REPL session."""
    log = get_logger("repl")
    session = create_session(
        settings,
        tier=tier,
        model_override=model_override,
        dry_run=dry_run,
        mode="repl",
    )
    show_banner(session, dry_run=dry_run)
    log.info(
        "REPL started",
        tier=session.router.current_tier,
        model=session.router.current_model,
    )

    reason = "eof"
    try:
        while True:
            try:
                user_input = input(_PROMPT).strip()
            except (KeyboardInterrupt, EOFError):
                reason = "interrupt"
                _print_session_summary(session.tracker)
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if not handle_repl_command(user_input, session):
                    reason = "command_exit"
                    break
                continue

            safe_input = user_input
            if session.airlock is not None:
                safe_input = session.airlock.sanitize_outbound(
                    user_input,
                    source="user_prompt",
                ).sanitized_text
            log.info("User prompt received", prompt=safe_input)
            try:
                session.agent.run(safe_input)
            except Exception as exc:
                log.error("Agent failed", error=str(exc))
                console.print(f"[error]Agent encountered an error: {exc}[/error]\n")
            console.print()
    finally:
        if session.mcp_manager is not None:
            session.mcp_manager.close_all()
        if session.memory is not None:
            session.memory.save()
        session.session_logger.log_event("session_end", mode="repl", reason=reason)


# ---------------------------------------------------------------------------
# Single-shot mode
# ---------------------------------------------------------------------------


def run_one_shot(
    prompt: str,
    settings: Settings,
    *,
    tier: str = "auto",
    model_override: str | None = None,
    dry_run: bool = False,
) -> None:
    """Run the agent once on a single prompt, then exit."""
    log = get_logger("single_shot")
    session = create_session(
        settings,
        tier=tier,
        model_override=model_override,
        dry_run=dry_run,
        mode="single",
    )

    safe_prompt = prompt
    if session.airlock is not None:
        safe_prompt = session.airlock.sanitize_outbound(
            prompt,
            source="user_prompt",
        ).sanitized_text
    log.info("Single-shot mode", prompt=safe_prompt)

    mode_note = " [DRY-RUN]" if dry_run else ""
    console.print(f"\n[dim]Running{mode_note}:[/dim] {safe_prompt}")

    try:
        try:
            session.agent.run(safe_prompt)
        except Exception as exc:
            log.error("Agent execution failed", error=str(exc))
            console.print(f"[error]Agent encountered an error: {exc}[/error]\n")
            session.session_logger.log_event(
                "session_end", mode="single", reason="error", error=str(exc)
            )
            raise SystemExit(1)

        if session.memory is not None:
            session.memory.save()
        _print_session_summary(session.tracker)
        session.session_logger.log_event("session_end", mode="single", reason="done")
        console.print()
    finally:
        if session.mcp_manager is not None:
            session.mcp_manager.close_all()
