"""
main.py — Axe entry point.

Two modes:
  - Single-shot: `axe "do something"` — runs once and exits
  - REPL:        `axe` with no args — enters interactive loop
"""

from __future__ import annotations

import sys

try:
    import readline
except ImportError:
    pass

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from config.settings import Settings, load_settings
from llm.openai_provider import OpenAIProvider
from llm.router import ModelRouter
from cost.tracker import CostTracker
from agent.context import ContextManager
from agent.snapshots import SnapshotManager
from project.memory import ProjectMemory
from utils.logger import get_logger, setup_logging
from tools.registry import ToolRegistry
from tools.shell import ShellTool
from tools.file_read import FileReadTool
from tools.file_write import FileWriteTool
from tools.file_edit import FileEditTool
from tools.glob_search import GlobSearchTool
from tools.grep_search import GrepSearchTool
from agent.core import Agent

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Rich console with custom theme
# ---------------------------------------------------------------------------

THEME = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "model.tier1": "bold blue",
    "model.tier2": "bold magenta",
    "model.tier3": "bold yellow",
    "model.openai": "bold blue",
    "cost": "dim green",
    "prompt": "bold cyan",
})

console = Console(theme=THEME)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def show_banner(settings: Settings) -> None:
    """Display the welcome banner with project info."""
    from rich.align import Align

    banner = Text()
    
    # Beautiful ASCII Art for Axe
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
    
    banner.append(ascii_axe, style="bold cyan")
    banner.append(f"Autonomous coding agent with intelligent routing (v{__version__})\n", style="dim italic")
    banner.append("─" * 58 + "\n\n", style="dim")
    
    banner.append("Models: ", style="bold")
    banner.append(f"[T1] {settings.llm.tier1_model}", style="model.tier1")
    banner.append(" → ", style="dim")
    banner.append(f"[T2] {settings.llm.tier2_model}", style="model.tier2")
    banner.append(" → ", style="dim")
    banner.append(f"[T3] {settings.llm.tier3_model}", style="model.tier3")
    banner.append(f"\nBudget: ", style="bold")
    banner.append(f"${settings.cost.session_budget:.2f}/session", style="cost")
    banner.append(f"  Max iterations: {settings.agent.max_iterations}", style="dim")

    console.print(Panel(
        Align.center(banner),
        border_style="bold magenta",
        title="[bold]🪓 Axe[/bold]",
        subtitle="[dim]Type /help for commands, /exit to quit[/dim]",
        padding=(1, 2),
    ))


def build_router_and_tracker(
    settings: Settings,
    model_override: str | None = None,
) -> tuple[ModelRouter, CostTracker]:
    """Create a ModelRouter + CostTracker pair from resolved settings."""
    openai_config = settings.llm.openai
    provider = OpenAIProvider(
        model=model_override or settings.llm.tier1_model,
        api_key=openai_config.api_key,
        base_url=openai_config.base_url,
        temperature=openai_config.temperature,
    )
    router = ModelRouter(provider, settings, console)
    tracker = CostTracker(
        session_budget=settings.cost.session_budget,
        alert_at_percent=settings.cost.alert_at_percent,
        log_file=settings.cost.log_file,
    )
    return router, tracker


# ---------------------------------------------------------------------------
# REPL Commands
# ---------------------------------------------------------------------------

REPL_COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/exit": "Exit the agent",
    "/quit": "Exit the agent",
    "/cost": "Show current session cost",
    "/version": "Show version info",
    "/memory": "View/edit project memory (e.g., /memory set language=python)",
    "/rewind": "Rewind to a previous step (e.g., /rewind 3)",
    "/clear": "Clear conversation history (keeps memory)",
    "/explain": "Preview plan without executing",
}


def handle_repl_command(command: str, settings: Settings, **kwargs: object) -> bool:
    """
    Handle a REPL slash-command.

    Returns True if the REPL should continue, False if it should exit.
    """
    cmd = command.strip().lower().split()[0]
    args = command.strip().split()[1:]

    if cmd in ("/exit", "/quit"):
        _print_session_summary(kwargs.get("tracker"))  # type: ignore[arg-type]
        return False

    elif cmd == "/help":
        console.print("\n[bold]Available commands:[/bold]")
        for name, desc in REPL_COMMANDS.items():
            console.print(f"  [prompt]{name:<12}[/prompt] {desc}")
        console.print()

    elif cmd == "/version":
        console.print(f"\n[info]Axe v{__version__}[/info]")
        console.print(f"  Default model: {settings.llm.tier1_model}")
        console.print(f"  Python: {sys.version.split()[0]}\n")

    elif cmd == "/cost":
        tracker: CostTracker | None = kwargs.get("tracker")
        if tracker is None or tracker.total_calls() == 0:
            console.print("\n[dim]No LLM calls recorded yet.[/dim]\n")
        else:
            console.print()
            console.print(tracker.summary_table())
            console.print(tracker.budget_line())
            console.print()

    elif cmd == "/memory":
        memory: ProjectMemory | None = kwargs.get("memory")
        if memory is None:
            console.print("\n[warning]Project memory not available.[/warning]\n")
        elif args and args[0] == "set" and len(args) >= 2:
            # /memory set key=value
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
        snapshots: SnapshotManager | None = kwargs.get("snapshots")
        if snapshots is None:
            console.print("\n[warning]Snapshots not available.[/warning]\n")
        elif not args:
            # Show available steps
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

    elif cmd == "/clear":
        agent: Agent | None = kwargs.get("agent")
        if agent is not None:
            agent.clear_conversation()
        console.print("\n[info]Conversation cleared. Project memory preserved.[/info]\n")

    elif cmd == "/explain":
        # TODO: Integrate with ExplainMode
        console.print("\n[warning]Explain mode — not yet implemented.[/warning]\n")

    else:
        console.print(f"[error]Unknown command: {cmd}[/error]")
        console.print("[dim]Type /help for available commands.[/dim]")

    return True


# ---------------------------------------------------------------------------
# REPL Loop
# ---------------------------------------------------------------------------

def _print_session_summary(tracker: CostTracker | None) -> None:
    """Print the end-of-session cost table (if any calls were made)."""
    if tracker is None or tracker.total_calls() == 0:
        console.print("\n[dim]Session ended. No LLM calls were made.[/dim]")
        return
    console.print()
    console.print(tracker.summary_table())
    console.print(tracker.budget_line())
    console.print()


def run_repl(settings: Settings, router: ModelRouter, tracker: CostTracker) -> None:
    """Run the interactive REPL session."""
    log = get_logger("repl")
    show_banner(settings)

    # --- Project memory ---
    memory = ProjectMemory(console=console)
    memory.load()
    if not memory.data:
        memory.auto_detect()
    project_context = memory.to_system_context()

    # --- Context manager ---
    ctx_manager = ContextManager(settings.context, console=console)
    ctx_manager.set_provider(router)

    # --- Snapshot manager ---
    snapshots = SnapshotManager(console=console)

    registry = ToolRegistry()
    registry.register(ShellTool(settings.sandbox))
    registry.register(FileReadTool(settings.context))
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GlobSearchTool())
    registry.register(GrepSearchTool())
    agent = Agent(
        router, registry, settings, console,
        tracker=tracker,
        context_manager=ctx_manager,
        snapshot_manager=snapshots,
        project_context=project_context,
    )

    while True:
        try:
            user_input = input("\x01\033[1;36m\x02❯ \x01\033[0m\x02").strip()
        except (KeyboardInterrupt, EOFError):
            memory.save()
            _print_session_summary(tracker)
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            should_continue = handle_repl_command(
                user_input, settings,
                tracker=tracker, memory=memory,
                snapshots=snapshots, agent=agent,
            )
            if not should_continue:
                memory.save()
                break
            continue

        log.info("User prompt received", prompt=user_input)
        try:
            agent.run(user_input)
        except Exception as e:
            log.error("Agent failed", error=str(e))
            console.print(f"[error]Agent encountered an error: {e}[/error]\n")

        console.print()


# ---------------------------------------------------------------------------
# Single-shot mode
# ---------------------------------------------------------------------------

def run_single_shot(
    prompt: str,
    router: ModelRouter,
    tracker: CostTracker,
    settings: Settings,
) -> None:
    """Run the agent once on a single prompt, then exit."""
    log = get_logger("single_shot")
    log.info("Single-shot mode", prompt=prompt)

    # --- Project memory ---
    memory = ProjectMemory(console=console)
    memory.load()
    if not memory.data:
        memory.auto_detect()
    project_context = memory.to_system_context()

    # --- Context manager ---
    ctx_manager = ContextManager(settings.context, console=console)
    ctx_manager.set_provider(router)

    # --- Snapshot manager ---
    snapshots = SnapshotManager(console=console)

    console.print(f"\n[dim]Running:[/dim] {prompt}")
    registry = ToolRegistry()
    registry.register(ShellTool(settings.sandbox))
    registry.register(FileReadTool(settings.context))
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GlobSearchTool())
    registry.register(GrepSearchTool())
    agent = Agent(
        router, registry, settings, console,
        tracker=tracker,
        context_manager=ctx_manager,
        snapshot_manager=snapshots,
        project_context=project_context,
    )

    try:
        agent.run(prompt)
    except Exception as e:
        log.error("Agent execution failed", error=str(e))
        console.print(f"[error]Agent encountered an error: {e}[/error]\n")
        raise SystemExit(1)

    memory.save()
    _print_session_summary(tracker)
    console.print()


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("prompt", required=False, default=None)
@click.option(
    "--model", "-m",
    "model_override",
    default=None,
    help="Override OpenAI model for this run (e.g. gpt-4.1-mini).",
)
@click.option(
    "--config", "-c",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to custom config.toml file.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
@click.option(
    "--version",
    is_flag=True,
    default=False,
    help="Show version and exit.",
)
def cli(
    prompt: str | None,
    model_override: str | None,
    config_path: str | None,
    verbose: bool,
    version: bool,
) -> None:
    """
    🪓 Axe — Autonomous coding agent with intelligent multi-model routing.

    Run with a PROMPT for single-shot mode, or with no arguments for interactive REPL.

    \b
    Examples:
      axe "list all python files"
      axe "fix the failing tests"
      axe                          # Enter REPL mode
    """
    if version:
        console.print(f"Axe v{__version__}")
        raise SystemExit(0)

    # Load configuration
    try:
        settings = load_settings(config_path)
    except Exception as e:
        console.print(f"[error]Failed to load config: {e}[/error]")
        raise SystemExit(1)

    # Initialize logging
    setup_logging(verbose=verbose)
    log = get_logger("main")
    log.debug("Config loaded", provider=settings.llm.default_provider)
    try:
        router, tracker = build_router_and_tracker(settings, model_override=model_override)
    except ValueError as e:
        console.print(f"[error]{e}[/error]")
        raise SystemExit(1)

    # Route to single-shot or REPL mode
    if prompt:
        run_single_shot(prompt, router, tracker, settings)
    else:
        run_repl(settings, router, tracker)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
