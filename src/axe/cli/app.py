"""Axe command-line entrypoint.

Parses CLI arguments, loads layered settings, initializes logging, and
dispatches to the interactive REPL or a one-shot agent run.
"""

from __future__ import annotations

import click

from axe import __version__
from axe.cli import configure_utf8_streams, repl
from axe.cli.git_hooks import install_pre_commit_hook, run_pre_commit_check
from axe.cli.security_prompt import confirm_airlock_bypass
from axe.config.settings import Settings, load_settings
from axe.llm.provider import ProviderError
from axe.utils.logger import get_logger, setup_logging

TIER_CHOICES = ["auto", "fast", "coder", "architect", "security"]

console = repl.console

configure_utf8_streams()


def _version_callback(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"Axe v{__version__}")
    ctx.exit()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("prompt", required=False, default=None)
@click.option(
    "--tier",
    type=click.Choice(TIER_CHOICES, case_sensitive=False),
    default="auto",
    show_default=True,
    help="Override the Jev auto-router with a fixed routing tier.",
)
@click.option(
    "--no-airlock",
    is_flag=True,
    default=False,
    help="Bypass outbound Code-Airlock redaction (requires explicit confirmation).",
)
@click.option(
    "--install-hook",
    is_flag=True,
    default=False,
    help="Install the Code-Airlock git pre-commit hook.",
)
@click.option(
    "--run-pre-commit",
    is_flag=True,
    default=False,
    help="Scan staged git diffs with Code-Airlock (used by the pre-commit hook).",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Confirm --no-airlock non-interactively. Does not confirm tool actions.",
)
@click.option(
    "--model",
    "-m",
    "model_override",
    default=None,
    help="Override the starting model for this run (e.g. openrouter/qwen/qwen3-coder-next).",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=None,
    help="Path to an extra TOML config file (highest file priority).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview actions without modifying files or running shell commands.",
)
@click.option(
    "--version",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=_version_callback,
    help="Show version and exit.",
)
def main(
    prompt: str | None,
    tier: str,
    no_airlock: bool,
    install_hook: bool,
    run_pre_commit: bool,
    yes: bool,
    model_override: str | None,
    config_path: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Axe - autonomous coding agent with intelligent multi-model routing.

    Run with a PROMPT for single-shot mode, or with no arguments for the
    interactive REPL.

    \b
    Examples:
      axe "list all python files"
      axe --tier architect "refactor the settings loader"
      axe --no-airlock "debug this failing test"
      axe                          # enter the REPL
    """
    if install_hook:
        try:
            install_pre_commit_hook(console=console)
        except (OSError, FileNotFoundError) as exc:
            console.print(f"[error]Failed to install hook: {exc}[/error]")
            raise SystemExit(1)
        raise SystemExit(0)

    try:
        settings: Settings = load_settings(config_path)
    except Exception as exc:
        console.print(f"[error]Failed to load config: {exc}[/error]")
        raise SystemExit(2)

    if run_pre_commit:
        raise SystemExit(run_pre_commit_check(settings=settings, console=console))

    if no_airlock:
        if not confirm_airlock_bypass(console, assume_yes=yes):
            console.print("[warning]Airlock was not bypassed. Exiting.[/warning]")
            raise SystemExit(1)
        settings.security.airlock.enabled = False
        settings.security.airlock.redact_outbound_context = False

    setup_logging(verbose=verbose, log_dir=".axe/logs")
    log = get_logger("main")
    log.debug(
        "Config loaded",
        provider=settings.providers.default,
        tier=tier,
        airlock=settings.security.airlock.enabled,
        jev=settings.router.jev.enabled,
    )

    try:
        if prompt:
            repl.run_one_shot(
                prompt,
                settings,
                tier=tier,
                model_override=model_override,
                dry_run=dry_run,
            )
        else:
            repl.start_repl(
                settings,
                tier=tier,
                model_override=model_override,
                dry_run=dry_run,
            )
    except (ValueError, ProviderError) as exc:
        # Unknown tier/provider or an exhausted fallback queue.
        console.print(f"[error]{exc}[/error]")
        raise SystemExit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        raise SystemExit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
