"""Interactive prompts for security-sensitive AirLock decisions.

Guided Security Issue panels, strict/warn modes, and the explicit
``--no-airlock`` confirmation all live here.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from axe.config.settings import Settings


@dataclass(slots=True)
class GuidedSecurityDecision:
    """User's choice in the Guided Security Issue flow."""

    choice: str
    reason: str = ""


def prompt_guided_security_issue(
    console: Console,
    findings: list[object],
    *,
    path: str | None = None,
    mode: str = "interactive",
    assume_yes: bool = False,
    remediation_hint: str = "",
    input_fn: Callable[[str], str] | None = None,
) -> GuidedSecurityDecision:
    """Render the Guided Security Issue panel and collect a choice.

    Security findings are display objects (``SecurityFinding``) and are only
    read through their public attributes; raw secret values are never printed.
    """
    if mode == "strict_block":
        return GuidedSecurityDecision("block", "strict_block mode")

    if mode == "warn":
        console.print(
            "[warning]Airlock warning: security findings were detected; "
            "warn mode allows the action.[/warning]"
        )
        return GuidedSecurityDecision("ignore", "warn mode")

    lines = ["[bold red]Airlock detected one or more security issues.[/bold red]", ""]
    if path:
        lines.append(f"Target: [cyan]{escape(path)}[/cyan]")

    for finding in findings[:10]:
        category = str(getattr(finding, "category", "security"))
        severity = str(getattr(finding, "severity", "high"))
        line = getattr(finding, "line", None)
        message = str(getattr(finding, "message", "") or "")
        masked = getattr(finding, "masked_snippet", None)
        item = f"• [bold]{escape(category)}[/bold] ({escape(severity)})"
        if line:
            item += f" at line {line}"
        if message:
            item += f" — {escape(message)}"
        if masked:
            item += f"\n  [dim]{escape(str(masked).strip()[:240])}[/dim]"
        lines.append(item)

    if remediation_hint:
        lines.append("")
        lines.append(f"[bold]Suggested remediation:[/bold] {escape(remediation_hint)}")

    lines.extend(
        [
            "",
            "[bold]\\[a][/bold] Auto-remediate via Airlock",
            "[bold]\\[e][/bold] Extract secret to .env and use an env lookup",
            "[bold]\\[i][/bold] Ignore once (audit this decision)",
            "[bold]\\[b][/bold] Block and ask the agent to rewrite",
        ]
    )
    console.print(
        Panel(
            "\n".join(lines),
            title="Guided Security Issue",
            border_style="bold red",
            padding=(1, 2),
        )
    )

    if assume_yes:
        return GuidedSecurityDecision("auto", "non-interactive --yes")

    if not sys.stdin.isatty():
        console.print(
            "[error]Guided security remediation requires an interactive "
            "terminal; blocking this action.[/error]"
        )
        return GuidedSecurityDecision("block", "non-interactive without --yes")

    prompt = input_fn or (lambda text: console.input(text))
    choices = {
        "a": "auto",
        "auto": "auto",
        "e": "extract",
        "extract": "extract",
        "i": "ignore",
        "ignore": "ignore",
        "b": "block",
        "block": "block",
    }
    for _ in range(5):
        try:
            answer = prompt(
                "[bold cyan]Choose \\[a/e/i/b]:[/bold cyan] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt, OSError):
            return GuidedSecurityDecision("block", "prompt interrupted")
        choice = choices.get(answer)
        if choice is not None:
            return GuidedSecurityDecision(choice, "user selected")
        console.print("[warning]Please choose a, e, i, or b.[/warning]")
    return GuidedSecurityDecision("block", "no valid choice")


def confirm_airlock_bypass(
    console: Console,
    *,
    assume_yes: bool = False,
) -> bool:
    """Ask the user to confirm running without outbound redaction.

    Args:
        console: Rich console used for output and input.
        assume_yes: Non-interactive confirmation flag (``--yes``).

    Returns:
        ``True`` only when the user explicitly typed ``yes`` (or passed
        ``--yes``). Any other answer, EOF, or interrupt returns ``False``.
    """
    console.print(
        Panel(
            "[bold red]Outbound AirLock redaction will be disabled for this session.[/bold red]\n\n"
            "Secrets, API keys, and personally identifiable information in files,\n"
            "command output, or prompts may be sent to the model provider in clear text.\n\n"
            "Only continue if you understand and accept that risk.",
            title="⚠  Airlock bypass",
            border_style="bold red",
            padding=(1, 2),
        )
    )

    if assume_yes:
        console.print(
            "[bold red]Proceeding without outbound redaction (--yes).[/bold red]"
        )
        return True

    if not sys.stdin.isatty():
        console.print(
            "[error]--no-airlock requires an interactive terminal or --yes.[/error]"
        )
        return False

    try:
        answer = console.input(
            "[bold red]Type 'yes' to continue without redaction:[/bold red] "
        )
    except (EOFError, KeyboardInterrupt, OSError):
        return False

    return answer.strip().lower() == "yes"


def airlock_status_markup(settings: Settings) -> str:
    """Return a compact Rich-markup status line for the AirLock settings."""
    airlock = settings.security.airlock
    enabled = "[success]enabled[/success]" if airlock.enabled else "[error]disabled[/error]"
    redaction = (
        "[success]on[/success]"
        if airlock.redact_outbound_context
        else "[warning]off[/warning]"
    )
    mcp = (
        f"[info]{airlock.mcp_command} {' '.join(airlock.mcp_args)}[/info]"
        if airlock.default_mcp_enabled
        else "[dim]disabled[/dim]"
    )
    return (
        f"Airlock: {enabled}  redaction: {redaction}  "
        f"mcp: {mcp}  remediation: [dim]{airlock.guided_remediation_mode}[/dim]"
    )
