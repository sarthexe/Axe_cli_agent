"""Smoke tests for the installed ``axe`` console entrypoint."""

from __future__ import annotations

import subprocess

import pytest
from click.testing import CliRunner

from axe.cli.app import TIER_CHOICES, main
from axe.config.settings import load_settings
from axe.llm.providers import ProviderRegistry
from axe.llm.router import ModelRouter


def test_help_lists_required_flags() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    for flag in ("--tier", "--no-airlock", "--install-hook", "--help"):
        assert flag in result.output


def test_version_flag_prints_package_version() -> None:
    result = CliRunner().invoke(main, ["--version"])

    assert result.exit_code == 0, result.output
    assert "0.2.0" in result.output


def test_install_hook_writes_pre_commit(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.chdir(repo)

    result = CliRunner().invoke(main, ["--install-hook"])

    assert result.exit_code == 0, result.output
    hook = repo / ".git" / "hooks" / "pre-commit"
    assert hook.exists()
    script = hook.read_text(encoding="utf-8")
    assert "axe --run-pre-commit" in script
    assert "Installed by Axe" in script


def test_missing_api_key_is_reported_at_call_time() -> None:
    settings = load_settings(env={}, include_user=False, include_workspace=False)
    registry = ProviderRegistry(settings)

    # No DEEPSEEK_API_KEY -> the slug falls back to OpenRouter.
    resolved = registry.resolve("deepseek/deepseek-v4")
    available, reason = resolved.availability()

    assert resolved.provider_name == "openrouter"
    assert available is False
    assert "OPENROUTER_API_KEY" in reason


@pytest.mark.parametrize("tier", TIER_CHOICES)
def test_every_cli_tier_is_routable(tier: str) -> None:
    settings = load_settings(env={}, include_user=False, include_workspace=False)
    expected = "coder" if tier == "auto" else tier
    registry = ProviderRegistry(settings)

    router = ModelRouter(registry, settings, initial_tier=expected)

    assert router.current_tier == expected
    assert router.current_model == settings.router.tiers[expected].models[0]
