"""Tests for the layered Axe configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from axe.config.settings import Settings, load_settings


def _isolated(**kwargs: object) -> Settings:
    """Load settings with user/workspace layers and environment isolated."""
    params: dict[str, object] = {
        "env": {},
        "include_user": False,
        "include_workspace": False,
    }
    params.update(kwargs)
    return load_settings(**params)  # type: ignore[arg-type]


def test_load_settings_loads_bundled_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """default_config.toml is loaded from the package, not the cwd."""
    monkeypatch.chdir(tmp_path)

    settings = _isolated()

    assert isinstance(settings, Settings)
    assert settings.router.jev.enabled is True
    assert "deepseek/deepseek-v4-pro" in settings.router.tiers["architect"].models
    assert settings.security.airlock.default_mcp_enabled is True


def test_bundled_defaults_are_cwd_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    first = _isolated()

    nested = tmp_path / "some" / "nested" / "dir"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    second = _isolated()

    assert first.router.jev.model == second.router.jev.model
    assert (
        first.router.tiers["architect"].models
        == second.router.tiers["architect"].models
    )
    assert (
        first.security.airlock.guided_remediation_mode
        == second.security.airlock.guided_remediation_mode
    )


def test_workspace_config_overrides_bundled_defaults(tmp_path: Path) -> None:
    axe_dir = tmp_path / ".axe"
    axe_dir.mkdir()
    (axe_dir / "config.toml").write_text(
        "[general]\nmax_iterations = 7\n\n[router.jev]\nenabled = false\n",
        encoding="utf-8",
    )

    settings = load_settings(env={}, cwd=tmp_path, include_user=False)

    assert settings.router.jev.enabled is False
    assert settings.general.max_iterations == 7
    # general.max_iterations is mirrored into the legacy agent setting.
    assert settings.agent.max_iterations == 7


def test_user_global_config_is_loaded(tmp_path: Path) -> None:
    xdg = tmp_path / "xdg"
    config_file = xdg / "axe" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(
        "[security.airlock]\ndefault_mcp_enabled = false\n",
        encoding="utf-8",
    )

    settings = load_settings(
        env={"XDG_CONFIG_HOME": str(xdg)},
        cwd=tmp_path,
        include_workspace=False,
    )

    assert settings.security.airlock.default_mcp_enabled is False


def test_environment_variables_override_files(tmp_path: Path) -> None:
    axe_dir = tmp_path / ".axe"
    axe_dir.mkdir()
    (axe_dir / "config.toml").write_text(
        "[general]\nmax_iterations = 3\n\n[router.jev]\nenabled = true\n",
        encoding="utf-8",
    )

    test_key = "sk-" + "or-test"
    settings = load_settings(
        env={
            "AXE_ROUTER__JEV__ENABLED": "false",
            "AXE_GENERAL__MAX_ITERATIONS": "12",
            "OPENROUTER_API_KEY": test_key,
        },
        cwd=tmp_path,
        include_user=False,
    )

    assert settings.router.jev.enabled is False
    assert settings.general.max_iterations == 12
    assert settings.agent.max_iterations == 12
    assert settings.providers.openrouter.api_key == test_key


def test_explicit_config_path_has_file_priority(tmp_path: Path) -> None:
    explicit = tmp_path / "override.toml"
    explicit.write_text("[router.jev]\nenabled = false\n", encoding="utf-8")

    settings = load_settings(explicit, env={}, cwd=tmp_path)

    assert settings.router.jev.enabled is False


def test_missing_explicit_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.toml", env={}, cwd=tmp_path)


def test_no_api_key_is_required_to_boot() -> None:
    settings = _isolated()

    # An empty key must not fail validation or import time.
    assert settings.providers.openrouter.api_key == ""
    assert settings.providers.ollama.base_url == "http://localhost:11434"
    assert settings.router.tiers.default_fallback == "coder"
