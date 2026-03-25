"""Type-safe configuration loading using Pydantic Settings."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-models for each TOML section
# ---------------------------------------------------------------------------

class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    model: str = "gpt-4.1-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""  # Loaded from env vars, never hardcoded
    temperature: float = 0.0
    context_window: int = 128_000


class LLMSettings(BaseModel):
    """Top-level LLM configuration with provider routing."""
    default_provider: str = "openai"
    openai: LLMProviderConfig = Field(default_factory=lambda: LLMProviderConfig(
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        context_window=128_000,
    ))
    tier1_model: str = "gpt-4.1-mini"
    tier2_model: str = "gpt-4.1"
    tier3_model: str = "o3-mini"


class RouterSettings(BaseModel):
    """Model router configuration — failure-based escalation."""
    escalate_after_failures: int = 2
    reset_on_success: bool = True
    show_model_in_output: bool = True


class CostSettings(BaseModel):
    """Cost tracking and budget limits."""
    session_budget: float = 1.00
    alert_at_percent: int = 80
    log_file: str = "~/.cli-agent/costs.jsonl"


class AgentSettings(BaseModel):
    """Core agent behavior."""
    max_iterations: int = 10
    max_retries_per_tool: int = 3
    confirmation_required: list[str] = Field(
        default_factory=lambda: ["rm", "drop", "kill", "truncate"]
    )


class SandboxSettings(BaseModel):
    """Subprocess sandbox constraints."""
    command_timeout_seconds: int = 30
    max_output_bytes: int = 51_200  # 50KB
    blocked_commands: list[str] = Field(
        default_factory=lambda: ["rm -rf /", "mkfs", "dd if=/dev/zero"]
    )


class ContextSettings(BaseModel):
    """Context window management."""
    max_file_lines: int = 500
    summarize_threshold: float = 0.6
    keep_recent_messages: int = 4
    effective_context_target: int = 32_768


class FeatureSettings(BaseModel):
    """Feature flags."""
    project_memory: bool = True
    error_patterns: bool = True
    session_snapshots: bool = True
    explain_mode: bool = False


# ---------------------------------------------------------------------------
# Root Settings model
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Root configuration — assembles all sub-models."""
    llm: LLMSettings = Field(default_factory=LLMSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    cost: CostSettings = Field(default_factory=CostSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    model_config = SettingsConfigDict(
        env_prefix="",
        env_nested_delimiter="__",
        extra="ignore",
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_USER_CONFIG = Path.home() / ".cli-agent" / "config.toml"


def load_settings(config_path: str | Path | None = None) -> Settings:
    """
    Load settings with priority: explicit path > user config > code defaults.

    Environment variables (including values from `.env`) are then applied.
    """
    # 1. Load .env values into environment for local development.
    load_dotenv(override=False)

    # 2. Load optional TOML overrides.
    data: dict[str, object] = {}
    if config_path:
        file_path = Path(config_path)
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
    else:
        file_path = _USER_CONFIG
        if file_path.exists():
            with open(file_path, "rb") as f:
                data = tomllib.load(f)

    # 3. Inject API keys from environment variables.

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        data.setdefault("llm", {}).setdefault("openai", {})["api_key"] = openai_key

    openai_base_url = os.environ.get("OPENAI_BASE_URL", "")
    if openai_base_url:
        data.setdefault("llm", {}).setdefault("openai", {})["base_url"] = openai_base_url

    return Settings(**data)
