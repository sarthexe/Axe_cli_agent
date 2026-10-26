"""Type-safe, layered configuration for Axe.

Configuration is assembled from the following sources, lowest priority first:

1. ``src/axe/config/default_config.toml`` (bundled package data)
2. ``~/.config/axe/config.toml`` (user global)
3. ``<repo_root>/.axe/config.toml`` (workspace local)
4. An explicit ``config_path`` (``--config`` / ``AXE_CONFIG``), if provided
5. Environment variables (``.env`` plus ``AXE_*`` / provider key vars)

No API key is mandatory at import or startup time: the CLI can boot with any
provider, and missing keys are only reported when a provider call is actually
attempted. This keeps ``axe --help`` and local Ollama/vLLM setups working
without an OpenRouter/DeepSeek/Zhipu account.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

ThinkingMode = Literal["off", "low", "medium", "high"]
GuidedRemediationMode = Literal["interactive", "strict_block", "warn"]

DEFAULT_CONFIG_NAME = "default_config.toml"
CONFIG_ENV_VAR = "AXE_CONFIG"

# Escalation order for the built-in tiers. Custom tiers are appended.
TIER_ORDER = ("fast", "coder", "architect", "security")


class GeneralSettings(BaseModel):
    """Cross-cutting session defaults."""

    project_memory_file: str = ".axe/memory.json"
    max_iterations: int = Field(default=25, gt=0)
    snapshot_before_edit: bool = True


class ProviderConfig(BaseModel):
    """Configuration for one OpenAI-compatible (or local) provider."""

    model_config = ConfigDict(extra="allow")

    base_url: str = ""
    api_key_env: str = ""
    api_key: str = ""
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    default_headers: dict[str, str] = Field(default_factory=dict)

    def resolved_api_key(self, env: Mapping[str, str] | None = None) -> str:
        """Return the configured key, falling back to ``api_key_env``."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            source = os.environ if env is None else env
            return source.get(self.api_key_env, "")
        return ""


def _provider(base_url: str, api_key_env: str = "") -> ProviderConfig:
    return ProviderConfig(base_url=base_url, api_key_env=api_key_env)


class ProvidersSettings(BaseModel):
    """Known providers plus any additional user-defined providers."""

    model_config = ConfigDict(extra="allow")

    default: str = "openrouter"
    openrouter: ProviderConfig = Field(
        default_factory=lambda: _provider(
            "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
        )
    )
    deepseek: ProviderConfig = Field(
        default_factory=lambda: _provider(
            "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"
        )
    )
    zhipu: ProviderConfig = Field(
        default_factory=lambda: _provider(
            "https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"
        )
    )
    openai: ProviderConfig = Field(
        default_factory=lambda: _provider(
            "https://api.openai.com/v1", "OPENAI_API_KEY"
        )
    )
    moonshot: ProviderConfig = Field(
        default_factory=lambda: _provider(
            "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"
        )
    )
    fireworks: ProviderConfig = Field(
        default_factory=lambda: _provider(
            "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"
        )
    )
    local_vllm: ProviderConfig = Field(
        default_factory=lambda: _provider(
            "http://localhost:8000/v1", "VLLM_API_KEY"
        )
    )
    llamacpp: ProviderConfig = Field(
        default_factory=lambda: _provider(
            "http://localhost:8080/v1", "LLAMACPP_API_KEY"
        )
    )
    ollama: ProviderConfig = Field(
        default_factory=lambda: _provider("http://localhost:11434")
    )

    @model_validator(mode="after")
    def _coerce_extra_providers(self) -> "ProvidersSettings":
        extra = self.__pydantic_extra__ or {}
        for name, value in list(extra.items()):
            if isinstance(value, dict):
                extra[name] = ProviderConfig.model_validate(value)
        return self

    def get(self, name: str, default: ProviderConfig | None = None) -> ProviderConfig | None:
        """Return a provider by name (including user-defined providers)."""
        value = getattr(self, name, None)
        if isinstance(value, ProviderConfig):
            return value
        extra = self.__pydantic_extra__ or {}
        value = extra.get(name)
        if isinstance(value, dict):
            value = ProviderConfig.model_validate(value)
            extra[name] = value
        if isinstance(value, ProviderConfig):
            return value
        return default

    def names(self) -> list[str]:
        """All configured provider names, excluding the ``default`` pointer."""
        known = [
            "openrouter",
            "deepseek",
            "zhipu",
            "openai",
            "moonshot",
            "fireworks",
            "local_vllm",
            "llamacpp",
            "ollama",
        ]
        extra = sorted((self.__pydantic_extra__ or {}).keys())
        return [name for name in dict.fromkeys([*known, *extra])]


class JevSettings(BaseModel):
    """Jev decision-model routing configuration."""

    enabled: bool = True
    provider: str = "openrouter"
    model: str = "typesafe/jev-latest"
    endpoint: str = "https://openrouter.ai/api/v1/evaluate"
    api_key_env: str = "JEV_API_KEY"
    api_key: str = ""
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    risk_gate_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    ast_compression_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    timeout_ms: int = Field(default=500, gt=0)


class TierSettings(BaseModel):
    """One open-weight routing tier."""

    models: list[str] = Field(default_factory=list)
    thinking_mode: ThinkingMode = "off"
    max_tokens: int = Field(default=4096, gt=0)


def _tier(models: list[str], thinking_mode: ThinkingMode, max_tokens: int) -> TierSettings:
    return TierSettings(models=models, thinking_mode=thinking_mode, max_tokens=max_tokens)


class TierMap(BaseModel):
    """Named routing tiers, including user-defined tiers.

    ``default_fallback`` lives in the same TOML table as the tiers so users
    can write ``[router.tiers] default_fallback = "coder"``. Subscript access
    is supported for convenience::

        settings.router.tiers["architect"].models
    """

    model_config = ConfigDict(extra="allow")

    default_fallback: str = "coder"
    fast: TierSettings = Field(
        default_factory=lambda: _tier(["qwen/qwen3-coder-next"], "off", 4096)
    )
    coder: TierSettings = Field(
        default_factory=lambda: _tier(["deepseek/deepseek-v4"], "low", 8192)
    )
    architect: TierSettings = Field(
        default_factory=lambda: _tier(["deepseek/deepseek-v4-pro"], "high", 16384)
    )
    security: TierSettings = Field(
        default_factory=lambda: _tier(["zhipu/glm-5.3"], "medium", 8192)
    )

    @model_validator(mode="after")
    def _coerce_extra_tiers(self) -> "TierMap":
        extra = self.__pydantic_extra__ or {}
        for name, value in list(extra.items()):
            if isinstance(value, dict):
                extra[name] = TierSettings.model_validate(value)
        return self

    def _resolve(self, name: str) -> TierSettings | None:
        value = getattr(self, name, None)
        if isinstance(value, TierSettings):
            return value
        extra = self.__pydantic_extra__ or {}
        value = extra.get(name)
        if isinstance(value, dict):
            value = TierSettings.model_validate(value)
            extra[name] = value
        return value if isinstance(value, TierSettings) else None

    def __getitem__(self, name: str) -> TierSettings:
        tier = self._resolve(name)
        if tier is None:
            raise KeyError(name)
        return tier

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._resolve(name) is not None

    def get(self, name: str, default: TierSettings | None = None) -> TierSettings | None:
        tier = self._resolve(name)
        return tier if tier is not None else default

    def names(self) -> list[str]:
        """Return tier names in escalation order, including custom tiers."""
        extra = sorted(
            name
            for name in (self.__pydantic_extra__ or {})
            if name not in TIER_ORDER
        )
        return [*TIER_ORDER, *extra]

    def items(self) -> list[tuple[str, TierSettings]]:
        """Return ``(name, tier)`` pairs in escalation order."""
        return [(name, self[name]) for name in self.names()]


class RouterSettings(BaseModel):
    """Model routing: Jev decision model + open-weight escalation tiers."""

    jev: JevSettings = Field(default_factory=JevSettings)
    tiers: TierMap = Field(default_factory=TierMap)

    # Legacy failure-based escalation knobs (still used by ModelRouter).
    escalate_after_failures: int = Field(default=2, ge=1)
    reset_on_success: bool = True
    show_model_in_output: bool = True

    @property
    def default_fallback(self) -> str:
        return self.tiers.default_fallback

    def tier(self, name: str) -> TierSettings:
        """Return a tier by name; raise ``KeyError`` if unknown."""
        return self.tiers[name]

    def escalation_ladder(self) -> list[tuple[str, str]]:
        """Flatten tiers into a failure-escalation ladder.

        All built-in tiers are included so ``--tier security`` (and custom
        tiers) can be selected explicitly. Failure escalation walks the list
        in order, starting from the selected tier.
        """
        ladder: list[tuple[str, str]] = []
        for name in self.tiers.names():
            tier = self.tiers.get(name)
            if tier is None:
                continue
            ladder.extend((name, model) for model in tier.models)
        return ladder


class AirlockSettings(BaseModel):
    """Code-Airlock default MCP server and security middleware settings."""

    enabled: bool = True
    default_mcp_enabled: bool = True
    mcp_command: str = "code-airlock"
    mcp_args: list[str] = Field(default_factory=lambda: ["mcp", "serve"])
    redact_outbound_context: bool = True
    restore_on_file_write: bool = True
    guided_remediation_mode: GuidedRemediationMode = "interactive"
    pre_commit_exclude_globs: list[str] = Field(default_factory=list)


class SecuritySettings(BaseModel):
    """Security features applied to the agent loop."""

    airlock: AirlockSettings = Field(default_factory=AirlockSettings)


class CostSettings(BaseModel):
    """Cost tracking and budget limits."""

    session_budget: float = 1.00
    alert_at_percent: int = 80
    log_file: str = "~/.config/axe/costs.jsonl"


class AgentSettings(BaseModel):
    """Core agent behavior."""

    max_iterations: int = 25
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


class Settings(BaseSettings):
    """Root configuration — assembles all sub-models."""

    general: GeneralSettings = Field(default_factory=GeneralSettings)
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cost: CostSettings = Field(default_factory=CostSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)

    model_config = SettingsConfigDict(
        env_prefix="AXE_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

# Bare environment variables that map onto provider credentials.
_PROVIDER_KEY_ENV: dict[str, tuple[str, str]] = {
    "OPENROUTER_API_KEY": ("openrouter", "api_key"),
    "DEEPSEEK_API_KEY": ("deepseek", "api_key"),
    "ZHIPU_API_KEY": ("zhipu", "api_key"),
    "OPENAI_API_KEY": ("openai", "api_key"),
    "MOONSHOT_API_KEY": ("moonshot", "api_key"),
    "FIREWORKS_API_KEY": ("fireworks", "api_key"),
    "VLLM_API_KEY": ("local_vllm", "api_key"),
    "LLAMACPP_API_KEY": ("llamacpp", "api_key"),
    "JEV_API_KEY": ("jev", "api_key"),
}

# Bare environment variables that override provider endpoints.
_PROVIDER_URL_ENV: dict[str, tuple[str, str]] = {
    "OPENROUTER_BASE_URL": ("openrouter", "base_url"),
    "DEEPSEEK_BASE_URL": ("deepseek", "base_url"),
    "ZHIPU_BASE_URL": ("zhipu", "base_url"),
    "OPENAI_BASE_URL": ("openai", "base_url"),
    "MOONSHOT_BASE_URL": ("moonshot", "base_url"),
    "FIREWORKS_BASE_URL": ("fireworks", "base_url"),
    "VLLM_BASE_URL": ("local_vllm", "base_url"),
    "LLAMACPP_BASE_URL": ("llamacpp", "base_url"),
    "OLLAMA_HOST": ("ollama", "base_url"),
}


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _load_bundled_defaults() -> dict[str, Any]:
    resource = resources.files("axe.config").joinpath(DEFAULT_CONFIG_NAME)
    with resource.open("rb") as fh:
        return tomllib.load(fh)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (lists are replaced)."""
    for key, value in override.items():
        if (
            isinstance(value, Mapping)
            and isinstance(base.get(key), dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _set_nested(data: dict[str, Any], parts: list[str], value: Any) -> None:
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _coerce_env_value(raw: str) -> Any:
    """Best-effort TOML coercion: true/false, ints, floats, lists, strings."""
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except (tomllib.TOMLDecodeError, KeyError):
        return raw


def _user_config_path(env: Mapping[str, str]) -> Path:
    xdg = env.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        base = Path(xdg).expanduser()
    else:
        home = env.get("HOME", "").strip() or env.get("USERPROFILE", "").strip()
        base = Path(home).expanduser() if home else Path.home()
        base = base / ".config"
    return base / "axe" / "config.toml"


def _find_repo_root(start: Path) -> Path:
    """Walk upwards until a ``.git`` entry is found; fall back to ``start``."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def _workspace_config_path(cwd: Path) -> Path:
    return _find_repo_root(cwd) / ".axe" / "config.toml"


def _apply_environment(data: dict[str, Any], env: Mapping[str, str]) -> None:
    """Apply environment overrides on top of merged TOML data."""
    providers = data.setdefault("providers", {})
    if not isinstance(providers, dict):  # pragma: no cover - defensive
        providers = {}
        data["providers"] = providers

    # 1. Bare provider credentials and endpoints.
    for var, (provider, field) in _PROVIDER_KEY_ENV.items():
        value = env.get(var, "").strip()
        if not value:
            continue
        if provider == "jev":
            jev = data.setdefault("router", {}).setdefault("jev", {})
            jev[field] = value
        else:
            provider_data = providers.setdefault(provider, {})
            if isinstance(provider_data, dict):
                provider_data[field] = value

    for var, (provider, field) in _PROVIDER_URL_ENV.items():
        value = env.get(var, "").strip()
        if not value:
            continue
        if var == "OLLAMA_HOST" and not value.startswith(("http://", "https://")):
            value = f"http://{value}"
        provider_data = providers.setdefault(provider, {})
        if isinstance(provider_data, dict):
            provider_data[field] = value

    # 2. Generic AXE_* overrides. Nested keys use a double underscore:
    #    AXE_ROUTER__JEV__ENABLED=false
    #    AXE_GENERAL__MAX_ITERATIONS=40
    for name, raw in env.items():
        if not name.upper().startswith("AXE_") or name == CONFIG_ENV_VAR:
            continue
        suffix = name[len("AXE_"):]
        parts = [part.lower() for part in suffix.split("__") if part]
        if not parts:
            continue
        _set_nested(data, parts, _coerce_env_value(raw))

    # 3. Resolve each provider's ``api_key_env`` from the environment.
    for provider_name, provider_data in providers.items():
        if provider_name == "default" or not isinstance(provider_data, dict):
            continue
        key_env = str(provider_data.get("api_key_env", "")).strip()
        if key_env and env.get(key_env):
            provider_data["api_key"] = env[key_env]

    jev = data.get("router", {}).get("jev", {})
    if isinstance(jev, dict):
        key_env = str(jev.get("api_key_env", "")).strip()
        if key_env and env.get(key_env):
            jev["api_key"] = env[key_env]


def load_settings(
    config_path: str | Path | None = None,
    *,
    cwd: str | Path | None = None,
    include_user: bool = True,
    include_workspace: bool = True,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Load the layered Axe configuration.

    Args:
        config_path: Optional explicit config file. Overrides the workspace
            layer (but not environment variables).
        cwd: Directory used to locate ``.axe/config.toml``. Defaults to cwd.
        include_user: Set to ``False`` to ignore ``~/.config/axe/config.toml``.
        include_workspace: Set to ``False`` to ignore the repo-local file.
        env: Full environment mapping. Defaults to ``os.environ`` with ``.env``
            loaded into it. Pass an empty dict in tests for isolation.

    Returns:
        A fully-validated :class:`Settings` instance.
    """
    if env is None:
        load_dotenv(override=False)
        env_map: Mapping[str, str] = os.environ
    else:
        env_map = env

    # 1. Bundled defaults (packaged with the wheel / editable install).
    data: dict[str, Any] = _load_bundled_defaults()

    # 2. User global config.
    if include_user:
        user_path = _user_config_path(env_map)
        if user_path.is_file():
            _deep_merge(data, _load_toml(user_path))

    # 3. Workspace local config.
    working_dir = Path(cwd) if cwd is not None else Path.cwd()
    if include_workspace:
        workspace_path = _workspace_config_path(working_dir)
        if workspace_path.is_file():
            _deep_merge(data, _load_toml(workspace_path))

    # 4. Explicit override file (--config or AXE_CONFIG).
    explicit = config_path or env_map.get(CONFIG_ENV_VAR) or None
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Config file not found: {explicit_path}")
        _deep_merge(data, _load_toml(explicit_path))

    # 5. Environment variables win over every file layer.
    _apply_environment(data, env_map)

    # ``general.max_iterations`` is the modern location; keep the legacy
    # ``agent.max_iterations`` in sync when the user has not set it explicitly.
    agent_data = data.get("agent", {})
    general_data = data.get("general", {})
    if isinstance(agent_data, dict) and isinstance(general_data, dict):
        if "max_iterations" not in agent_data and "max_iterations" in general_data:
            agent_data["max_iterations"] = general_data["max_iterations"]

    return Settings(**data)
