"""Base tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Return the JSON schema for the tool parameters."""
        raise NotImplementedError

    def validate(self, arguments: dict[str, Any]) -> None:
        """Validate the arguments before execution. Raises ValueError if invalid."""
        pass

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool and return its string output."""
        raise NotImplementedError
