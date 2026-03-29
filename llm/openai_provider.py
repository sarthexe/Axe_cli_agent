"""OpenAI provider implementation."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from llm.provider import LLMProvider, LLMResponse, ToolCall


class OpenAIProvider(LLMProvider):
    """OpenAI SDK implementation for the common provider interface."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or llm.openai.api_key in config.")

        self.model = model
        self.temperature = temperature
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        input_messages: list[dict[str, Any]] = []

        if messages is not None:
            input_messages.extend(messages)
        else:
            if system_prompt:
                input_messages.append({"role": "system", "content": system_prompt})

            input_messages.append({"role": "user", "content": prompt})

        request: dict[str, Any] = {
            "model": self.model,
            "messages": input_messages,
            "temperature": self.temperature,
        }
        if tools:
            request["tools"] = self._build_tools(tools)

        response = self._client.chat.completions.create(**request)
        response_message = response.choices[0].message
        
        response_text = response_message.content or ""
        tool_calls = self._extract_tool_calls(response_message)
        response_raw = response.model_dump()
        
        # Capture raw assistant message for tool feedback loop
        if tool_calls:
            response_raw["message"] = response_message.model_dump(exclude_none=True)

        return LLMResponse(
            text=response_text.strip(),
            tool_calls=tool_calls,
            raw=response_raw,
        )

    def _build_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for tool in tools:
            name = str(tool.get("name", "")).strip()
            if not name:
                continue
            spec: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description", "")),
                }
            }
            parameters = tool.get("parameters")
            if isinstance(parameters, dict):
                spec["function"]["parameters"] = parameters
            prepared.append(spec)
        return prepared

    def _extract_tool_calls(self, message: ChatCompletionMessage) -> list[ToolCall]:
        calls: list[ToolCall] = []
        if not message.tool_calls:
            return calls

        for item in message.tool_calls:
            if item.type != "function":
                continue

            name = item.function.name
            raw_arguments = item.function.arguments
            
            parsed_arguments: dict[str, Any] = {}
            if isinstance(raw_arguments, str):
                try:
                    parsed_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    parsed_arguments = {}
            elif isinstance(raw_arguments, dict):
                parsed_arguments = raw_arguments

            calls.append(ToolCall(
                name=name,
                arguments=parsed_arguments,
                id=item.id,
                raw=item.model_dump(),
            ))

        return calls
