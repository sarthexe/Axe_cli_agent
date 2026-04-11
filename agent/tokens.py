"""Tiktoken-based token counting for OpenAI models.

Uses the ``cl100k_base`` encoding which is compatible with all gpt-4.1
variants and o3-mini.  The encoder is cached at module level for
performance.
"""

from __future__ import annotations

import tiktoken

# Lazy singleton — created on first call
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Return the number of tokens in a plain text string."""
    return len(_get_encoder().encode(text))


def count_messages(messages: list[dict[str, object]]) -> int:
    """Return total token count for a list of chat messages.

    Accounts for the per-message overhead used by the OpenAI chat
    completions API (~4 tokens per message for role/name framing).
    """
    total = 0
    enc = _get_encoder()
    for msg in messages:
        # 4 tokens per message overhead (role + framing)
        total += 4
        content = msg.get("content")
        if isinstance(content, str):
            total += len(enc.encode(content))
        # tool_calls in assistant messages contain JSON — estimate
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    total += len(enc.encode(str(fn.get("name", ""))))
                    total += len(enc.encode(str(fn.get("arguments", ""))))
    # 2 tokens for reply priming
    total += 2
    return total
