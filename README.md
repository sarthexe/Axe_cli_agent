# CLI Agent

A terminal-based AI coding agent with a cost-aware multi-model architecture.

## Quick Start

1. Install dependencies:
   - `uv sync`
2. Configure environment:
   - `cp .env.example .env`
   - edit `.env` and set `OPENAI_API_KEY`
3. Run:
   - `uv run python main.py`
   - or `uv run python main.py "hello"`

## Day 1 Scope

- Click CLI entry point
- Config loading from `.env` (plus optional `~/.cli-agent/config.toml` or `--config`)
- OpenAI provider integration
- Basic REPL (`input -> GPT-4.1-mini -> print`)
