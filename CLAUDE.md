# CLAUDE.md

## Development Commands
* Run application: `python app.py` (or correct command for language)
* Run tests: `pytest`
* Lint/Format: `ruff check .`

## Codebase Architecture
Production-grade event-sourcing framework for LLM agent architectures. Captures state mutations, tool executions, and LLM interactions as immutable event streams. Enables deterministic replay, runtime state patching, time-travel debugging, and Git-like branching for agent trajectories. Solves non-deterministic agent debugging by separating execution from state history.

Initial file structure:
* `agent_ledger/__init__.py`
* `agent_ledger/models.py`
* `agent_ledger/store.py`
* `agent_ledger/projector.py`
* `agent_ledger/engine.py`
* `agent_ledger/cli.py`
* `tests/test_engine.py`
