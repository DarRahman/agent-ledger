# agent-ledger

**Event-sourced state management and time-travel debugging engine for LLM agents.**

---

## Overview

`agent-ledger` captures every state mutation, tool execution, and LLM interaction in your agent as an immutable event stream. This separation of execution from state history enables deterministic replay, runtime state patching, time-travel debugging, and Git-like branching for agent trajectories — solving the fundamental problem of debugging non-deterministic LLM agents.

## Key Features

- **Immutable Event Streams** — Every agent action (LLM calls, tool executions, state mutations) is recorded as a typed, sequenced event.
- **State Projection** — Reconstruct agent state at any point in time from the event log. Time-travel to any sequence number or timestamp.
- **Deterministic Replay** — Resume agent execution from any historical checkpoint. Replayed side-effects are served from history; only new calls execute.
- **Branching** — Spawn alternative agent trajectories from any checkpoint with optional state diffs. Explore "what if" scenarios without re-running upstream work.
- **Divergence Detection** — Replay automatically detects when new execution diverges from history, truncates stale events, and continues recording.
- **Thread-Safe Storage** — In-memory and SQLite (WAL mode) backends with lock-based concurrency.
- **Decorator & Context Manager APIs** — Instrument agent code with `@session.tool()`, `@session.llm()`, `@session.run()`, or `with session.track_tool_call()`.
- **CLI Inspector** — Terminal tool to list sessions, view event timelines, project state at any sequence, and diff states between two points.
- **Async Support** — All decorators and context managers work with both sync and async functions.

## Architecture