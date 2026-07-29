# agent-ledger

**Event-sourced state management and time-travel debugging engine for LLM agents.**

---

## Overview

`agent-ledger` captures every state mutation, tool execution, and LLM interaction in your agent as an immutable event stream. This separation of execution from state history unlocks deterministic replay, runtime state patching, time-travel debugging, and Git-like branching for agent trajectories — solving the fundamental problem of debugging non-deterministic LLM agents.

## Key Features

- **Immutable Event Streams** — Every agent action (LLM calls, tool executions, state mutations) is recorded as a typed, sequenced event.
- **State Projection** — Reconstruct agent state at any point in time from the event log. Time-travel to any sequence number or timestamp.
- **Deterministic Replay** — Resume agent execution from any historical checkpoint. Replayed side-effects are served from history; only new calls execute.
- **Branching** — Spawn alternative agent trajectories from any checkpoint with optional state diffs. Explore "what if" scenarios without re-running from scratch.
- **Divergence Detection** — Replay engine automatically detects when new execution diverges from history, truncates stale events, and continues cleanly.
- **Thread-Safe Stores** — In-memory and SQLite backends with locking and WAL support.
- **CLI Inspector** — Terminal tool to list sessions, view event timelines, project state at any point, and diff states between sequences.
- **Decorator & Context Manager APIs** — Instrument agent code with minimal boilerplate via `@session.run()`, `@session.tool()`, `@session.llm()`, or context managers.

## Architecture