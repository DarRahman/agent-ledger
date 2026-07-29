# agent-ledger

**Event-sourced state management and time-travel debugging engine for LLM agents.**

---

## Overview

`agent-ledger` captures every state mutation, tool execution, and LLM interaction in your agent as an immutable event stream. This separation of execution from state history unlocks deterministic replay, runtime state patching, time-travel debugging, and Git-like branching for agent trajectories — solving the fundamental problem of debugging non-deterministic LLM agents.

## Key Features

- **Immutable Event Streams** — Every agent action recorded as a typed, sequenced event with full payload capture
- **Deterministic Replay** — Resume agent execution from any historical checkpoint, replaying cached side-effects without re-executing LLM calls or tools
- **Time-Travel Debugging** — Project agent state at any point in the event timeline using sequence numbers or timestamps
- **Git-like Branching** — Spawn alternative agent trajectories from any checkpoint with optional state diffs applied at the fork point
- **Divergence Detection** — Automatic detection when replayed execution diverges from history, with store truncation and seamless recovery
- **Execution Interception** — Context managers and decorators to transparently capture LLM calls, tool executions, and state mutations
- **Thread-Safe Storage** — In-memory and SQLite (WAL mode) event store backends with full thread safety
- **CLI Inspector** — Terminal tool to list sessions, view event timelines, project state, and diff state between sequence points
- **Async Support** — Full `async/await` support for all decorators and context managers

## Architecture