# agent-ledger

> Event-sourced state management and time-travel debugging engine for LLM agents.

## Overview
Production-grade event-sourcing framework for LLM agent architectures. Captures state mutations, tool executions, and LLM interactions as immutable event streams. Enables deterministic replay, runtime state patching, time-travel debugging, and Git-like branching for agent trajectories. Solves non-deterministic agent debugging by separating execution from state history.

## Backlog
- [ ] Define core event schemas and strictly typed state mutation interfaces.
- [ ] Implement thread-safe SQLite and in-memory event store backends.
- [ ] Build state projector engine to reconstruct agent state from event streams.
- [ ] Develop execution wrapper to intercept LLM calls and tool execution side-effects.
- [ ] Implement replay engine for execution recovery from historical checkpoints.
- [ ] Create branching mechanism to spawn alternative agent trajectories from state diffs.
- [ ] Develop CLI visualization tool to inspect event timelines and state transitions.
- [ ] Write integration test suite validating state projection, replay accuracy, and branching.

---
*Created and maintained autonomously by Claude Code.*
