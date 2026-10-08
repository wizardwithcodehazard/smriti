# smruti (स्मृति) — Engineering Tasks & Roadmap

> **Tracking Implementation, Verification, and Delivery Milestones**
> *Status Key: `[ ] Pending` | `[/] In Progress` | `[x] Completed`*

---

## Overall Progress

- [x] **Phase 1: Project Scaffolding & Episodic Working Stream (Tier 1)**
- [x] **Phase 2: Dual-Valence Cortical Mesh & Inhibitory Gate (Tier 3)**
- [x] **Phase 3: Memory Consolidation Pipeline ("The Sleep Cycle") (Tier 2)**
- [x] **Phase 4: FastMCP Protocol Server Integration**
- [x] **Phase 5: Developer CLI & End-to-End Test Suite**

---

## Detailed Task Breakdown

### Phase 1: Project Scaffolding & Episodic Working Stream (Tier 1)
*Goal: Establish the repository structure, dependency manifests, SQLite WAL storage foundation, and sub-millisecond append stream.*

- [x] **Task 1.1: Project Scaffolding & Packaging**
  - [x] Create `pyproject.toml` with dependencies (`fastmcp`, `typer`, `pydantic`, `fastembed`, `pytest`).
  - [x] Initialize package structure (`smruti/`, `smruti/storage/`, `smruti/engine/`, `smruti/interfaces/`, `tests/`).
  - [x] Configure `smruti/config.py` for dynamic path discovery (`.smruti/` directory) and parameter defaults.
- [x] **Task 1.2: Pydantic Data Models (`smruti/models.py`)**
  - [x] Implement `Episode` schema (id, timestamp, session_id, context, action, result, status, latency_ms).
  - [x] Implement `InhibitionResult` schema (passed, matched_signature, reason, suggested_fix).
  - [x] Implement `AntiMemory` schema (id, signature, pattern, context_tags, times_triggered, severity).
  - [x] Implement `CorticalRule` schema (id, rule_text, category, confidence, access_count, last_accessed_at, strength).
- [x] **Task 1.3: SQLite WAL Storage Substrate (`smruti/storage/db.py`)**
  - [x] Configure connection with `PRAGMA journal_mode = WAL;` and `PRAGMA synchronous = NORMAL;`.
  - [x] Write schema migration for `episodes`, `anti_memories`, and `cortical_rules` tables.
  - [x] Add indexing on `(session_id, timestamp)`, `(status)`, and `(signature)`.
- [x] **Task 1.4: High-Speed Episodic Stream API (`smruti/engine/stream.py`)**
  - [x] Implement `StreamBuffer.append(action, result, status, context, latency_ms)`.
  - [x] Benchmark write speed to enforce sub-millisecond commit latency (achieved ~0.139ms/append).
  - [x] Write unit test: `tests/test_stream.py`.

---

### Phase 2: Dual-Valence Cortical Mesh & Inhibitory Gate (Tier 3)
*Goal: Implement positive heuristic storage with biological decay, alongside proactive inhibitory anti-memory signature matching.*

- [x] **Task 2.1: Preflight Inhibitory Gate (`smruti/engine/inhibitory.py`)**
  - [x] Implement `InhibitoryGate.record_anti_memory(signature, pattern, reason, suggested_fix)`.
  - [x] Implement `InhibitoryGate.check_action(action, context) -> InhibitionResult`.
  - [x] Implement regex and substring signature matching against known fatal commands and errors.
  - [x] Write unit test: `tests/test_inhibition.py` (verify matching bad actions are blocked).
- [x] **Task 2.2: Positive Heuristics & Synaptic Decay (`smruti/engine/cortex.py`)**
  - [x] Implement `Cortex.add_rule(rule_text, category, base_strength)`.
  - [x] Implement biological decay formula:
    `effective_strength = base_strength * exp(-decay_rate * (current_time - last_accessed_at))`.
  - [x] Implement Hebbian reinforcement: increment `access_count` and boost `strength` on recall.
  - [x] Implement `Cortex.recall_rules(query, limit) -> list[CorticalRule]`.
  - [x] Write unit test: `tests/test_decay.py` (verify mathematical decay over simulated time).

---

### Phase 3: Memory Consolidation Pipeline ("The Sleep Cycle") (Tier 2)
*Goal: Build the offline distillation engine that cleans raw episodic logs, extracts generalized rules, updates anti-memories, and sweeps decaying facts.*

- [x] **Task 3.1: Episodic Compactor (`smruti/engine/consolidator.py`)**
  - [x] Implement `Consolidator.sleep(session_id=None)` routine.
  - [x] Extract failing trajectories from unprocessed episodes and convert recurring errors into permanent `AntiMemory` nodes.
  - [x] Extract successful trajectories and convert verified workflows into concise `CorticalRule` entries.
  - [x] Prune raw episodic token blobs older than the retention threshold to prevent database bloat.
- [x] **Task 3.2: Synaptic Pruning Sweep**
  - [x] Implement automatic archival/deletion for rules whose decayed strength drops below threshold (e.g. 0.15).
  - [x] Write unit test: `tests/test_consolidation.py`.

---

### Phase 4: FastMCP Protocol Server Integration
*Goal: Expose smruti as a native Model Context Protocol (MCP) server over standard I/O for AI IDEs (Cursor, Claude Code, Antigravity).*

- [x] **Task 4.1: FastMCP Server Core (`smruti/interfaces/mcp_server.py`)**
  - [x] Initialize `FastMCP("smruti")`.
  - [x] Tool 1: `smruti_preflight_check(action: str, context: str = "")` — blocks known failure patterns.
  - [x] Tool 2: `smruti_record_episode(action: str, outcome: str, status: str, latency_ms: float = 0.0)` — logs execution results into Tier 1.
  - [x] Tool 3: `smruti_record_anti_memory(signature: str, pattern: str, reason: str, suggested_fix: str = "")` — explicit negative learning.
  - [x] Tool 4: `smruti_recall_heuristics(query: str, limit: int = 5)` — retrieves top-weighted positive rules with Hebbian reinforcement.
  - [x] Tool 5: `smruti_trigger_sleep()` — triggers memory consolidation and synaptic pruning.
  - [x] Write integration test: `tests/test_mcp.py`.

---

### Phase 5: Developer CLI & End-to-End Simulation Suite
*Goal: Provide a production-ready command line tool for inspection and manual triggers, alongside full test coverage.*

- [x] **Task 5.1: Typer CLI (`smruti/interfaces/cli.py`)**
  - [x] Command `smruti init`: Initializes `.smruti/smruti.db` in the current working directory.
  - [x] Command `smruti status`: Displays active positive rules, anti-memories, and decay health statistics.
  - [x] Command `smruti sleep`: Manually triggers consolidation cycle.
  - [x] Command `smruti audit`: Prints chronological timeline of episodes and blocked actions.
  - [x] Command `smruti mcp`: Runs the FastMCP server over standard I/O.
- [x] **Task 5.2: Public Package Entrypoint (`smruti/__init__.py`)**
  - [x] Export `DatabaseManager`, `StreamBuffer`, `InhibitoryGate`, `Cortex`, `Consolidator`.
- [x] **Task 5.3: End-to-End Test Suite Execution**
  - [x] Run full pytest suite across stream, inhibition, decay, consolidation, CLI, and MCP server (16/16 tests passing).
