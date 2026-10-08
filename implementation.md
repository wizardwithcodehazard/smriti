# smruti (स्मृति) — Technical Implementation Specification

> **A Biologically Inspired, Dual-Valence Cognitive Memory Engine for Autonomous AI Agents**
> *Bridging fast episodic buffering, offline consolidation, inhibitory anti-memories, and synaptic decay.*

---

## 1. Executive Summary & Problem Space

Current AI memory solutions (Cognee, Mem0, Zep, GraphRAG) approach agent memory as a relational or graph database with vector indexing bolted on. This architecture introduces severe cognitive limitations when paired with autonomous coding agents:

1. **Eager Ingestion Latency:** Every turn incurs heavy chunking, multiple LLM extraction calls, and graph compilation. Writes are slow, expensive, and block the agent's reasoning loop.
2. **Graph Bloat (Missing Consolidation):** Lacking a "sleep" or distillation phase, every casual interaction spawns noisy entity nodes (`User`, `Hello`, `Command`), rapidly cluttering graph traversal.
3. **Absence of Negative Knowledge (Anti-Memories):** Existing engines only store positive assertions (`X is Y`). They cannot represent failure trajectories, resulting in agents repeating identical mistakes, syntax errors, or compiler dead ends across sessions.
4. **Static Flat Weighting:** A file or note ingested months ago holds the exact same retrieval weight as a fact recalled two minutes ago. There is no biological forgetting curve or reinforcement mechanism.

**smruti** replaces this paradigm with a biological, dual-valence memory engine built on three decoupled tiers.

---

## 2. The Tri-Tier Biological Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   TIER 1: EPISODIC WORKING STREAM                     │
│  • Sub-millisecond append (Zero LLM overhead on write)                 │
│  • Causal trajectory: [Context] -> [Action] -> [Outcome] -> [Duration] │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼ (Triggered "Sleep" / Consolidation)
┌────────────────────────────────────────────────────────────────────────┐
│                   TIER 2: THE CONSOLIDATION ENGINE                     │
│  • Strips 90% verbatim token noise                                     │
│  • Distills episodes into generalized rules and heuristics             │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐ ┌───────────────────────────────┐
│     TIER 3A: POSITIVE KNOWLEDGE      │ │   TIER 3B: INHIBITORY MEMORY  │
│  • Verified solution paths           │ │   • Anti-memories / Dead ends │
│  • Ebbinghaus reinforcement & decay  │ │   • Exact failure signatures  │
│  • Spreading associative activation  │ │   • Proactive error prevention│
└──────────────────────────────────────┘ └───────────────────────────────┘
```

### Tier 1: Episodic Working Stream (Hippocampal Buffer)
- Append-only, transaction-safe event stream.
- Captures actions, tool invocations, shell commands, and execution results in real time.
- Operates at sub-millisecond latency using SQLite in Write-Ahead Logging (`WAL`) mode.
- **Zero LLM overhead during writes.** The agent is never stalled waiting for embeddings or entity extraction.

### Tier 2: Memory Consolidation Engine ("The Sleep Cycle")
- Periodic or threshold-triggered background worker (runs during idle agent periods or session boundaries).
- Replays recent episodic trajectories, separates verified successes from failed attempts, and prunes verbose raw text.
- Generalizes episodes into abstract schemas and heuristics.
- Applies biological synaptic decay sweeps across dormant memories.

### Tier 3: Dual-Valence Cortical Mesh
- **Tier 3A (Positive Excitatory Knowledge):**
  - Stores verified solution strategies, project invariants, and architectural rules.
  - Governed by Hebbian reinforcement: frequently recalled rules gain higher retrieval priority (`strength`).
  - Governed by the Ebbinghaus forgetting curve: unused memories fade exponentially over time.
- **Tier 3B (Inhibitory Anti-Memories — Vimarsha):**
  - Stores known dead ends, syntax traps, compiler failure signatures, and environmental collisions.
  - Active preflight gate: intercepts agent actions *before* execution if a matching failure signature is detected.

---

## 3. Technology Stack & Design Rationale

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Native to AI agent tooling, IDE plugins, and Python packaging. |
| **Episodic Stream (Buffer)** | **SQLite (WAL Mode)** | Zero-configuration, zero-daemon, embedded, sub-1ms ACID commits. |
| **Embedded Graph Storage** | **SQLite Adjacency Tables / Kùzu** | In-process relational adjacency mesh with optional Kùzu backend. Zero Docker requirements. |
| **Vector Engine (Local)** | **FastEmbed / sqlite-vec** | In-process CPU-friendly embeddings without requiring external vector microservices. |
| **IDE Protocol Interface** | **FastMCP (Model Context Protocol)** | First-class stdio integration for Cursor, Claude Code, Antigravity, and Windsurf. |
| **Developer CLI** | **Typer** | Intuitive terminal tool for memory inspection, manual consolidation, and system health audits. |

---

## 4. Package & Directory Structure

```text
smruti/
├── pyproject.toml              # Project dependencies: fastmcp, typer, pydantic, fastembed
├── README.md                   # Quickstart, architecture overview, and MCP configuration
├── implementation.md           # Technical specification and mathematical/logical models
├── tasks.md                    # Actionable task tracking and milestone progress
├── smruti/
│   ├── __init__.py             # Public Python API: smrutiEngine, Guard
│   ├── config.py               # Database paths, decay half-life, thresholds
│   ├── models.py               # Pydantic schemas (Episode, AntiMemory, Rule, Valence)
│   ├── storage/
│   │   ├── __init__.py
│   │   └── db.py               # SQLite WAL mode initialization, indexing, and migrations
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── stream.py           # Tier 1: Sub-millisecond episodic buffer (append-only)
│   │   ├── inhibitory.py       # Tier 3B: Exact & fuzzy failure signature matcher
│   │   ├── cortex.py           # Tier 3A: Positive rules, associative links, synaptic decay
│   │   └── consolidator.py     # Tier 2: "Sleep" consolidation & token pruning pipeline
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── mcp_server.py       # FastMCP Server (stdio tools for AI coding assistants)
│   │   └── cli.py              # Typer CLI (smruti init, inspect, sleep, audit)
└── tests/
    ├── test_stream.py          # Sub-ms append benchmark & schema checks
    ├── test_inhibition.py      # Preflight interception verification
    ├── test_decay.py           # Synaptic decay & reinforcement math verification
    ├── test_consolidation.py   # Token compression & schema distillation tests
    └── test_mcp.py             # MCP tool invocation tests
```

---

## 5. Mathematical & Algorithmic Formulation

### Biological Synaptic Decay
Memory strength degrades exponentially according to the Ebbinghaus retention model:

```text
effective_strength = base_strength * exp(-decay_rate * (current_time - last_accessed_at))
```

- When a rule is retrieved and successfully utilized by the agent:
  - `access_count` increments by 1.
  - `base_strength` increases (Hebbian reinforcement: `base_strength = min(1.0, base_strength + reinforcement_factor)`).
  - `last_accessed_at` updates to `current_time`.
- When `effective_strength` falls below `prune_threshold` (e.g. 0.15) and has not been accessed across multiple consolidation cycles, the rule is compacted or archived.

### Inhibitory Matching Logic
An inhibitory anti-memory matches an action candidate if:
1. **Exact Signature Match:** Exact command or pattern match with a previously failed trajectory.
2. **Contextual Collision:** Action contains known destructive flags or conflicting dependencies in the current environment context.
3. **Regex / Error Trace Substring:** Action resembles an error-inducing command under identical project conditions.

---

## 6. End-to-End System Flow

```
                      [ Agent Prompt / Plan ]
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ STEP 1: PREFLIGHT     │
                     │ INHIBITION CHECK      │
                     └───────────┬───────────┘
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼ (Match Found)                                   ▼ (Clear)
[ INTERCEPT AGENT ]                              [ AGENT EXECUTES ACTION ]
"Blocked: Known dead end: ..."                            │
                                                          ▼
                                                 ┌───────────────────────┐
                                                 │ STEP 2: FAST EPISODIC │
                                                 │ APPEND (Sub-1ms)      │
                                                 └───────────┬───────────┘
                                                             │
                                                             ▼ (Idle / Periodic)
                                                 ┌───────────────────────┐
                                                 │ STEP 3: CONSOLIDATION │
                                                 │ ("Sleep" Distillation)│
                                                 │ Strips token noise    │
                                                 │ Updates synaptic decay│
                                                 └───────────┬───────────┘
                                                             │
                                                             ▼
                                                 [ PERMANENT CORTICAL MESH ]
                                                 • Positive Heuristics (Tier 3A)
                                                 • Inhibitory Anti-Memories (Tier 3B)
```

---

## 7. Delivery Interfaces

### Interface A: FastMCP Server
Equips AI IDEs (Cursor, Claude Code, Antigravity) with four native tools over stdio:
1. `smruti_preflight_check(action, context)`: Intercepts actions before execution.
2. `smruti_record_episode(action, outcome, status, latency_ms)`: Logs execution results into Tier 1.
3. `smruti_recall_heuristics(query, limit)`: Injects top-weighted positive rules into working memory.
4. `smruti_trigger_sleep()`: Runs the consolidation cycle.

### Interface B: Typer CLI
Provides developers with terminal inspection and manual management:
- `smruti init`: Initializes `.smruti/` database in the current project root.
- `smruti status`: Displays active positive rules, anti-memories, and decay health.
- `smruti sleep`: Forces memory consolidation over uncompacted episodes.
- `smruti audit`: Shows chronological trajectory logs and blocked dead ends.
