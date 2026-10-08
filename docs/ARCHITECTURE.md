# Smruti Technical Architecture

Smruti is a dual-valence, biologically inspired cognitive memory engine designed to eliminate hallucination loops, catastrophic shell commands, and agent amnesia without adding high LLM latency overhead to tool loops.

---

## 1. The Dual-Valence Cognitive Model

Traditional agent memory systems are strictly positive: they store text, documents, or graph entities describing what the system *knows*.

In contrast, biological brains rely heavily on **GABAergic inhibitory networks**: synaptic pathways dedicated exclusively to suppressing known catastrophic actions and dead ends.

Smruti implements this duality across three distinct temporal speeds:

```text
[ Agent Action Loop ]
         │
         ▼
┌───────────────────────────────┐
│ Tier 3B: Inhibitory Gate      │ < 2ms (Exact substring, dense vector, guarded regex)
│ (Catastrophic Prevention)     │
└───────────────┬───────────────┘
                │ Passed
                ▼
┌───────────────────────────────┐
│ Execution & Tier 1 Buffer     │ < 1ms SQLite WAL append (Zero LLM overhead)
│ (Sub-millisecond Episode Log) │
└───────────────┬───────────────┘
                │ Periodic / Task Completion
                ▼
┌───────────────────────────────┐
│ Tier 2: The Sleep Cycle       │ Offline background distillation & Ebbinghaus decay
│ (Consolidation Engine)        │
└───────────────┬───────────────┘
                │ Distills & Promotes
                ▼
┌───────────────────────────────┐
│ Tier 3A: Cortical Knowledge   │ Vectorized BLAS cosine search + Graph spreading
│ (Positive Heuristics & Rules) │
└───────────────────────────────┘
```

---

## 2. Temporal Tiers

### Tier 1 — Episodic Stream (`stream.py`)
- **Latency**: `< 1ms` per write.
- **Mechanism**: Every bash command, file edit, or tool invocation is written as an `Episode` row in SQLite.
- **Design Rule**: No LLMs or embedding calculations occur during episodic logging. This ensures the agent is never bottlenecked while actively performing work.

### Tier 2 — The Sleep Consolidation Cycle (`consolidator.py`)
- **Trigger**: Run offline via `smruti sleep`, on task milestone completion, or after reaching an episodic threshold.
- **Mechanism**:
  1. Replays unconsolidated episodes.
  2. Aggregates recurrent command failures and compiler crashes.
  3. Promotes repeated negative patterns into permanent Tier 3B anti-memories.
  4. Distills verified successful workflows into positive cortical rules.
  5. Computes biological decay and sweeps historical raw log entries.

### Tier 3A — Cortical Positive Mesh (`cortex.py`)
- Stores positive heuristics, coding conventions, architectural invariants, and user preferences.
- **Ebbinghaus Forgetting Curve**:
  Rule strength decays mathematically over time based on an exponential power law:
  
  effective_strength = base_strength * exp(-time_elapsed / (half_life * access_multiplier))
  
  Where `access_multiplier = 1.0 + ln(1 + access_count)`.
  Memories that are frequently recalled or reinforced resist decay, while unused rules naturally fade out of active context.

### Tier 3B — Inhibitory Reflex Mesh (`inhibitory.py`)
- Stores anti-memories (patterns representing known dead ends, recursive loops, or destructive operations).
- Evaluates candidate actions before execution through a multi-stage waterfall:
  1. **Literal Substring Match**: Exact pattern containment in candidate action (`< 0.1ms`).
  2. **Neural Dense Vector Match**: Cosine similarity match against active anti-memories (`< 2ms`).
  3. **Read-Only Intent Bypass**: Read-only diagnostic commands (e.g., `ls`, `grep`, `SELECT`) require a 0.95 similarity threshold before triggering a block, avoiding false positives on inspection tools.
  4. **Guarded Regex Fallback**: Sandboxed fallback for explicit shell wildcard patterns with syntax error shielding and ReDoS limits.

---

## 3. Contrastive Dense Semantic Anchors (`in_flight.py`)

Rather than relying on brittle regex patterns (`.*always.*`, `.*never.*`) to detect user instructions, Smruti uses **Contrastive Dense Semantic Anchors**:

1. Sentences are embedded into 384-dimensional vector space using quantized `bge-small-en-v1.5`.
2. The vector is evaluated against positive directive anchors (e.g., *"This is a strict constraint that must always be followed"*) and negative chitchat anchors (e.g., *"Here is the console output from the compiler"*).
3. If the directive similarity is significantly higher than chitchat (`similarity >= 0.45` and `directive_sim >= chitchat_sim`), Smruti autonomously classifies the input as a durable directive and registers it into the cortical mesh.

---

## 4. Vectorized BLAS Search (`embeddings.py`)

Smruti does not require an external vector database daemon (such as Qdrant or Chroma).

- Active rule vectors are stacked into an in-memory 2D numpy matrix (`N x 384`).
- Similarity retrieval uses vectorized BLAS dot products:
  
  similarities = (rule_matrix @ query_vector) / (norms * query_norm)
  
- Searches across thousands of rules complete in `< 1ms` on standard consumer laptop CPUs.

---

## 5. Storage Engine & Concurrency (`db.py`)

- **Database**: SQLite3 in WAL (Write-Ahead Logging) mode.
- **Concurrency Pragmas**:
  - `PRAGMA busy_timeout = 5000;`: Eliminates `database is locked` race conditions between parallel agent processes.
  - `PRAGMA cache_size = -8000;`: Allocates 8 MB in-memory cache for fast index lookups.
  - `PRAGMA synchronous = NORMAL;`: Optimizes write throughput while preserving ACID guarantees.
- **Pluggable Architecture**: Supports switching to PostgreSQL via the `SMRUTI_DATABASE_URL` environment variable.
