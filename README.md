# smruti (स्मृति)

> **Biologically Inspired, Dual-Valence Cognitive Memory for Autonomous AI Agents**

smruti equips autonomous coding and reasoning agents with an active, biological memory architecture:

1. **Tier 1 (Episodic Working Stream):** Sub-millisecond, append-only SQLite WAL stream. Zero LLM latency while acting.
2. **Tier 2 (The "Sleep" Consolidation Engine):** Periodic background distillation that strips 90% verbatim token noise into schemas and rules.
3. **Tier 3 (Dual-Valence Cortical Mesh):**
   - **Positive Heuristics (Tier 3A):** Verified solution paths with biological Ebbinghaus decay and Hebbian reinforcement.
   - **Inhibitory Anti-Memories (Tier 3B):** Active preflight interception gate that prevents agents from repeating known fatal mistakes and compiler errors.

---

## Quickstart

### Installation
```bash
pip install -e .
```

### CLI Inspection
```bash
smruti init      # Initialize .smruti/ database in your project
smruti status    # View positive rules, anti-memories, and decay health
smruti sleep     # Run offline consolidation over recent episodes
smruti audit     # Inspect chronological agent trajectories
```

### MCP (Model Context Protocol) Server for Cursor / Claude Code / Antigravity
Add to your MCP configuration (`mcpServers`):
```json
{
  "smruti": {
    "command": "smruti",
    "args": ["mcp"]
  }
}
```
