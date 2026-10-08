"""
mcp_server.py - FastMCP Server for Smriti Cognitive Memory Engine.
Exposes preflight gate, sub-ms episodic recording, positive heuristics, and sleep cycle
to AI coding environments (Cursor, Claude Code, Antigravity, Windsurf) over stdio.
"""


from fastmcp import FastMCP

from smriti.engine.consolidator import Consolidator
from smriti.engine.cortex import Cortex
from smriti.engine.inhibitory import InhibitoryGate
from smriti.engine.stream import StreamBuffer
from smriti.models import ActionStatus
from smriti.storage.db import DatabaseManager, get_db

# Initialize FastMCP Server
mcp = FastMCP("Smriti Cognitive Memory")

_cached_db: DatabaseManager | None = None
_cached_bundle: tuple[StreamBuffer, InhibitoryGate, Cortex, Consolidator] | None = None

def _get_engine():
    global _cached_db, _cached_bundle
    db = get_db()
    if _cached_bundle is None or _cached_db is not db:
        _cached_db = db
        stream = StreamBuffer(db)
        inhibitory = InhibitoryGate(db)
        cortex = Cortex(db)
        consolidator = Consolidator(db, stream=stream, inhibitory=inhibitory, cortex=cortex)
        _cached_bundle = (stream, inhibitory, cortex, consolidator)
    return _cached_bundle

def _parse_status(status_str: str) -> ActionStatus:
    s = status_str.strip().lower()
    if s == "success":
        return ActionStatus.SUCCESS
    elif s == "failure":
        return ActionStatus.FAILURE
    elif s == "error":
        return ActionStatus.ERROR
    elif s == "intercepted":
        return ActionStatus.INTERCEPTED
    return ActionStatus.FAILURE

@mcp.tool()
def smriti_preflight_check(action: str, context: str = "", project_root: str | None = None) -> str:
    """
    Active Preflight Gate: Call this BEFORE running any terminal command, code modification, or tool.
    Checks against Tier 3B inhibitory anti-memories to prevent known dead ends, recursive loops, and crashes.
    """
    _, inhibitory, _, _ = _get_engine()
    res = inhibitory.check_action(action, context, project_root=project_root)
    if not res.passed:
        return (
            f"🚫 [BLOCKED BY SMRITI INHIBITORY GATE]\n"
            f"Signature: {res.matched_signature}\n"
            f"Severity: {res.severity.upper()}\n"
            f"Reason: {res.reason}\n"
            f"Suggested Fix: {res.suggested_fix or 'Change parameters to avoid known dead end.'}"
        )
    return "✅ [PASSED] No known failure signatures detected."

@mcp.tool()
def smriti_record_episode(
    action: str,
    outcome: str = "",
    status: str = "success",
    latency_ms: float = 0.0,
    context: str = ""
) -> str:
    """
    Logs an action and its outcome into the Tier 1 episodic stream with sub-millisecond commit latency.
    Zero LLM latency overhead during execution.
    Supports all statuses: success, failure, error, intercepted.
    """
    stream, _, _, _ = _get_engine()
    act_status = _parse_status(status)
    ep = stream.append(
        action=action,
        result=outcome,
        status=act_status,
        context=context,
        latency_ms=latency_ms
    )
    return f"Logged episode {ep.id[:8]} [status={ep.status.value}] in Tier 1 buffer."

@mcp.tool()
def smriti_record_anti_memory(
    signature: str,
    pattern: str,
    reason: str,
    suggested_fix: str = "",
    project_root: str | None = None
) -> str:
    """
    Explicitly stores an inhibitory anti-memory (dead end / fatal bug) to intercept future runs.
    Can be scoped to a specific project_root workspace.
    """
    _, inhibitory, _, _ = _get_engine()
    anti = inhibitory.record_anti_memory(
        signature=signature,
        pattern=pattern,
        reason=reason,
        suggested_fix=suggested_fix or None,
        project_root=project_root
    )
    scope_str = f" [scoped to {project_root}]" if project_root else " [global]"
    return f"Created inhibitory anti-memory '{anti.signature}' matching pattern '{anti.pattern}'{scope_str}."

@mcp.tool()
def smriti_recall_heuristics(query: str = "", limit: int = 5) -> str:
    """
    Recalls top-weighted positive rules (Tier 3A) governed by vector semantic similarity,
    associative graph spreading activation, and biological Ebbinghaus decay.
    Reinforces accessed memories.
    """
    _, _, cortex, _ = _get_engine()
    recalled = cortex.recall_rules(query=query, limit=limit, reinforce=True)
    if not recalled:
        return "No relevant cortical rules found in active memory."

    lines = ["🧠 [SMRITI RECALLED HEURISTICS]"]
    for i, (rule, score) in enumerate(recalled, 1):
        lines.append(
            f"{i}. [{rule.category}] {rule.rule_text} "
            f"(effective_strength: {score:.2f}, access_count: {rule.access_count})"
        )
    return "\n".join(lines)

@mcp.tool()
def smriti_trigger_sleep(project_root: str | None = None) -> str:
    """
    Triggers Tier 2 Memory Consolidation ("The Sleep Cycle"):
    Replays unconsolidated episodes, promotes recurring failures into anti-memories,
    distills verified successes into positive rules, sweeps historical buffers, and decays dormant facts.
    """
    _, _, _, consolidator = _get_engine()
    report = consolidator.sleep(project_root=project_root)
    return (
        f"🌙 [CONSOLIDATION CYCLE COMPLETE]\n"
        f"• Episodes Analyzed: {report['processed_episodes']}\n"
        f"• New Anti-Memories Promoted: {report['promoted_anti_memories']}\n"
        f"• Positive Rules Promoted: {report['promoted_positive_rules']}\n"
        f"• Decayed Rules Pruned: {report['pruned_decayed_rules']}\n"
        f"• Historical Episodes Cleared: {report['pruned_raw_episodes']}"
    )

@mcp.tool()
def smriti_forget(target: str, target_type: str = "anti_memory") -> str:
    """
    Explicitly removes or deactivates an outdated anti-memory or cortical rule.
    target_type: 'anti_memory' (signature or ID) or 'rule' (rule ID).
    """
    _, inhibitory, cortex, _ = _get_engine()
    t_type = target_type.strip().lower()
    if t_type in {"anti_memory", "anti", "antimemory"}:
        success = inhibitory.forget_anti_memory(target)
        if success:
            return f"Deactivated inhibitory anti-memory '{target}'."
        return f"No active anti-memory matched '{target}'."
    elif t_type in {"rule", "positive_rule"}:
        success = cortex.forget_rule(target)
        if success:
            return f"Removed cortical rule '{target}'."
        return f"No cortical rule matched ID '{target}'."
    else:
        return f"Unknown target_type '{target_type}'. Use 'anti_memory' or 'rule'."

@mcp.tool()
def smriti_list_anti_memories(project_root: str | None = None) -> str:
    """
    Lists all active inhibitory anti-memories. Helps the agent inspect why an action might be blocked.
    """
    _, inhibitory, _, _ = _get_engine()
    memories = inhibitory.list_all(active_only=True, project_root=project_root)
    if not memories:
        return "No active inhibitory anti-memories found."

    lines = [f"🛡️ [ACTIVE SMRITI ANTI-MEMORIES] ({len(memories)} total)"]
    for m in memories:
        scope = f" [project: {m.project_root}]" if m.project_root else " [global]"
        fix = f" | Fix: {m.suggested_fix}" if m.suggested_fix else ""
        lines.append(f"• [{m.signature}] pattern: '{m.pattern}' (triggered: {m.times_triggered}x){scope}{fix}")
    return "\n".join(lines)

def run_server():
    """Runs the FastMCP server over standard I/O."""
    mcp.run()

if __name__ == "__main__":
    run_server()
