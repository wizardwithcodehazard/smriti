"""
cli.py - Typer Command Line Interface for Smriti.
Provides commands for inspection, manual sleep cycles, memory retraction (forget), and FastMCP server.
Engineered for reliable cross-platform terminal output (including Windows cp1252 encodings).
"""

import sys

import typer

from smriti.config import get_config
from smriti.engine.consolidator import Consolidator
from smriti.engine.cortex import Cortex
from smriti.engine.inhibitory import InhibitoryGate
from smriti.engine.stream import StreamBuffer
from smriti.storage.db import get_db

app = typer.Typer(
    name="smriti",
    help="Smriti: Biologically Inspired Cognitive Memory Engine for Autonomous AI Agents.",
    add_completion=False
)

def _safe_echo(text: str) -> None:
    """Safely outputs text, encoding fallbacks for Windows cp1252 terminals."""
    try:
        typer.echo(text)
    except UnicodeEncodeError:
        # Fallback to ASCII representation
        encoding = getattr(sys.stdout, "encoding", "ascii") or "ascii"
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
        typer.echo(safe_text)

@app.command()
def init():
    """Initializes a local .smriti/ memory workspace in the current directory."""
    config = get_config()
    get_db(config)
    _safe_echo(f"Initialized Smriti memory database at: {config.db_path}")

@app.command()
def status():
    """Displays active memory statistics: episodes, positive rules, graph edges, and anti-memories."""
    config = get_config()
    db = get_db(config)
    stream = StreamBuffer(db)
    inhibitory = InhibitoryGate(db)
    cortex = Cortex(db, config)

    ep_count = stream.count()
    unconsolidated_count = len(stream.get_unconsolidated())
    rule_count = cortex.count()
    edges_count = len(cortex.list_edges())
    anti_memories = inhibitory.list_all(active_only=False)
    active_anti_count = sum(1 for a in anti_memories if a.is_active)

    _safe_echo("Smriti Memory System Status")
    _safe_echo(f"  * Workspace: {config.smriti_dir}")
    _safe_echo(f"  * Tier 1 Episodic Buffer: {ep_count} recorded episodes ({unconsolidated_count} pending consolidation)")
    _safe_echo(f"  * Tier 3A Cortical Mesh: {rule_count} active positive rules | {edges_count} associative graph edges")
    _safe_echo(f"  * Tier 3B Inhibitory Mesh: {active_anti_count} active anti-memories ({len(anti_memories)} total)")

@app.command()
def sleep(
    session_id: str | None = typer.Option(None, help="Optional session ID to consolidate"),
    project_root: str | None = typer.Option(None, help="Optional project root to scope anti-memories")
):
    """Runs Tier 2 Memory Consolidation ("The Sleep Cycle") over unconsolidated episodes."""
    config = get_config()
    db = get_db(config)
    stream = StreamBuffer(db)
    inhibitory = InhibitoryGate(db)
    cortex = Cortex(db, config)
    consolidator = Consolidator(db, config, stream, inhibitory, cortex)

    _safe_echo("[Smriti] Running memory consolidation cycle...")
    report = consolidator.sleep(session_id=session_id, project_root=project_root)
    _safe_echo(f"  + Processed {report['processed_episodes']} episodes")
    _safe_echo(f"  + Promoted {report['promoted_anti_memories']} new anti-memories")
    _safe_echo(f"  + Promoted {report['promoted_positive_rules']} new positive rules")
    _safe_echo(f"  + Pruned {report['pruned_decayed_rules']} decayed rules")
    _safe_echo(f"  + Cleared {report['pruned_raw_episodes']} historical raw episodes")

@app.command()
def forget(
    target: str = typer.Argument(..., help="Signature or pattern of anti-memory, or rule ID"),
    is_rule: bool = typer.Option(False, "--rule", "-r", help="Set to true if target is a cortical rule ID instead of anti-memory")
):
    """Retracts or deactivates an outdated anti-memory or cortical rule."""
    config = get_config()
    db = get_db(config)
    if is_rule:
        cortex = Cortex(db, config)
        if cortex.forget_rule(target):
            _safe_echo(f"[OK] Successfully deleted cortical rule '{target}'.")
        else:
            _safe_echo(f"[FAILED] No cortical rule found with ID '{target}'.")
    else:
        inhibitory = InhibitoryGate(db)
        if inhibitory.forget_anti_memory(target):
            _safe_echo(f"[OK] Successfully deactivated inhibitory anti-memory '{target}'.")
        else:
            _safe_echo(f"[FAILED] No anti-memory found with signature '{target}'.")

@app.command()
def audit(limit: int = typer.Option(10, help="Number of recent episodes to show")):
    """Displays recent episodic trajectory stream."""
    config = get_config()
    db = get_db(config)
    stream = StreamBuffer(db)
    episodes = stream.get_recent(limit=limit)

    if not episodes:
        _safe_echo("No episodes recorded in Tier 1 buffer yet.")
        return

    _safe_echo(f"Last {len(episodes)} Recorded Episodes:")
    for ep in episodes:
        status_icon = "+" if ep.status.value == "success" else "x"
        cons_mark = "[consolidated]" if ep.consolidated_at else "[pending]"
        _safe_echo(f"  [{status_icon}] ({ep.status.value.upper()}) {ep.action[:50]} -> {ep.result[:50]} {cons_mark}")

@app.command()
def mcp():
    """Runs the FastMCP server over standard I/O for Cursor / Claude Code / Antigravity."""
    from smriti.interfaces.mcp_server import run_server
    run_server()

if __name__ == "__main__":
    app()
