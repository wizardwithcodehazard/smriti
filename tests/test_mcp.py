"""
test_mcp.py - Integration tests for FastMCP server tools.
"""

import tempfile
from pathlib import Path

import pytest

from smriti.config import SmritiConfig
from smriti.interfaces.mcp_server import (
    smriti_preflight_check,
    smriti_recall_heuristics,
    smriti_record_anti_memory,
    smriti_record_episode,
    smriti_trigger_sleep,
)
from smriti.storage.db import DatabaseManager


@pytest.fixture
def mcp_env(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SmritiConfig(
            project_dir=Path(tmpdir),
            db_filename="test_mcp.db"
        )
        db = DatabaseManager(config)
        # Patch default db
        monkeypatch.setattr("smriti.storage.db._default_db", db)
        try:
            yield db
        finally:
            db.close()

def test_mcp_record_episode_and_preflight(mcp_env):
    # 1. Initially preflight check should pass
    res_pass = smriti_preflight_check(action="npm start")
    assert "[PASSED]" in res_pass

    # 2. Record an anti-memory explicitly
    res_anti = smriti_record_anti_memory(
        signature="npm_missing_deps",
        pattern="npm start",
        reason="Missing node_modules directory.",
        suggested_fix="Run npm install before npm start."
    )
    assert "Created inhibitory anti-memory" in res_anti

    # 3. Now preflight check MUST block npm start
    res_block = smriti_preflight_check(action="npm start")
    assert "[BLOCKED BY SMRITI INHIBITORY GATE]" in res_block
    assert "npm_missing_deps" in res_block
    assert "Run npm install" in res_block

def test_mcp_sleep_and_recall_heuristics(mcp_env):
    # Log 2 failures and a resolution
    smriti_record_episode(action="python setup.py install", outcome="deprecated", status="failure")
    smriti_record_episode(action="pip install -e .", outcome="success", status="success")

    # Run sleep consolidation via MCP tool
    sleep_res = smriti_trigger_sleep()
    assert "[CONSOLIDATION CYCLE COMPLETE]" in sleep_res

    # Recall heuristics
    recall_res = smriti_recall_heuristics("pip install")
    assert "SMRITI RECALLED HEURISTICS" in recall_res
    assert "pip install -e ." in recall_res

def test_mcp_forget_and_list_anti_memories(mcp_env):
    from smriti.interfaces.mcp_server import smriti_forget, smriti_list_anti_memories

    # Record anti-memory
    smriti_record_anti_memory(
        signature="bad_flag_rm",
        pattern="rm -rf /tmp/data",
        reason="Deletes temp data"
    )

    # List anti-memories
    listing = smriti_list_anti_memories()
    assert "bad_flag_rm" in listing

    # Verify blocked
    block_res = smriti_preflight_check("rm -rf /tmp/data")
    assert "[BLOCKED BY SMRITI INHIBITORY GATE]" in block_res

    # Forget anti-memory
    forget_res = smriti_forget("bad_flag_rm", "anti_memory")
    assert "Deactivated inhibitory anti-memory" in forget_res

    # Now must pass
    pass_res = smriti_preflight_check("rm -rf /tmp/data")
    assert "[PASSED]" in pass_res

