"""
test_inhibition.py - Unit tests for Tier 3B: Inhibitory Anti-Memories & Preflight Gate.
"""

import tempfile
from pathlib import Path

import pytest

from smruti.config import smrutiConfig
from smruti.engine.inhibitory import InhibitoryGate
from smruti.storage.db import DatabaseManager


@pytest.fixture
def gate():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = smrutiConfig(
            project_dir=Path(tmpdir),
            db_filename="test_inhibition.db"
        )
        db = DatabaseManager(config)
        try:
            yield InhibitoryGate(db)
        finally:
            db.close()

def test_safe_action_passes(gate):
    res = gate.check_action("git status")
    assert res.passed is True
    assert res.matched_signature is None

def test_recorded_anti_memory_blocks_action(gate):
    gate.record_anti_memory(
        signature="port_8000_collision",
        pattern="uvicorn.*--port 8000",
        reason="Port 8000 is occupied by running Docker backend container.",
        suggested_fix="Use --port 8080 or stop the docker container first."
    )
    
    # 1. Matching command should be BLOCKED
    res = gate.check_action("uvicorn main:app --port 8000")
    assert res.passed is False
    assert res.matched_signature == "port_8000_collision"
    assert "8000" in res.reason
    assert "8080" in res.suggested_fix

    # 2. Non-matching command should PASS
    res_ok = gate.check_action("uvicorn main:app --port 8080")
    assert res_ok.passed is True

def test_anti_memory_trigger_count_increments(gate):
    gate.record_anti_memory(
        signature="uncompiled_decide_depth",
        pattern="decide",
        reason="decide tactic exceeds maxRecDepth on large integers.",
        suggested_fix="Formulate explicit factor certificate or use decide +native."
    )
    
    # Block twice
    gate.check_action("by decide")
    gate.check_action("exact by decide")
    
    all_anti = gate.list_all()
    assert len(all_anti) == 1
    assert all_anti[0].times_triggered == 2
