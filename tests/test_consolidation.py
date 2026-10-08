"""
test_consolidation.py - Unit tests for Tier 2: Memory Consolidation ("The Sleep Cycle").
"""

import tempfile
from pathlib import Path

import pytest

from smruti.config import smrutiConfig
from smruti.engine.consolidator import Consolidator
from smruti.engine.cortex import Cortex
from smruti.engine.inhibitory import InhibitoryGate
from smruti.engine.stream import StreamBuffer
from smruti.models import ActionStatus
from smruti.storage.db import DatabaseManager


@pytest.fixture
def memory_system():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = smrutiConfig(
            project_dir=Path(tmpdir),
            db_filename="test_consolidation.db",
            default_decay_rate=0.1,
            prune_threshold=0.20
        )
        db = DatabaseManager(config)
        stream = StreamBuffer(db)
        inhibitory = InhibitoryGate(db)
        cortex = Cortex(db, config)
        consolidator = Consolidator(db, config, stream, inhibitory, cortex)
        try:
            yield {
                "db": db,
                "config": config,
                "stream": stream,
                "inhibitory": inhibitory,
                "cortex": cortex,
                "consolidator": consolidator
            }
        finally:
            db.close()

def test_consolidation_detects_recurring_failures(memory_system):
    stream = memory_system["stream"]
    inhibitory = memory_system["inhibitory"]
    consolidator = memory_system["consolidator"]

    # Log 2 failures with command 'npm'
    stream.append("npm run build", result="ERR: ELIFECYCLE", status=ActionStatus.FAILURE)
    stream.append("npm run build", result="ERR: ELIFECYCLE", status=ActionStatus.FAILURE)

    report = consolidator.sleep()
    assert report["processed_episodes"] == 2
    assert report["promoted_anti_memories"] >= 1

    # Verify that an anti-memory was created
    anti_memories = inhibitory.list_all()
    assert len(anti_memories) >= 1
    assert any("npm" in a.pattern for a in anti_memories)

def test_consolidation_distills_success_after_failure(memory_system):
    stream = memory_system["stream"]
    cortex = memory_system["cortex"]
    consolidator = memory_system["consolidator"]

    # Causal sequence: failure -> success
    stream.append("python -m pip install pkg", result="permission denied", status=ActionStatus.FAILURE)
    stream.append("python -m pip install --user pkg", result="successfully installed", status=ActionStatus.SUCCESS)

    report = consolidator.sleep()
    assert report["promoted_positive_rules"] >= 1
    assert cortex.count() >= 1

    recalled = cortex.recall_rules("install")
    assert len(recalled) >= 1
    top_rule, _ = recalled[0]
    assert "--user" in top_rule.rule_text
