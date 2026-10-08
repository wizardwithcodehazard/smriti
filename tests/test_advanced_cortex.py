"""
test_advanced_cortex.py - Comprehensive tests for neural vector embeddings,
semantic rule deduplication, associative graph spreading activation,
project-scoped inhibitory anti-memories, and consolidation idempotency.
"""

import time

import pytest

from smriti.config import SmritiConfig
from smriti.engine.consolidator import Consolidator
from smriti.engine.cortex import Cortex
from smriti.engine.inhibitory import InhibitoryGate
from smriti.engine.stream import StreamBuffer
from smriti.models import ActionStatus
from smriti.storage.db import DatabaseManager, close_all_dbs


@pytest.fixture
def test_env(tmp_path):
    config = SmritiConfig(
        project_dir=tmp_path,
        db_filename="test_advanced.db",
        dedup_similarity_threshold=0.80,
        associative_edge_threshold=0.60
    )
    db = DatabaseManager(config)
    yield config, db
    close_all_dbs()

def test_semantic_rule_recall_without_exact_keywords(test_env):
    """Verifies that vector embeddings enable recall even with ZERO exact keyword overlap."""
    config, db = test_env
    cortex = Cortex(db, config)

    # Insert positive rule
    cortex.add_rule(
        rule_text="Always use poetry run to execute commands within the isolated virtual environment",
        category="tooling"
    )

    # Query using completely different vocabulary
    query = "how to launch tasks inside the isolated python sandbox"
    recalled = cortex.recall_rules(query=query, limit=1)

    assert len(recalled) == 1
    top_rule, score = recalled[0]
    assert "poetry run" in top_rule.rule_text
    assert score > 1.0  # Boosted by neural semantic similarity

def test_semantic_rule_deduplication(test_env):
    """Verifies that semantically equivalent rules are reinforced rather than duplicated."""
    config, db = test_env
    cortex = Cortex(db, config)

    # 1. Add base rule
    r1 = cortex.add_rule(
        rule_text="Keep database connections short-lived and closed promptly in threaded servers",
        category="database",
        base_strength=0.7
    )
    assert cortex.count() == 1
    initial_access = r1.access_count

    # 2. Add semantically equivalent rule
    r2 = cortex.add_rule(
        rule_text="Ensure sqlite database connections are terminated immediately in multithreaded servers",
        category="database",
        base_strength=0.7
    )

    # Count must remain 1 — no duplicate row!
    assert cortex.count() == 1
    assert r2.id == r1.id
    # Base strength and access count must be reinforced
    assert r2.access_count == initial_access + 1
    assert r2.base_strength > 0.7

def test_associative_edges_and_spreading_activation(test_env):
    """Verifies that related rules form graph edges and spreading activation boosts linked nodes."""
    config, db = test_env
    cortex = Cortex(db, config)

    # Add two related rules
    r_migration = cortex.add_rule(
        rule_text="Use alembic migrations to update relational database schemas systematically",
        category="database"
    )
    r_rollback = cortex.add_rule(
        rule_text="Always test database schema rollback scripts before applying migrations",
        category="database"
    )

    edges = cortex.list_edges()
    assert len(edges) >= 1
    # Check that edge connects the two rules
    edge_pairs = {(e.rule_id_a, e.rule_id_b) for e in edges}
    assert (r_migration.id, r_rollback.id) in edge_pairs

    # Query focused on migrations
    recalled = cortex.recall_rules(query="alembic database migration", limit=2)
    assert len(recalled) == 2
    recalled_ids = [r.id for r, _ in recalled]
    assert r_migration.id in recalled_ids
    assert r_rollback.id in recalled_ids

def test_anti_memory_project_scoping(test_env):
    """Verifies that project-scoped anti-memories do not block unrelated workspaces."""
    config, db = test_env
    gate = InhibitoryGate(db)

    # Record anti-memory scoped only to frontend project
    gate.record_anti_memory(
        signature="frontend_build_node_v18",
        pattern="npm run build",
        reason="Requires Node 20+, fails under older versions",
        project_root="/workspace/frontend"
    )

    # Action executed in backend workspace -> must pass
    res_backend = gate.check_action(
        action="npm run build",
        project_root="/workspace/backend"
    )
    assert res_backend.passed is True

    # Action executed in frontend workspace -> must block
    res_frontend = gate.check_action(
        action="npm run build",
        project_root="/workspace/frontend"
    )
    assert res_frontend.passed is False
    assert res_frontend.matched_signature == "frontend_build_node_v18"

def test_anti_memory_forget_command(test_env):
    """Verifies that anti-memories can be forgotten / retracted when environments change."""
    config, db = test_env
    gate = InhibitoryGate(db)

    gate.record_anti_memory(
        signature="temporary_port_lock",
        pattern="python server.py --port 8000",
        reason="Port 8000 currently occupied"
    )

    # Verify initially blocked
    res_before = gate.check_action("python server.py --port 8000")
    assert res_before.passed is False

    # Forget anti-memory
    forgotten = gate.forget_anti_memory("temporary_port_lock")
    assert forgotten is True

    # Subsequent check must pass
    res_after = gate.check_action("python server.py --port 8000")
    assert res_after.passed is True

def test_consolidation_idempotency_and_marking(test_env):
    """Verifies that sleep cycle marks episodes as consolidated and is idempotent on repeat runs."""
    config, db = test_env
    stream = StreamBuffer(db)
    gate = InhibitoryGate(db)
    cortex = Cortex(db, config)
    consolidator = Consolidator(db, config, stream=stream, inhibitory=gate, cortex=cortex)

    # Append test episodes
    stream.append("docker compose up", result="port bind failed", status=ActionStatus.FAILURE)
    stream.append("docker compose up", result="port bind failed", status=ActionStatus.FAILURE)
    stream.append("docker compose -f docker-compose.prod.yml up", result="started", status=ActionStatus.SUCCESS)

    # First sleep run
    report1 = consolidator.sleep()
    assert report1["processed_episodes"] == 3
    assert report1["promoted_anti_memories"] >= 1
    assert report1["promoted_positive_rules"] >= 1

    # Verify all episodes are marked as consolidated
    unconsolidated = stream.get_unconsolidated()
    assert len(unconsolidated) == 0

    # Second sleep run immediately after -> must be idempotent (0 processed)
    report2 = consolidator.sleep()
    assert report2["processed_episodes"] == 0
    assert report2["promoted_anti_memories"] == 0
    assert report2["promoted_positive_rules"] == 0

def test_raw_retention_days_cleanup(test_env):
    """Verifies that historical raw episodes older than retention threshold are purged after consolidation."""
    config, db = test_env
    config.raw_retention_days = 1  # 1 day retention
    stream = StreamBuffer(db)
    gate = InhibitoryGate(db)
    cortex = Cortex(db, config)
    consolidator = Consolidator(db, config, stream=stream, inhibitory=gate, cortex=cortex)

    # Add old episode timestamped 3 days ago
    old_time = time.time() - (3 * 86400)
    ep_old = stream.append("old command", result="ok", status=ActionStatus.SUCCESS)
    # Mark it as consolidated at old_time and set timestamp
    conn = db.get_connection()
    with conn:
        conn.execute("UPDATE episodes SET timestamp = ?, consolidated_at = ? WHERE id = ?", (old_time, old_time, ep_old.id))

    # Add fresh episode today
    stream.append("fresh command", result="ok", status=ActionStatus.SUCCESS)

    assert stream.count() == 2

    # Run sleep
    report = consolidator.sleep()
    assert report["pruned_raw_episodes"] == 1
    # Only 1 fresh episode should remain
    assert stream.count() == 1
