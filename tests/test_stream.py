"""
test_stream.py - Unit tests and benchmark for Tier 1: Episodic Working Stream.
"""

import tempfile
import time
from pathlib import Path

import pytest

from smriti.config import SmritiConfig
from smriti.engine.stream import StreamBuffer
from smriti.models import ActionStatus
from smriti.storage.db import DatabaseManager


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SmritiConfig(
            project_dir=Path(tmpdir),
            db_filename="test_smriti.db"
        )
        db = DatabaseManager(config)
        try:
            yield db
        finally:
            db.close()

def test_db_wal_mode_enabled(temp_db):
    conn = temp_db.get_connection()
    journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert journal_mode.lower() == "wal", f"Expected WAL mode, got {journal_mode}"

def test_stream_append_and_retrieve(temp_db):
    stream = StreamBuffer(temp_db)
    
    episode = stream.append(
        action="lake build",
        result="error: unknown identifier 'x'",
        status=ActionStatus.FAILURE,
        context="building theorem 111291",
        latency_ms=12.5,
        session_id="session_01"
    )
    
    assert episode.id is not None
    assert episode.action == "lake build"
    assert episode.status == ActionStatus.FAILURE
    
    recent = stream.get_recent(limit=10, session_id="session_01")
    assert len(recent) == 1
    assert recent[0].id == episode.id
    assert recent[0].action == "lake build"

def test_stream_append_sub_millisecond_benchmark(temp_db):
    stream = StreamBuffer(temp_db, auto_consolidate=False)
    
    iterations = 100
    start = time.perf_counter()
    for i in range(iterations):
        stream.append(
            action=f"action_step_{i}",
            result="ok",
            status=ActionStatus.SUCCESS,
            session_id="benchmark"
        )
    total_time_s = time.perf_counter() - start
    avg_ms = (total_time_s / iterations) * 1000.0
    
    print(f"\nBenchmark: {iterations} appends in {total_time_s:.4f}s (Avg: {avg_ms:.3f}ms / append)")
    assert avg_ms < 5.0, f"Append latency too high: {avg_ms:.3f}ms"
    assert stream.count(session_id="benchmark") == iterations
