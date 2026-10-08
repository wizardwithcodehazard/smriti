"""
test_decay.py - Unit tests for Tier 3A: Biological Synaptic Decay & Hebbian Reinforcement.
"""

import tempfile
from pathlib import Path

import pytest

from smriti.config import SmritiConfig
from smriti.engine.cortex import Cortex
from smriti.storage.db import DatabaseManager


@pytest.fixture
def cortex():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SmritiConfig(
            project_dir=Path(tmpdir),
            db_filename="test_decay.db",
            default_decay_rate=0.1,  # 10% per hour
            prune_threshold=0.20,
            reinforcement_boost=0.15
        )
        db = DatabaseManager(config)
        try:
            yield Cortex(db, config)
        finally:
            db.close()

def test_add_and_recall_rule(cortex):
    rule = cortex.add_rule(
        rule_text="Always run tests with pytest before git push",
        category="testing"
    )
    assert rule.id is not None
    assert cortex.count() == 1

    recalled = cortex.recall_rules("pytest git push")
    assert len(recalled) == 1
    top_rule, score = recalled[0]
    assert top_rule.id == rule.id
    assert top_rule.access_count == 1  # reinforced

def test_exponential_decay_over_simulated_time(cortex):
    rule = cortex.add_rule(
        rule_text="Temporary hotfix for Python 3.10",
        category="hotfix",
        base_strength=1.0
    )
    
    # Simulate 0 hours elapsed
    t0 = rule.last_accessed_at
    s0 = rule.calculate_effective_strength(decay_rate=0.1, current_time=t0)
    assert s0 == 1.0

    # Simulate 20 hours elapsed (exp(-0.1 * 20) = exp(-2.0) ≈ 0.1353)
    t20 = t0 + (20 * 3600.0)
    s20 = rule.calculate_effective_strength(decay_rate=0.1, current_time=t20)
    assert s20 < 0.20
    assert s20 > 0.10

def test_prune_decayed_rules(cortex):
    rule = cortex.add_rule(
        rule_text="Obsolete dependency workaround",
        category="deprecated",
        base_strength=0.8
    )
    
    # At t0, rule strength is 0.8 > 0.20 threshold -> not pruned
    t0 = rule.last_accessed_at
    pruned_0 = cortex.prune_decayed_rules(current_time=t0)
    assert pruned_0 == 0
    assert cortex.count() == 1

    # At t + 40 hours, strength ≈ 0.8 * exp(-4) ≈ 0.0146 < 0.20 -> pruned
    t40 = t0 + (40 * 3600.0)
    pruned_40 = cortex.prune_decayed_rules(current_time=t40)
    assert pruned_40 == 1
    assert cortex.count() == 0
