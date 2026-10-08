import pytest
from typer.testing import CliRunner

import smriti.config as config_mod
import smriti.storage.db as db_mod
from smriti.config import SmritiConfig
from smriti.interfaces.cli import app
from smriti.engine.inhibitory import InhibitoryGate
from smriti.engine.cortex import Cortex
from smriti.storage.db import get_db

runner = CliRunner()

@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    test_config = SmritiConfig(project_dir=tmp_path)
    monkeypatch.setattr(config_mod, "get_config", lambda: test_config)
    import smriti.interfaces.cli as cli_mod
    monkeypatch.setattr(cli_mod, "get_config", lambda: test_config)
    yield test_config

    if db_mod._default_db is not None:
        db_mod._default_db.close()
        db_mod._default_db = None

def test_cli_preflight_pass_and_block(cli_env):
    # Pass on clean environment
    res_pass = runner.invoke(app, ["preflight", "python main.py"])
    assert res_pass.exit_code == 0
    assert "[PASSED]" in res_pass.stdout

    # Insert an anti-memory
    db = get_db(cli_env)
    inhibitory = InhibitoryGate(db)
    inhibitory.record_anti_memory(
        signature="fatal_wipe",
        pattern=r"format\s+c:",
        reason="Disk wipe attempt",
        suggested_fix="Do not format system drive"
    )

    # Blocked action should return code 1 and block message
    res_blocked = runner.invoke(app, ["preflight", "format c:"])
    assert res_blocked.exit_code == 1
    assert "[BLOCKED BY SMRITI INHIBITORY GATE]" in res_blocked.stdout
    assert "fatal_wipe" in res_blocked.stdout

def test_cli_recall(cli_env):
    # Empty recall
    res_empty = runner.invoke(app, ["recall", "frontend"])
    assert res_empty.exit_code == 0
    assert "No relevant cortical rules found" in res_empty.stdout

    # Add a cortical rule
    db = get_db(cli_env)
    cortex = Cortex(db, cli_env)
    cortex.add_rule(
        rule_text="Run frontend dev server using npm run dev from frontend directory",
        category="frontend",
        confidence=0.9
    )

    # Recall should find it
    res_recall = runner.invoke(app, ["recall", "frontend dev"])
    assert res_recall.exit_code == 0
    assert "[SMRITI RECALLED HEURISTICS]" in res_recall.stdout
    assert "npm run dev" in res_recall.stdout
