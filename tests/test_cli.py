import pytest
from typer.testing import CliRunner

import smruti.config as config_mod
import smruti.storage.db as db_mod
from smruti.config import smrutiConfig
from smruti.interfaces.cli import app

runner = CliRunner()

@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    test_config = smrutiConfig(project_dir=tmp_path)

    monkeypatch.setattr(config_mod, "get_config", lambda: test_config)
    import smruti.interfaces.cli as cli_mod
    monkeypatch.setattr(cli_mod, "get_config", lambda: test_config)
    yield test_config

    # Cleanup connections
    if db_mod._default_db is not None:
        db_mod._default_db.close()
        db_mod._default_db = None

def test_cli_init_and_status(cli_env):
    res_init = runner.invoke(app, ["init"])
    assert res_init.exit_code == 0
    assert "Initialized smruti memory database" in res_init.stdout

    res_status = runner.invoke(app, ["status"])
    assert res_status.exit_code == 0
    assert "smruti Memory System Status" in res_status.stdout
    assert "0 recorded episodes" in res_status.stdout

def test_cli_audit_empty_and_populated(cli_env):
    res_audit_empty = runner.invoke(app, ["audit"])
    assert res_audit_empty.exit_code == 0
    assert "No episodes recorded" in res_audit_empty.stdout

    # Insert an episode via stream
    from smruti.engine.stream import StreamBuffer
    from smruti.models import ActionStatus
    from smruti.storage.db import get_db

    db = get_db(cli_env)
    stream = StreamBuffer(db)
    stream.append(action="test command", result="test output", status=ActionStatus.SUCCESS)

    res_audit = runner.invoke(app, ["audit"])
    assert res_audit.exit_code == 0
    assert "Last 1 Recorded Episodes" in res_audit.stdout
    assert "test command" in res_audit.stdout

def test_cli_sleep(cli_env):
    from smruti.engine.stream import StreamBuffer
    from smruti.models import ActionStatus
    from smruti.storage.db import get_db

    db = get_db(cli_env)
    stream = StreamBuffer(db)
    stream.append(action="pytest tests/", result="ModuleNotFoundError: pytest", status=ActionStatus.FAILURE)
    stream.append(action="pytest tests/", result="ModuleNotFoundError: pytest", status=ActionStatus.FAILURE)
    stream.append(action="python -m pytest tests/", result="2 passed", status=ActionStatus.SUCCESS)

    res_sleep = runner.invoke(app, ["sleep"])
    assert res_sleep.exit_code == 0
    assert "Running memory consolidation cycle" in res_sleep.stdout
    assert "Processed 3 episodes" in res_sleep.stdout
    assert "Promoted 1 new anti-memories" in res_sleep.stdout
    assert "Promoted 1 new positive rules" in res_sleep.stdout
