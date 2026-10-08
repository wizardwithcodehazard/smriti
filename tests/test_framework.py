import pytest
from smriti import Smriti, SmritiConfig, SmritiInhibitionError, DatabaseManager
from smriti.models import ActionStatus

@pytest.fixture
def smriti_client(tmp_path):
    config = SmritiConfig(project_dir=tmp_path)
    db = DatabaseManager(config)
    client = Smriti(config=config, db=db)
    yield client
    db.close()

def test_smriti_facade_basic_lifecycle(smriti_client):
    # 1. Preflight on clean slate
    pre = smriti_client.preflight("npm run build")
    assert pre.passed is True

    # 2. Record episode
    ep = smriti_client.record("npm run build", outcome="Build succeeded", status="success")
    assert ep.id is not None
    assert ep.status == ActionStatus.SUCCESS

    # 3. Record anti-memory
    anti = smriti_client.record_anti_memory(
        signature="rm_rf_root",
        pattern=r"rm\s+-rf\s+/",
        reason="Catastrophic root deletion",
        suggested_fix="Target specific project directory"
    )
    assert anti.signature == "rm_rf_root"

    # 4. Preflight intercepts fatal pattern
    blocked = smriti_client.preflight("rm -rf /")
    assert blocked.passed is False
    assert "Catastrophic" in blocked.reason

    # 5. Sleep consolidation
    report = smriti_client.sleep()
    assert isinstance(report, dict)
    assert "processed_episodes" in report

    # 6. Forget anti-memory
    success = smriti_client.forget("rm_rf_root")
    assert success is True
    # Verify it now passes
    after_forget = smriti_client.preflight("rm -rf /")
    assert after_forget.passed is True

def test_smriti_guard_decorator_interception(smriti_client):
    smriti_client.record_anti_memory(
        signature="drop_prod_db",
        pattern=r"DROP\s+DATABASE\s+production",
        reason="Production database drop forbidden",
        suggested_fix="Operate only on staging"
    )

    # Protected tool function
    @smriti_client.guard()
    def run_query(cmd: str):
        return f"Executed: {cmd}"

    # Safe call passes and records success
    res_safe = run_query("SELECT * FROM users")
    assert res_safe == "Executed: SELECT * FROM users"

    # Blocked call returns interception report
    res_blocked = run_query("DROP DATABASE production")
    assert isinstance(res_blocked, dict)
    assert res_blocked["status"] == "blocked"
    assert "Production database drop forbidden" in res_blocked["reason"]

    # Blocked call with raise_on_blocked=True
    @smriti_client.guard(raise_on_blocked=True)
    def run_query_strict(cmd: str):
        return f"Executed: {cmd}"

    with pytest.raises(SmritiInhibitionError):
        run_query_strict("DROP DATABASE production")
