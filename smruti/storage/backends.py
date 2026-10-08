"""
backends.py - Pluggable Storage Backend Interface for smruti.
Supports local SQLite (default) and PostgreSQL for remote team-shared memory clusters.
"""

import abc
import logging
import sqlite3
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class StorageConnection(Protocol):
    """Protocol for unified database connection behavior."""
    def execute(self, sql: str, parameters: Any = ...) -> Any: ...
    def executescript(self, sql_script: str) -> Any: ...
    def commit(self) -> None: ...
    def close(self) -> None: ...


class BaseStorageBackend(abc.ABC):
    """Abstract base class for smruti storage backends."""

    @abc.abstractmethod
    def get_connection(self) -> Any:
        """Returns an active connection for executing queries."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Closes all open connections."""
        pass

    @abc.abstractmethod
    def ping(self) -> bool:
        """Checks connection health."""
        pass


class SQLiteStorageBackend(BaseStorageBackend):
    """Standard thread-safe SQLite WAL backend for single-machine environments."""

    def __init__(self, db_manager):
        self.db_manager = db_manager

    def get_connection(self) -> sqlite3.Connection:
        return self.db_manager.get_connection()

    def close(self) -> None:
        self.db_manager.close()

    def ping(self) -> bool:
        try:
            conn = self.get_connection()
            cur = conn.execute("SELECT 1;")
            return cur.fetchone()[0] == 1
        except Exception:
            return False


class PostgresStorageBackend(BaseStorageBackend):
    """
    PostgreSQL backend for centralized team sharing of anti-memories and cortical rules.
    Connects to any PostgreSQL instance, Supabase, Neon, or RDS database.
    Requires psycopg (v3) or psycopg2.
    """

    POSTGRES_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at DOUBLE PRECISION NOT NULL,
        description TEXT
    );

    CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        timestamp DOUBLE PRECISION NOT NULL,
        session_id TEXT NOT NULL,
        context TEXT,
        action TEXT NOT NULL,
        result TEXT,
        status TEXT NOT NULL,
        latency_ms DOUBLE PRECISION NOT NULL,
        metadata_json TEXT,
        consolidated_at DOUBLE PRECISION
    );

    CREATE TABLE IF NOT EXISTS anti_memories (
        id TEXT PRIMARY KEY,
        signature TEXT UNIQUE NOT NULL,
        pattern TEXT NOT NULL,
        reason TEXT NOT NULL,
        suggested_fix TEXT,
        context_tags TEXT,
        times_triggered INTEGER DEFAULT 0,
        severity TEXT DEFAULT 'high',
        created_at DOUBLE PRECISION NOT NULL,
        last_seen DOUBLE PRECISION NOT NULL,
        valence TEXT DEFAULT 'inhibitory',
        project_root TEXT,
        is_active INTEGER DEFAULT 1,
        embedding TEXT,
        expires_at DOUBLE PRECISION
    );

    CREATE TABLE IF NOT EXISTS cortical_rules (
        id TEXT PRIMARY KEY,
        rule_text TEXT NOT NULL,
        category TEXT DEFAULT 'general',
        confidence DOUBLE PRECISION DEFAULT 1.0,
        access_count INTEGER DEFAULT 0,
        created_at DOUBLE PRECISION NOT NULL,
        last_accessed_at DOUBLE PRECISION NOT NULL,
        base_strength DOUBLE PRECISION DEFAULT 1.0,
        valence TEXT DEFAULT 'positive',
        embedding TEXT,
        source_episode_ids TEXT
    );

    CREATE TABLE IF NOT EXISTS rule_edges (
        rule_id_a TEXT NOT NULL,
        rule_id_b TEXT NOT NULL,
        weight DOUBLE PRECISION NOT NULL,
        created_at DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (rule_id_a, rule_id_b)
    );
    """

    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self._conn = None
        self._driver = self._detect_driver()

    def _detect_driver(self):
        try:
            import psycopg
            return "psycopg"
        except ImportError:
            try:
                import psycopg2
                return "psycopg2"
            except ImportError:
                return None

    def get_connection(self):
        if self._driver is None:
            raise ImportError(
                "PostgreSQL backend requires 'psycopg' or 'psycopg2'. "
                "Install via: pip install psycopg[binary]"
            )
        if self._conn is None or getattr(self._conn, "closed", False):
            if self._driver == "psycopg":
                import psycopg
                self._conn = psycopg.connect(self.connection_url, autocommit=True)
            else:
                import psycopg2
                self._conn = psycopg2.connect(self.connection_url)
                self._conn.autocommit = True
            self._ensure_schema()
        return self._conn

    def _ensure_schema(self):
        cur = self._conn.cursor()
        cur.execute(self.POSTGRES_SCHEMA_SQL)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def ping(self) -> bool:
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1;")
            row = cur.fetchone()
            return row[0] == 1
        except Exception as e:
            logger.warning("Postgres backend ping failed: %s", e)
            return False
