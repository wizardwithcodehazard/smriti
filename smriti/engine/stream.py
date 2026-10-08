"""
stream.py - Tier 1: Sub-millisecond Episodic Working Buffer (Hippocampal stream).
Append-only, transaction-safe SQLite WAL stream with zero LLM overhead.
Supports unconsolidated queries, batch consolidation marking, and retention sweeps.
"""

import json
import logging
import time
from typing import Any

from smriti.models import ActionStatus, Episode
from smriti.storage.db import DatabaseManager, get_db

logger = logging.getLogger(__name__)

class StreamBuffer:
    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db()

    def append(
        self,
        action: str,
        result: str = "",
        status: ActionStatus = ActionStatus.SUCCESS,
        context: str = "",
        latency_ms: float = 0.0,
        session_id: str = "default_session",
        metadata: dict[str, Any] | None = None
    ) -> Episode:
        """
        Appends an action trajectory into the episodic buffer in sub-millisecond time.
        Zero LLM latency, zero graph blocking.
        """
        episode = Episode(
            session_id=session_id,
            context=context,
            action=action,
            result=result,
            status=status,
            latency_ms=latency_ms,
            metadata=metadata or {},
            consolidated_at=None
        )

        conn = self.db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO episodes (
                    id, timestamp, session_id, context, action, result, status,
                    latency_ms, metadata_json, consolidated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.id,
                    episode.timestamp,
                    episode.session_id,
                    episode.context,
                    episode.action,
                    episode.result,
                    episode.status.value,
                    episode.latency_ms,
                    json.dumps(episode.metadata),
                    None
                )
            )

        return episode

    def get_recent(self, limit: int = 50, session_id: str | None = None) -> list[Episode]:
        """Retrieves recent episodes ordered chronologically descending."""
        query = "SELECT * FROM episodes"
        params = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = self.db.get_connection()
        rows = conn.execute(query, params).fetchall()

        return [self._row_to_episode(r) for r in rows]

    def get_unconsolidated(self, limit: int = 200, session_id: str | None = None) -> list[Episode]:
        """Retrieves raw episodes that have not yet been processed by a sleep consolidation cycle."""
        query = "SELECT * FROM episodes WHERE consolidated_at IS NULL"
        params = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        conn = self.db.get_connection()
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def mark_consolidated(self, episode_ids: list[str], consolidated_at: float) -> int:
        """Marks a batch of episodes as consolidated."""
        if not episode_ids:
            return 0
        conn = self.db.get_connection()
        with conn:
            # Batch update using chunks if needed
            placeholders = ",".join(["?"] * len(episode_ids))
            cur = conn.execute(
                f"UPDATE episodes SET consolidated_at = ? WHERE id IN ({placeholders})",
                [consolidated_at, *episode_ids]
            )
            return cur.rowcount

    def prune_older_than(self, retention_seconds: float, only_consolidated: bool = True) -> int:
        """Prunes historical episodes older than retention threshold to prevent disk bloat."""
        cutoff_time = time.time() - retention_seconds
        conn = self.db.get_connection()
        with conn:
            if only_consolidated:
                cur = conn.execute(
                    "DELETE FROM episodes WHERE timestamp < ? AND consolidated_at IS NOT NULL",
                    (cutoff_time,)
                )
            else:
                cur = conn.execute("DELETE FROM episodes WHERE timestamp < ?", (cutoff_time,))
            return cur.rowcount

    def count(self, session_id: str | None = None) -> int:
        """Returns total episode count."""
        query = "SELECT COUNT(*) FROM episodes"
        params = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        conn = self.db.get_connection()
        return conn.execute(query, params).fetchone()[0]

    @staticmethod
    def _row_to_episode(r) -> Episode:
        consolidated = r["consolidated_at"] if "consolidated_at" in r else None
        return Episode(
            id=r["id"],
            timestamp=r["timestamp"],
            session_id=r["session_id"],
            context=r["context"] or "",
            action=r["action"],
            result=r["result"] or "",
            status=ActionStatus(r["status"]),
            latency_ms=r["latency_ms"],
            metadata=json.loads(r["metadata_json"]) if r["metadata_json"] else {},
            consolidated_at=consolidated
        )
