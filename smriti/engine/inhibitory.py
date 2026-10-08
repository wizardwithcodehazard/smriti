"""
inhibitory.py - Tier 3B: Inhibitory Anti-Memories (Vimarsha).
Preflight gate that intercepts agent actions before execution if a matching failure signature is found.
Features:
1. Fast-path literal containment
2. Dense vector neural semantic matching (replaces brittle regex)
3. Optional LLM arbiter evaluation for nuanced semantic decisions
4. Project scoping and dynamic memory retraction (forget)
"""

import json
import logging
import re
from collections.abc import Callable
from datetime import datetime, timezone

from smriti.models import AntiMemory, InhibitionResult, ValenceType
from smriti.storage.db import DatabaseManager, get_db
from smriti.storage.embeddings import EmbeddingEngine, get_embedding_engine

logger = logging.getLogger(__name__)

REGEX_CHARS = set(r"^$.*+?{}[]\|()")

class InhibitoryGate:
    def __init__(
        self,
        db: DatabaseManager | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        llm_evaluator: Callable[[str, AntiMemory], tuple[bool, str | None]] | None = None,
        semantic_threshold: float = 0.82
    ):
        self.db = db or get_db()
        self.embeddings = embedding_engine or get_embedding_engine()
        self.llm_evaluator = llm_evaluator
        self.semantic_threshold = semantic_threshold
        self._compiled_patterns: dict[str, re.Pattern | None] = {}

    def record_anti_memory(
        self,
        signature: str,
        pattern: str,
        reason: str,
        suggested_fix: str | None = None,
        context_tags: list[str] | None = None,
        severity: str = "high",
        project_root: str | None = None
    ) -> AntiMemory:
        """Records or updates a known failure signature with dense vector embeddings and project scoping."""
        clean_sig = signature.strip()
        clean_pat = pattern.strip()
        clean_reason = reason.strip()

        # Compute neural embedding over signature, pattern, and reason
        embed_payload = f"{clean_sig} {clean_pat} {clean_reason}"
        vec = self.embeddings.embed_text(embed_payload)
        vec_json = json.dumps(vec)

        anti_memory = AntiMemory(
            signature=clean_sig,
            pattern=clean_pat,
            reason=clean_reason,
            suggested_fix=suggested_fix.strip() if suggested_fix else None,
            context_tags=context_tags or [],
            severity=severity,
            valence=ValenceType.INHIBITORY,
            project_root=project_root.strip() if project_root else None,
            is_active=True,
            embedding=vec
        )

        conn = self.db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO anti_memories (
                    id, signature, pattern, reason, suggested_fix, context_tags,
                    times_triggered, severity, created_at, last_seen,
                    valence, project_root, is_active, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    pattern = excluded.pattern,
                    reason = excluded.reason,
                    suggested_fix = excluded.suggested_fix,
                    context_tags = excluded.context_tags,
                    severity = excluded.severity,
                    last_seen = excluded.last_seen,
                    project_root = excluded.project_root,
                    is_active = 1,
                    embedding = excluded.embedding
                """,
                (
                    anti_memory.id,
                    anti_memory.signature,
                    anti_memory.pattern,
                    anti_memory.reason,
                    anti_memory.suggested_fix,
                    json.dumps(anti_memory.context_tags),
                    anti_memory.times_triggered,
                    anti_memory.severity,
                    anti_memory.created_at,
                    anti_memory.last_seen,
                    anti_memory.valence.value,
                    anti_memory.project_root,
                    1 if anti_memory.is_active else 0,
                    vec_json
                )
            )

        logger.info("Recorded anti-memory '%s' (project: %s)", clean_sig, project_root)
        return anti_memory

    def check_action(
        self,
        action: str,
        context: str = "",
        project_root: str | None = None
    ) -> InhibitionResult:
        """
        Active Preflight Gate: Evaluates candidate action against active anti-memories.
        1. Fast substring containment.
        2. Neural vector semantic similarity match.
        3. Optional LLM arbiter for nuanced semantic judgment.
        """
        conn = self.db.get_connection()
        normalized_action = action.strip()
        combined_text = f"{context} {normalized_action}".lower()

        # SQL level filter: Only active anti-memories for this project (or global)
        if project_root:
            query = """
                SELECT * FROM anti_memories 
                WHERE is_active = 1 AND (project_root IS NULL OR project_root = ?)
                ORDER BY times_triggered DESC
            """
            rows = conn.execute(query, (project_root.strip(),)).fetchall()
        else:
            query = """
                SELECT * FROM anti_memories 
                WHERE is_active = 1
                ORDER BY times_triggered DESC
            """
            rows = conn.execute(query).fetchall()

        if not rows:
            return InhibitionResult(passed=True)

        # Compute candidate action embedding for neural semantic matching
        action_vec = self.embeddings.embed_text(normalized_action)

        for r in rows:
            anti_mem = self._row_to_anti_memory(r)
            pattern = anti_mem.pattern
            is_match = False
            match_source = "exact"

            # Tier 1: Fast Substring Containment
            pat_lower = pattern.lower()
            if pat_lower in combined_text or pat_lower in normalized_action.lower():
                is_match = True
                match_source = "literal_substring"

            # Tier 2: Neural Dense Vector Semantic Matching (eliminates regex brittleness)
            if not is_match and anti_mem.embedding and action_vec:
                sim = EmbeddingEngine.cosine_similarity(action_vec, anti_mem.embedding)
                if sim >= self.semantic_threshold:
                    is_match = True
                    match_source = f"semantic_similarity ({sim:.2f})"

            # Fallback legacy regex check for explicit pattern expressions
            if not is_match and any(c in REGEX_CHARS for c in pattern):
                compiled = self._get_or_compile_regex(pattern)
                if compiled is not None:
                    try:
                        if compiled.search(normalized_action):
                            is_match = True
                            match_source = "regex"
                    except Exception:
                        pass

            # Tier 3: Optional LLM Arbiter
            if is_match and self.llm_evaluator is not None:
                # Ask LLM's brain to confirm or veto the block
                should_block, llm_reason = self.llm_evaluator(normalized_action, anti_mem)
                if not should_block:
                    logger.info("LLM Arbiter overrode block for '%s' against action '%s'", anti_mem.signature, normalized_action)
                    continue
                if llm_reason:
                    anti_mem.reason = llm_reason

            if is_match:
                now = datetime.now(timezone.utc).timestamp()
                with conn:
                    conn.execute(
                        """
                        UPDATE anti_memories
                        SET times_triggered = times_triggered + 1,
                            last_seen = ?
                        WHERE id = ?
                        """,
                        (now, anti_mem.id)
                    )

                logger.warning(
                    "Action '%s' blocked by anti-memory '%s' via %s",
                    normalized_action, anti_mem.signature, match_source
                )
                return InhibitionResult(
                    passed=False,
                    matched_signature=anti_mem.signature,
                    reason=anti_mem.reason,
                    suggested_fix=anti_mem.suggested_fix,
                    severity=anti_mem.severity
                )

        return InhibitionResult(passed=True)

    def forget_anti_memory(self, signature_or_id: str) -> bool:
        """Deactivates an anti-memory so it no longer intercepts actions."""
        conn = self.db.get_connection()
        key = signature_or_id.strip()
        with conn:
            cur = conn.execute(
                """
                UPDATE anti_memories
                SET is_active = 0
                WHERE signature = ? OR id = ?
                """,
                (key, key)
            )
            matched = cur.rowcount > 0

        self._compiled_patterns.pop(key, None)
        return matched

    def list_all(self, active_only: bool = True, project_root: str | None = None) -> list[AntiMemory]:
        """Lists recorded anti-memories, optionally filtered by active status and project."""
        conn = self.db.get_connection()
        sql = "SELECT * FROM anti_memories WHERE 1=1"
        params = []

        if active_only:
            sql += " AND is_active = 1"
        if project_root:
            sql += " AND (project_root IS NULL OR project_root = ?)"
            params.append(project_root.strip())

        sql += " ORDER BY times_triggered DESC, last_seen DESC"
        rows = conn.execute(sql, params).fetchall()

        return [self._row_to_anti_memory(r) for r in rows]

    def _get_or_compile_regex(self, pattern: str) -> re.Pattern | None:
        """Safely retrieves or compiles a regex with length bounds and ReDoS defense."""
        if pattern in self._compiled_patterns:
            return self._compiled_patterns[pattern]

        if len(pattern) > 300:
            self._compiled_patterns[pattern] = None
            return None

        if not any(c in REGEX_CHARS for c in pattern):
            self._compiled_patterns[pattern] = None
            return None

        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._compiled_patterns[pattern] = compiled
            return compiled
        except re.error:
            self._compiled_patterns[pattern] = None
            return None

    @staticmethod
    def _row_to_anti_memory(r) -> AntiMemory:
        context_tags = json.loads(r["context_tags"]) if r["context_tags"] else []
        project_root = r["project_root"] if "project_root" in r else None
        is_active = bool(r["is_active"]) if "is_active" in r else True
        valence = ValenceType(r["valence"]) if ("valence" in r and r["valence"]) else ValenceType.INHIBITORY
        embedding = json.loads(r["embedding"]) if ("embedding" in r and r["embedding"]) else None

        return AntiMemory(
            id=r["id"],
            signature=r["signature"],
            pattern=r["pattern"],
            reason=r["reason"],
            suggested_fix=r["suggested_fix"],
            context_tags=context_tags,
            times_triggered=r["times_triggered"],
            severity=r["severity"],
            created_at=r["created_at"],
            last_seen=r["last_seen"],
            valence=valence,
            project_root=project_root,
            is_active=is_active,
            embedding=embedding
        )
