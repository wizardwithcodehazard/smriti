"""
framework.py - High-level Python Developer SDK for the Smriti Cognitive Memory Engine.
Provides an ergonomic facade and function guard decorator for autonomous agent frameworks
(LangChain, CrewAI, AutoGen, or custom agent loops).
"""

import functools
import time
from typing import Any, Callable, Optional

from smriti.config import SmritiConfig, get_config
from smriti.engine.consolidator import Consolidator
from smriti.engine.cortex import Cortex
from smriti.engine.inhibitory import InhibitoryGate
from smriti.engine.stream import StreamBuffer
from smriti.models import (
    ActionStatus,
    AntiMemory,
    CorticalRule,
    Episode,
    InhibitionResult,
)
from smriti.storage.db import DatabaseManager, get_db


class SmritiInhibitionError(Exception):
    """Raised when an action is blocked by the Smriti inhibitory gate."""

    def __init__(self, result: InhibitionResult):
        self.result = result
        super().__init__(
            f"Action blocked by Smriti Inhibitory Gate: {result.reason} "
            f"(Fix: {result.suggested_fix or 'None provided'})"
        )


class Smriti:
    """
    Unified client for Smriti Cognitive Memory.
    Combines working memory stream, inhibitory preflight gate, cortical rule mesh, and sleep consolidation.
    """

    def __init__(
        self,
        config: Optional[SmritiConfig] = None,
        db: Optional[DatabaseManager] = None
    ):
        self.config = config or get_config()
        self.db = db or get_db(self.config)
        self.stream = StreamBuffer(self.db)
        self.inhibitory = InhibitoryGate(self.db)
        self.cortex = Cortex(self.db, self.config)
        self.consolidator = Consolidator(
            self.db,
            self.config,
            stream=self.stream,
            inhibitory=self.inhibitory,
            cortex=self.cortex
        )

    def preflight(
        self,
        action: str,
        context: str = "",
        project_root: Optional[str] = None
    ) -> InhibitionResult:
        """
        Active Preflight Check: Run before executing any command or modifying code.
        Returns InhibitionResult with .passed, .reason, .suggested_fix, etc.
        """
        return self.inhibitory.check_action(
            action,
            context=context,
            project_root=project_root
        )

    def record(
        self,
        action: str,
        outcome: str = "",
        status: ActionStatus | str = ActionStatus.SUCCESS,
        latency_ms: float = 0.0,
        context: str = ""
    ) -> Episode:
        """
        Appends an action-outcome trace to the Tier 1 working memory buffer (<1ms latency).
        """
        if isinstance(status, str):
            s = status.strip().lower()
            if s == "success":
                act_status = ActionStatus.SUCCESS
            elif s == "failure":
                act_status = ActionStatus.FAILURE
            elif s == "error":
                act_status = ActionStatus.ERROR
            elif s == "intercepted":
                act_status = ActionStatus.INTERCEPTED
            else:
                act_status = ActionStatus.FAILURE
        else:
            act_status = status

        return self.stream.append(
            action=action,
            result=outcome,
            status=act_status,
            context=context,
            latency_ms=latency_ms
        )

    def recall(
        self,
        query: str = "",
        limit: int = 5,
        reinforce: bool = True
    ) -> list[tuple[CorticalRule, float]]:
        """
        Recalls top-ranked cortical rules using dense vector similarity,
        associative spreading activation, and Ebbinghaus decay.
        """
        return self.cortex.recall_rules(query=query, limit=limit, reinforce=reinforce)

    def add_rule(
        self,
        rule_text: str,
        category: str = "general",
        confidence: float = 1.0,
        base_strength: float = 1.0
    ) -> CorticalRule:
        """
        Stores a verified positive heuristic or architectural invariant in the cortical rule mesh.
        """
        return self.cortex.add_rule(
            rule_text=rule_text,
            category=category,
            confidence=confidence,
            base_strength=base_strength
        )

    def sleep(
        self,
        session_id: Optional[str] = None,
        project_root: Optional[str] = None
    ) -> dict[str, int]:
        """
        Runs the Tier 2 memory consolidation cycle.
        """
        return self.consolidator.sleep(session_id=session_id, project_root=project_root)

    def record_anti_memory(
        self,
        signature: str,
        pattern: str,
        reason: str,
        suggested_fix: Optional[str] = None,
        project_root: Optional[str] = None
    ) -> AntiMemory:
        """
        Explicitly records a dead end or failure pattern in the inhibitory gate.
        """
        return self.inhibitory.record_anti_memory(
            signature=signature,
            pattern=pattern,
            reason=reason,
            suggested_fix=suggested_fix,
            project_root=project_root
        )

    def forget(self, target: str, is_rule: bool = False) -> bool:
        """
        Deactivates an anti-memory (by signature or ID) or deletes a cortical rule (by ID).
        """
        if is_rule:
            return self.cortex.forget_rule(target)
        return self.inhibitory.forget_anti_memory(target)

    def guard(
        self,
        project_root: Optional[str] = None,
        raise_on_blocked: bool = False
    ):
        """
        Decorator for agent tools and execution functions.
        Automatically intercepts blocked actions before execution, records the outcome,
        and either returns an inhibition dictionary or raises SmritiInhibitionError.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Infer action string
                action_text = ""
                if args and isinstance(args[0], str):
                    action_text = args[0]
                elif "action" in kwargs:
                    action_text = str(kwargs["action"])
                elif "command" in kwargs:
                    action_text = str(kwargs["command"])
                elif "cmd" in kwargs:
                    action_text = str(kwargs["cmd"])
                else:
                    action_text = f"{func.__name__}({args}, {kwargs})"

                preflight = self.preflight(action_text, project_root=project_root)
                if not preflight.passed:
                    self.record(
                        action=action_text,
                        outcome=f"Blocked: {preflight.reason}",
                        status=ActionStatus.INTERCEPTED
                    )
                    if raise_on_blocked:
                        raise SmritiInhibitionError(preflight)
                    return {
                        "status": "blocked",
                        "reason": preflight.reason,
                        "suggested_fix": preflight.suggested_fix,
                        "signature": preflight.matched_signature
                    }

                t0 = time.perf_counter()
                try:
                    res = func(*args, **kwargs)
                    elapsed = (time.perf_counter() - t0) * 1000
                    self.record(
                        action=action_text,
                        outcome=str(res)[:500],
                        status=ActionStatus.SUCCESS,
                        latency_ms=elapsed
                    )
                    return res
                except Exception as exc:
                    elapsed = (time.perf_counter() - t0) * 1000
                    self.record(
                        action=action_text,
                        outcome=str(exc)[:500],
                        status=ActionStatus.ERROR,
                        latency_ms=elapsed
                    )
                    raise
            return wrapper
        return decorator
