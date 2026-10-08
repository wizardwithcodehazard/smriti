"""
llm.py - Extensible LLM Brain Interface for Smriti.
Provides optional neural LLM arbitration during preflight and knowledge distillation during sleep cycles.
Gracefully supports OpenAI, Anthropic, or local HTTP models with zero-dependency heuristic fallbacks.
"""

import json
import logging
import os
from typing import Any

from smriti.models import AntiMemory

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("SMRITI_LLM_MODEL") or "gpt-4o-mini"
        self.base_url = base_url or os.getenv("SMRITI_LLM_BASE_URL")

    def is_configured(self) -> bool:
        """Returns True if an LLM key or local endpoint is configured."""
        return bool(self.api_key or self.base_url)

    def evaluate_action_preflight(self, action: str, anti_memory: AntiMemory) -> tuple[bool, str | None]:
        """
        Asks LLM's brain if candidate action represents a catastrophic repetition of known dead end.
        Returns: (should_block: bool, reason: Optional[str])
        """
        if not self.is_configured():
            # If no LLM configured, trust the neural embedding/pattern match
            return (True, anti_memory.reason)

        prompt = (
            f"You are a preflight cognitive safety gate for an autonomous AI agent.\n"
            f"Candidate Action: '{action}'\n"
            f"Known Failure Signature: '{anti_memory.signature}'\n"
            f"Pattern: '{anti_memory.pattern}'\n"
            f"Prior Failure Reason: '{anti_memory.reason}'\n\n"
            f"Determine if this candidate action will trigger the known failure. "
            f"Reply strictly with JSON: {{\"block\": true/false, \"explanation\": \"short reason\"}}"
        )

        try:
            res_text = self._call_llm(prompt)
            data = json.loads(res_text)
            return (bool(data.get("block", True)), data.get("explanation"))
        except Exception as e:
            logger.warning("LLM Preflight arbiter call failed: %s; falling back to conservative block", e)
            return (True, anti_memory.reason)

    def summarize_failure_cluster(self, cluster_data: list[dict[str, Any]]) -> str:
        """Uses LLM to summarize recurring command failure trajectory and strip 90% token noise."""
        if not self.is_configured():
            first = cluster_data[0]
            return f"Repeated failures in '{first.get('action', '')[:40]}'. Error: {first.get('result', '')[:100]}"

        prompt = (
            f"Summarize these repeated tool/command failures into a crisp, single-sentence failure invariant:\n"
            f"{json.dumps(cluster_data, indent=2)}\n\n"
            f"Output only the distilled reason."
        )
        try:
            return self._call_llm(prompt).strip()
        except Exception as e:
            logger.warning("LLM Summarizer failed: %s", e)
            first = cluster_data[0]
            return f"Repeated failures in '{first.get('action', '')[:40]}'. Error: {first.get('result', '')[:100]}"

    def distill_resolution_heuristic(self, transitions: list[dict[str, Any]]) -> str:
        """Distills problem-solving breakthrough into a clean positive rule."""
        if not self.is_configured():
            t = transitions[0]
            return f"When '{t.get('failed_action')}' fails, use '{t.get('fix_action')}' instead."

        prompt = (
            f"An agent failed at an action, then successfully solved it with an alternative action.\n"
            f"Transitions: {json.dumps(transitions, indent=2)}\n\n"
            f"Synthesize this into a reusable heuristic rule (e.g., 'When X fails because of Y, do Z')."
        )
        try:
            return self._call_llm(prompt).strip()
        except Exception as e:
            logger.warning("LLM Resolution distillation failed: %s", e)
            t = transitions[0]
            return f"When '{t.get('failed_action')}' fails, use '{t.get('fix_action')}' instead."

    def _call_llm(self, prompt: str) -> str:
        """Internal dispatcher using httpx to avoid heavy external SDK lock-in."""
        import httpx
        headers = {"Content-Type": "application/json"}
        
        # Check if OpenAI compatible
        if os.getenv("OPENAI_API_KEY") or "openai" in self.model.lower() or not os.getenv("ANTHROPIC_API_KEY"):
            url = self.base_url or "https://api.openai.com/v1/chat/completions"
            headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        else:
            # Anthropic Claude format
            url = self.base_url or "https://api.anthropic.com/v1/messages"
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
            payload = {
                "model": self.model if "claude" in self.model else "claude-3-5-haiku-latest",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}]
            }
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["content"][0]["text"]
