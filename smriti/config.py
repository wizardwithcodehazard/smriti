"""
config.py - Dynamic environment and runtime configuration for Smriti.
Discovers local .smriti/ directory or falls back to user home directory.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SmritiConfig:
    # Storage settings
    db_filename: str = "smriti.db"
    project_dir: Path = Path.cwd()
    smriti_dir_name: str = ".smriti"
    
    # Biological decay parameters
    # decay_rate: controls how fast unused rules decay over time (in hours^-1)
    default_decay_rate: float = 0.05
    # prune_threshold: rules with effective strength below this are archived
    prune_threshold: float = 0.15
    # reinforcement_boost: strength added per successful recall
    reinforcement_boost: float = 0.15
    
    # Semantic & Associative Graph parameters
    dedup_similarity_threshold: float = 0.85
    associative_edge_threshold: float = 0.60
    spreading_activation_factor: float = 0.25
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Consolidation thresholds
    consolidation_turn_interval: int = 15
    raw_retention_days: int = 7

    @property
    def smriti_dir(self) -> Path:
        """Finds .smriti in current working directory or ancestors, else creates in cwd."""
        current = self.project_dir.resolve()
        for parent in [current, *current.parents]:
            candidate = parent / self.smriti_dir_name
            if candidate.is_dir():
                return candidate
        return current / self.smriti_dir_name

    @property
    def db_path(self) -> Path:
        return self.smriti_dir / self.db_filename

def get_config() -> SmritiConfig:
    return SmritiConfig()
