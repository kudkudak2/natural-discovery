from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import importlib.util


_pydantic_spec = importlib.util.find_spec("pydantic")
if _pydantic_spec is None:
    raise RuntimeError(
        "Missing dependency: pydantic. Install with `pip install pydantic` (or add it to your environment)."
    )

from pydantic import BaseModel, Field  # noqa: E402


class ARCPalette(BaseModel):
    """Color mapping used for visualization / consumers."""

    background: int = 0
    blue: int = 1
    red: int = 2
    green: int = 3
    yellow: int = 4


class ARCExamplePair(BaseModel):
    x: List[List[int]] = Field(..., description="Input grid")
    y: List[List[int]] = Field(..., description="Output grid")


class ARCTestCase(BaseModel):
    x: List[List[int]] = Field(..., description="Test input grid (the query)")
    y: List[List[int]] = Field(..., description="Ground-truth test output grid (answer)")


class ARCTask(BaseModel):
    task_id: str
    skill_id: int
    skill_name: str
    puzzle_variant: Optional[str] = Field(
        default=None,
        description="Optional per-task latent variant identifier (e.g., explosion kernel mode).",
    )
    puzzle_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional per-task latent parameters (JSON-serializable) for analysis/repro.",
    )
    grid_size: int
    rule_color: Optional[int] = Field(
        default=None,
        description="Hidden task-level rule color shared across demos/test when applicable; null otherwise.",
    )
    demos: List[ARCExamplePair]
    test: ARCTestCase


class ARCDataset(BaseModel):
    dataset_id: str
    created_at: str
    split: str = Field(..., description="e.g. train / ood")
    ood: bool
    skills: List[int]
    grid_size: int
    palette: ARCPalette = Field(default_factory=ARCPalette)
    tasks: List[ARCTask]
    extra: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


