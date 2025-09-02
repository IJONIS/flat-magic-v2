"""SKU data models and structures."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Set


@dataclass(frozen=True)
class SkuPattern:
    """Immutable SKU pattern data structure."""
    base_pattern: str
    variant_suffix: str
    full_sku: str
    group_key: str
    
    def __post_init__(self) -> None:
        """Validate SKU pattern data."""
        if not all([self.base_pattern, self.full_sku, self.group_key]):
            raise ValueError("All SKU pattern fields must be non-empty")


@dataclass
class ParentChildRelationship:
    """Parent-child SKU relationship data structure."""
    parent_sku: str
    child_skus: Set[str] = field(default_factory=set)
    pattern_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child_sku: str) -> None:
        """Add child SKU to relationship."""
        if child_sku != self.parent_sku:
            self.child_skus.add(child_sku)
    
    def calculate_confidence(self) -> None:
        """Calculate pattern confidence based on child count and consistency."""
        child_count = len(self.child_skus)
        if child_count == 0:
            self.pattern_confidence = 0.0
        elif child_count == 1:
            self.pattern_confidence = 0.5
        elif child_count <= 5:
            self.pattern_confidence = 0.8
        else:
            self.pattern_confidence = 0.95


@dataclass
class ProcessingJob:
    """Job metadata and configuration."""
    job_id: str
    input_path: Path
    output_dir: Path
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "initialized"
    
    def __post_init__(self) -> None:
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)