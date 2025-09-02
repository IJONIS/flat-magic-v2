"""Core data structures for flat file template analysis.

This module provides dataclasses and type definitions for the template analyzer
and extractor components, following modern Python patterns with full type hints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


@dataclass(frozen=True)
class ColumnMapping:
    """Immutable column mapping data structure.
    
    Represents a mapping between technical and display names with
    requirement status and position information.
    """
    technical_name: str
    display_name: str
    row_index: int
    technical_col: str
    display_col: str
    requirement_status: Optional[str] = None
    requirement_col: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'technical_name': self.technical_name,
            'display_name': self.display_name,
            'row_index': self.row_index,
            'technical_column': self.technical_col,
            'display_column': self.display_col
        }
        
        if self.requirement_status is not None:
            result['requirement_status'] = self.requirement_status
        if self.requirement_col is not None:
            result['requirement_column'] = self.requirement_col
            
        return result


@dataclass
class PerformanceMetrics:
    """Performance measurement data for analysis operations."""
    operation_name: str
    duration_ms: float
    peak_memory_mb: float
    memory_delta_mb: float
    rows_processed: int
    early_termination: bool = False
    
    def __post_init__(self) -> None:
        """Calculate derived performance metrics."""
        self.throughput_rows_per_second = (
            (self.rows_processed / (self.duration_ms / 1000)) 
            if self.duration_ms > 0 else 0
        )
        self.memory_efficiency_kb_per_row = (
            (self.peak_memory_mb * 1024 / self.rows_processed) 
            if self.rows_processed > 0 else 0
        )


@dataclass
class TemplateAnalysisResult:
    """Complete template analysis result with performance metrics."""
    job_id: int
    worksheet_name: str
    header_row: int
    technical_column: str
    display_column: str
    column_mappings: List[ColumnMapping] = field(default_factory=list)
    total_mappings: int = 0
    validation_errors: List[str] = field(default_factory=list)
    performance_metrics: Optional[Dict[str, PerformanceMetrics]] = field(default_factory=dict)
    requirement_column: Optional[str] = None
    requirement_statistics: Optional[Dict[str, int]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        analysis_metadata = {
            'worksheet_name': self.worksheet_name,
            'header_row': self.header_row,
            'technical_column': self.technical_column,
            'display_column': self.display_column,
            'total_mappings': self.total_mappings
        }
        
        if self.requirement_column is not None:
            analysis_metadata['requirement_column'] = self.requirement_column
        if self.requirement_statistics:
            analysis_metadata['requirement_statistics'] = self.requirement_statistics
        
        base_dict = {
            'job_id': self.job_id,
            'analysis_metadata': analysis_metadata,
            'column_mappings': [mapping.to_dict() for mapping in self.column_mappings],
            'validation': {
                'errors': self.validation_errors,
                'valid': len(self.validation_errors) == 0
            }
        }
        
        if self.performance_metrics:
            base_dict['performance'] = {}
            for name, metrics in self.performance_metrics.items():
                if hasattr(metrics, 'duration_ms'):
                    base_dict['performance'][name] = {
                        'duration_ms': metrics.duration_ms,
                        'peak_memory_mb': metrics.peak_memory_mb,
                        'throughput_rows_per_second': metrics.throughput_rows_per_second,
                        'memory_efficiency_kb_per_row': metrics.memory_efficiency_kb_per_row,
                        'early_termination': metrics.early_termination
                    }
                else:
                    base_dict['performance'][name] = metrics
        
        return base_dict


@dataclass
class ExtractionMetrics:
    """Performance metrics for data extraction operations."""
    operation_name: str
    duration_ms: float
    peak_memory_mb: float
    values_extracted: int
    columns_processed: int
    
    def __post_init__(self) -> None:
        """Calculate derived performance metrics."""
        self.throughput_columns_per_second = (
            (self.columns_processed / (self.duration_ms / 1000)) 
            if self.duration_ms > 0 else 0
        )
        self.values_per_second = (
            (self.values_extracted / (self.duration_ms / 1000)) 
            if self.duration_ms > 0 else 0
        )


@dataclass
class FieldValidation:
    """Field validation results based on requirement status."""
    field_name: str
    requirement_status: str
    valid_values: List[str] = field(default_factory=list)
    invalid_values: List[str] = field(default_factory=list)
    empty_values: int = 0
    total_values: int = 0
    unicode_values: int = 0
    formula_values: int = 0
    merged_cell_values: int = 0
    
    @property
    def validation_passed(self) -> bool:
        """Check if validation passes based on requirement status."""
        if self.requirement_status == "mandatory" and not self.valid_values:
            return False
        return True
    
    @property
    def coverage_percentage(self) -> float:
        """Calculate percentage of valid values."""
        if self.total_values == 0:
            return 0.0
        return (len(self.valid_values) / self.total_values) * 100.0


@dataclass
class ExtractionResult:
    """Complete extraction results for value extraction."""
    job_id: Union[str, int]
    source_template: str
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    field_validations: Dict[str, FieldValidation] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    performance_metrics: List[ExtractionMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "source_template": self.source_template,
            "extraction_metadata": self.extraction_metadata,
            "field_validations": {
                field_name: {
                    "field_name": validation.field_name,
                    "valid_values": validation.valid_values,
                    "invalid_values": validation.invalid_values,
                }
                for field_name, validation in self.field_validations.items()
            },
            "validation_errors": self.validation_errors
        }


class TemplateDetectionError(Exception):
    """Raised when template detection fails."""
    pass
