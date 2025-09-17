"""Data models for AI mapping operations.

This module defines the data structures used throughout
the AI mapping workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MappingInput(BaseModel):
    """Input data for AI mapping operations."""
    
    parent_sku: str = Field(..., description="Parent SKU identifier")
    mandatory_fields: Dict[str, Any] = Field(..., description="Mandatory fields schema")
    product_data: Dict[str, Any] = Field(..., description="Source product data")
    template_structure: Optional[Dict[str, Any]] = Field(default=None, description="Template structure from step4_template.json")
    business_context: str = Field(default="German Amazon marketplace product", description="Business context")


class MappingResult(BaseModel):
    """Individual field mapping result."""
    
    source_field: str = Field(..., description="Source field name")
    target_field: str = Field(..., description="Target mandatory field name")
    mapped_value: Any = Field(..., description="Mapped value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Mapping confidence score")
    reasoning: str = Field(..., description="Reasoning for the mapping")


class TransformationResult(BaseModel):
    """Result of AI mapping transformation with structured variant data."""
    
    parent_sku: str = Field(description="Parent SKU identifier")
    parent_data: Dict[str, Any] = Field(default_factory=dict, description="Parent product data")
    variant_data: Dict[str, Any] = Field(default_factory=dict, description="Variant-level data")
    metadata: Union[Dict[str, Any], str] = Field(default_factory=dict, description="Processing metadata")


class ProcessingConfig(BaseModel):
    """Configuration for mapping processing."""
    
    max_retries: int = Field(default=2, description="Maximum retry attempts")
    timeout_seconds: int = Field(default=30, description="Request timeout")
    enable_fallback: bool = Field(default=True, description="Enable fallback to alternative methods")
    batch_size: int = Field(default=1, description="Batch processing size")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")


class ProcessingResult(BaseModel):
    """Result from processing a parent directory."""
    
    parent_sku: str = Field(..., description="Parent SKU identifier")
    success: bool = Field(..., description="Whether processing was successful")
    mapped_fields_count: int = Field(default=0, description="Number of successfully mapped fields")
    unmapped_count: int = Field(default=0, description="Number of unmapped mandatory fields")
    confidence: float = Field(default=0.0, description="Overall mapping confidence")
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    output_file: Optional[str] = Field(None, description="Output file path")
    error: Optional[str] = Field(None, description="Error message if failed")
    format_warnings: int = Field(default=0, description="Number of format warnings")
    format_compliant: bool = Field(default=True, description="Whether output is format compliant")


class BatchProcessingResult(BaseModel):
    """Result from batch processing multiple parents."""
    
    summary: Dict[str, Any] = Field(default_factory=dict, description="Processing summary statistics")
    performance: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    results: List[ProcessingResult] = Field(default_factory=list, description="Individual processing results")
    
    class Config:
        arbitrary_types_allowed = True