"""Pydantic models for AI mapping operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MappingInput(BaseModel):
    """Input data for AI mapping operation."""
    
    parent_sku: str = Field(description="Parent SKU identifier")
    mandatory_fields: Dict[str, Any] = Field(description="Required fields schema")
    product_data: Dict[str, Any] = Field(description="Compressed product data")
    business_context: Optional[str] = Field(default=None, description="Additional context")
    template_structure: Optional[Dict[str, Any]] = Field(default=None, description="Template structure for guidance")


class Metadata(BaseModel):
    """Metadata for AI transformation output."""
    
    parent_id: str = Field(description="Parent identifier")
    job_id: str = Field(description="Job identifier")
    transformation_timestamp: str = Field(description="ISO timestamp of transformation")
    ai_model: str = Field(description="AI model used for transformation")
    mapping_confidence: float = Field(ge=0.0, le=1.0, description="Overall mapping confidence")
    total_variants: int = Field(ge=0, description="Total number of variants")


class ParentData(BaseModel):
    """Parent-level product data fields."""
    
    feed_product_type: str = Field(description="Product type classification")
    brand_name: str = Field(description="Brand name")
    outer_material_type: str = Field(description="Outer material type")
    target_gender: str = Field(description="Target gender")
    age_range_description: str = Field(description="Age range description")
    bottoms_size_system: str = Field(description="Size system for bottoms")
    bottoms_size_class: str = Field(description="Size class for bottoms")
    country_of_origin: str = Field(description="Country of origin")
    department_name: str = Field(description="Department name")
    recommended_browse_nodes: str = Field(description="Recommended browse nodes")


class VariantRecord(BaseModel):
    """Individual variant record within variance data."""
    
    item_sku: str = Field(description="Item SKU identifier")
    size_name: str = Field(description="Size name")
    color_name: str = Field(description="Color name")
    size_map: str = Field(description="Mapped size value")
    color_map: str = Field(description="Mapped color value")


class AITransformationOutput(BaseModel):
    """Complete AI transformation output matching example_output_ai.json structure."""
    
    metadata: Metadata = Field(description="Transformation metadata")
    parent_data: ParentData = Field(description="Parent-level product data")
    variance_data: List[VariantRecord] = Field(description="List of variant records")


class TransformationResult(BaseModel):
    """Complete AI transformation result (legacy compatibility)."""
    
    parent_sku: str = Field(description="Parent SKU identifier")
    parent_data: Dict[str, Any] = Field(description="Transformed parent-level data")
    variance_data: List[Dict[str, str]] = Field(description="List of variant records")
    metadata: Dict[str, Any] = Field(description="Transformation metadata")


# Legacy models for backward compatibility
class FieldMapping(BaseModel):
    """Individual field mapping result (DEPRECATED)."""
    
    source_field: str = Field(description="Source field name")
    target_field: str = Field(description="Target mandatory field")
    mapped_value: Any = Field(description="Mapped value")
    confidence: float = Field(ge=0.0, le=1.0, description="Mapping confidence")
    reasoning: str = Field(description="AI reasoning for mapping")


class MappingResult(BaseModel):
    """Complete AI mapping result (DEPRECATED - use TransformationResult)."""
    
    parent_sku: str = Field(description="Parent SKU identifier")
    mapped_fields: List[FieldMapping] = Field(description="Field mappings")
    unmapped_mandatory: List[str] = Field(description="Unmapped mandatory fields")
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence")
    processing_notes: str = Field(description="Processing summary")


class AIProcessingConfig(BaseModel):
    """Configuration for AI processing operations."""
    
    model_name: str = Field(default="gemini-2.5-flash", description="AI model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Maximum output tokens")
    timeout_seconds: int = Field(default=30, gt=0, description="Request timeout")
    batch_size: int = Field(default=10, gt=0, description="Batch processing size")
    max_concurrent: int = Field(default=5, gt=0, description="Max concurrent requests")