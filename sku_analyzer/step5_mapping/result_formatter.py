"""Result formatting and summary generation utilities.

This module provides comprehensive result formatting, statistics generation,
and summary creation for AI mapping operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .models import (
    ProcessingResult, BatchProcessingResult, TransformationResult,
    MappingInput
)


class ResultFormatter:
    """Handles result formatting and summary generation.
    
    Provides utilities for processing statistics, batch summaries,
    and file I/O operations.
    """
    
    def __init__(self):
        """Initialize result formatter."""
        self.logger = logging.getLogger(__name__)
        self.processing_stats = {
            "total_processed": 0,
            "successful_mappings": 0,
            "failed_mappings": 0,
            "fallback_used": 0,
            "average_confidence": 0.0,
            "total_processing_time": 0.0
        }
    
    async def load_json_async(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file asynchronously.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded JSON data
        """
        def load_json():
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        
        return await asyncio.to_thread(load_json)
    
    async def save_compliant_result(
        self,
        compliant_result: Dict[str, Any],
        output_file: Path
    ) -> None:
        """Save compliant result to file.
        
        Args:
            compliant_result: Compliant result dictionary to save
            output_file: Output file path
        """
        def save_json():
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open('w', encoding='utf-8') as f:
                json.dump(compliant_result, f, indent=2, ensure_ascii=False)
        
        await asyncio.to_thread(save_json)
    
    def extract_template_fields(self, template_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Extract field definitions from template structure for AI mapping.
        
        Args:
            template_structure: Template structure from step 4
            
        Returns:
            Field definitions compatible with MappingInput format
        """
        extracted_fields = {}
        
        # Type safety check
        if not isinstance(template_structure, dict):
            self.logger.warning(
                f"Expected dict for template_structure, got {type(template_structure).__name__}: {template_structure}"
            )
            return extracted_fields
        
        # Extract parent fields with type safety
        parent_product = template_structure.get('parent_product', {})
        if not isinstance(parent_product, dict):
            parent_product = {}
        
        parent_fields = parent_product.get('fields', {})
        if not isinstance(parent_fields, dict):
            parent_fields = {}
        
        for field_name, field_info in parent_fields.items():
            if isinstance(field_info, dict):
                extracted_fields[field_name] = {
                    'display_name': field_info.get('display_name', field_name),
                    'data_type': field_info.get('data_type', 'string'),
                    'constraints': field_info.get('constraints', {}),
                    'level': 'parent',
                    'validation_rules': field_info.get('validation_rules', {})
                }
        
        # Extract variant fields with type safety
        child_variants = template_structure.get('child_variants', {})
        if not isinstance(child_variants, dict):
            child_variants = {}
        
        variant_fields = child_variants.get('fields', {})
        if not isinstance(variant_fields, dict):
            variant_fields = {}
        
        for field_name, field_info in variant_fields.items():
            if isinstance(field_info, dict):
                extracted_fields[field_name] = {
                    'display_name': field_info.get('display_name', field_name),
                    'data_type': field_info.get('data_type', 'string'),
                    'constraints': field_info.get('constraints', {}),
                    'level': 'variant',
                    'variation_type': field_info.get('variation_type', 'attribute'),
                    'validation_rules': field_info.get('validation_rules', {})
                }
        
        return extracted_fields
    
    def update_processing_stats(
        self,
        transformation_result: TransformationResult,
        processing_time: float
    ) -> None:
        """Update processing statistics.
        
        Args:
            transformation_result: Completed transformation result
            processing_time: Processing duration in seconds
        """
        self.processing_stats["total_processed"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        # Update confidence average
        current_avg = self.processing_stats["average_confidence"]
        total_processed = self.processing_stats["total_processed"]
        # Type safety for metadata access
        metadata = transformation_result.metadata
        if not isinstance(metadata, dict):
            self.logger.warning(
                f"Expected dict for metadata, got {type(metadata).__name__}: {metadata}"
            )
            metadata = {}
        
        confidence = metadata.get("confidence", 0.0)
        
        self.processing_stats["average_confidence"] = (
            (current_avg * (total_processed - 1) + confidence) 
            / total_processed
        )
    
    def generate_processing_summary(
        self,
        results: List[ProcessingResult],
        ai_client_stats: Dict[str, Any]
    ) -> BatchProcessingResult:
        """Generate comprehensive processing summary.
        
        Args:
            results: List of processing results
            ai_client_stats: AI client performance statistics
            
        Returns:
            Batch processing summary
        """
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        total_confidence = sum(r.confidence for r in successful_results)
        avg_confidence = (
            total_confidence / len(successful_results) 
            if successful_results else 0.0
        )
        
        return BatchProcessingResult(
            summary={
                "total_parents": len(results),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "success_rate": len(successful_results) / len(results) if results else 0.0,
                "average_confidence": avg_confidence
            },
            performance={
                "total_processing_time_ms": sum(r.processing_time_ms for r in results),
                "average_processing_time_ms": (
                    sum(r.processing_time_ms for r in results) / len(results)
                    if results else 0.0
                ),
                "ai_client_stats": ai_client_stats,
                "processor_stats": self.processing_stats
            },
            results=results
        )
    
    def get_performance_stats(self, ai_client_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Get current performance statistics.
        
        Args:
            ai_client_stats: AI client performance statistics
            
        Returns:
            Combined performance statistics
        """
        return {
            **self.processing_stats,
            "ai_client_performance": ai_client_stats
        }
