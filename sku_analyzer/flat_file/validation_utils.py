"""Validation utilities for template analysis.

This module provides focused functionality for validating analysis results
and ensuring data integrity.
"""

from __future__ import annotations

import logging
from typing import List, Set

from .data_structures import ColumnMapping, TemplateAnalysisResult


class ValidationUtils:
    """Utilities for validating analysis results."""
    
    MIN_REQUIRED_MAPPINGS = 10
    
    def __init__(self) -> None:
        """Initialize validation utilities."""
        self.logger = logging.getLogger(__name__)
    
    def validate_analysis_result(self, result: TemplateAnalysisResult) -> None:
        """Validate analysis result with fast checks.
        
        Args:
            result: Analysis result to validate
        """
        # Quick mapping count check
        if result.total_mappings < self.MIN_REQUIRED_MAPPINGS:
            error = (
                f"Insufficient mappings: {result.total_mappings} "
                f"(required: {self.MIN_REQUIRED_MAPPINGS})"
            )
            result.validation_errors.append(error)
        
        # Efficient duplicate detection
        duplicates = self._find_duplicate_technical_names(result.column_mappings)
        if duplicates:
            error = f"Duplicate technical names: {sorted(duplicates)}"
            result.validation_errors.append(error)
        
        # Fast empty value checks
        empty_stats = self._count_empty_values(result.column_mappings)
        if empty_stats['empty_technical'] > 0:
            result.validation_errors.append(
                f"Empty technical names: {empty_stats['empty_technical']} mappings"
            )
        if empty_stats['empty_display'] > 0:
            result.validation_errors.append(
                f"Empty display names: {empty_stats['empty_display']} mappings"
            )
        
        # Log validation summary
        if result.validation_errors:
            for error in result.validation_errors:
                self.logger.warning(f"Validation error: {error}")
        else:
            self.logger.info("✅ Analysis validation passed")
    
    def _find_duplicate_technical_names(self, mappings: List[ColumnMapping]) -> Set[str]:
        """Find duplicate technical names using efficient set operations.
        
        Args:
            mappings: List of column mappings to check
            
        Returns:
            Set of duplicate technical names
        """
        tech_names = [mapping.technical_name for mapping in mappings]
        seen = set()
        duplicates = set()
        
        for name in tech_names:
            if name in seen:
                duplicates.add(name)
            else:
                seen.add(name)
        
        return duplicates
    
    def _count_empty_values(self, mappings: List[ColumnMapping]) -> dict[str, int]:
        """Count empty technical and display names efficiently.
        
        Args:
            mappings: List of column mappings to check
            
        Returns:
            Dictionary with empty value counts
        """
        empty_technical = sum(
            1 for mapping in mappings 
            if not mapping.technical_name.strip()
        )
        empty_display = sum(
            1 for mapping in mappings 
            if not mapping.display_name.strip()
        )
        
        return {
            'empty_technical': empty_technical,
            'empty_display': empty_display
        }
    
    def validate_mandatory_fields(
        self, 
        field_validations: dict[str, any]
    ) -> List[str]:
        """Validate mandatory fields have values.
        
        Args:
            field_validations: Dictionary of field validations
            
        Returns:
            List of validation errors for mandatory fields
        """
        errors = []
        
        for field_name, validation in field_validations.items():
            if (
                hasattr(validation, 'requirement_status') and 
                validation.requirement_status == "mandatory" and 
                not validation.validation_passed
            ):
                errors.append(f"Mandatory field '{field_name}' has no valid values")
        
        return errors
