"""Format enforcement and validation for AI mapping results.

This module ensures that AI mapping results conform to expected
formats and data structures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple


class FormatEnforcer:
    """Enforces format compliance for AI mapping results.
    
    Validates and transforms AI mapping results to ensure they
    conform to expected output formats and data structures.
    """
    
    def __init__(self, example_loader=None):
        """Initialize format enforcer.
        
        Args:
            example_loader: Optional example loader for format templates
        """
        self.example_loader = example_loader
        self.logger = logging.getLogger(__name__)
    
    def enforce_format(
        self,
        mapping_result: Dict[str, Any],
        parent_sku: str,
        strict: bool = False
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Enforce format compliance on mapping result.
        
        Args:
            mapping_result: Raw mapping result from AI
            parent_sku: Parent SKU identifier
            strict: Whether to enforce strict format compliance
            
        Returns:
            Tuple of (compliant_result, format_warnings)
        """
        warnings = []
        
        # Type safety: ensure mapping_result is a dictionary
        if not isinstance(mapping_result, dict):
            error_msg = f"Invalid mapping_result type: {type(mapping_result).__name__}, expected dict"
            self.logger.error(error_msg)
            warnings.append(error_msg)
            return self._create_minimal_compliant_result(parent_sku), warnings
        
        try:
            # Create compliant structure
            compliant_result = {
                "parent_sku": parent_sku,
                "transformation_timestamp": mapping_result.get(
                    "transformation_timestamp", 
                    "2024-01-01T00:00:00.000000"
                ),
                "parent_data": {},
                "variants": [],
                "metadata": {
                    "total_variants": 0,
                    "mapping_confidence": 0.0,
                    "processing_notes": "AI mapping completed",
                    "unmapped_mandatory_fields": []
                }
            }
            
            # Extract parent data
            parent_data = mapping_result.get("parent_data", {})
            if isinstance(parent_data, dict):
                compliant_result["parent_data"] = parent_data
            else:
                warnings.append("Invalid parent_data structure - using empty dict")
            
            # Extract variant data
            variant_data = mapping_result.get("variant_data", {})
            variants = []
            
            if isinstance(variant_data, dict):
                # Convert variant_data to variants list
                for variant_id, variant_data in variant_data.items():
                    if isinstance(variant_data, dict):
                        variant_entry = {
                            "variant_id": variant_id,
                            "data": variant_data
                        }
                        variants.append(variant_entry)
                    else:
                        warnings.append(f"Invalid variant data for {variant_id}")
            
            compliant_result["variants"] = variants
            
            # Extract metadata
            metadata = mapping_result.get("metadata", {})
            if isinstance(metadata, dict):
                compliant_result["metadata"].update({
                    "total_variants": len(variants),
                    "mapping_confidence": metadata.get("confidence", 0.0),
                    "processing_notes": metadata.get("processing_notes", "AI mapping completed"),
                    "unmapped_mandatory_fields": metadata.get("unmapped_mandatory_fields", metadata.get("unmapped_mandatory", []))
                })
            else:
                warnings.append("Invalid metadata structure - using defaults")
            
            # Validate required fields
            validation_warnings = self._validate_required_fields(compliant_result, strict)
            warnings.extend(validation_warnings)
            
            return compliant_result, warnings
            
        except Exception as e:
            error_msg = f"Format enforcement failed: {e}"
            self.logger.error(error_msg)
            warnings.append(error_msg)
            
            # Return minimal compliant structure
            return self._create_minimal_compliant_result(parent_sku), warnings
    
    def _validate_required_fields(
        self,
        compliant_result: Dict[str, Any],
        strict: bool
    ) -> List[str]:
        """Validate required fields in compliant result.
        
        Args:
            compliant_result: Compliant result structure
            strict: Whether to enforce strict validation
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check required top-level fields
        required_fields = ["parent_sku", "parent_data", "variants", "metadata"]
        for field in required_fields:
            if field not in compliant_result:
                if strict:
                    warnings.append(f"Missing required field: {field}")
                else:
                    # Add default value
                    compliant_result[field] = self._get_default_value(field)
                    warnings.append(f"Added default value for missing field: {field}")
        
        # Validate metadata structure
        metadata = compliant_result.get("metadata", {})
        required_metadata_fields = [
            "total_variants", "mapping_confidence", "processing_notes"
        ]
        
        for field in required_metadata_fields:
            if field not in metadata:
                if strict:
                    warnings.append(f"Missing required metadata field: {field}")
                else:
                    metadata[field] = self._get_default_metadata_value(field)
                    warnings.append(f"Added default metadata value: {field}")
        
        return warnings
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing field.
        
        Args:
            field: Field name
            
        Returns:
            Default value for the field
        """
        defaults = {
            "parent_sku": "unknown",
            "parent_data": {},
            "variants": [],
            "metadata": {
                "total_variants": 0,
                "mapping_confidence": 0.0,
                "processing_notes": "Default values used"
            }
        }
        return defaults.get(field, None)
    
    def _get_default_metadata_value(self, field: str) -> Any:
        """Get default metadata value.
        
        Args:
            field: Metadata field name
            
        Returns:
            Default metadata value
        """
        defaults = {
            "total_variants": 0,
            "mapping_confidence": 0.0,
            "processing_notes": "Default metadata values used",
            "unmapped_mandatory_fields": []  # Preserve validation results if available
        }
        return defaults.get(field, None)
    
    def _create_minimal_compliant_result(self, parent_sku: str) -> Dict[str, Any]:
        """Create minimal compliant result structure.
        
        Args:
            parent_sku: Parent SKU identifier
            
        Returns:
            Minimal compliant result structure
        """
        return {
            "parent_sku": parent_sku,
            "transformation_timestamp": "2024-01-01T00:00:00.000000",
            "parent_data": {},
            "variants": [],
            "metadata": {
                "total_variants": 0,
                "mapping_confidence": 0.0,
                "processing_notes": "Minimal compliant structure created due to format errors",
                "unmapped_mandatory_fields": []
            }
        }
    
    def validate_output_format(self, result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate output format compliance.
        
        Args:
            result: Result to validate
            
        Returns:
            Tuple of (is_valid, validation_issues)
        """
        issues = []
        
        # Check top-level structure
        required_keys = ["parent_sku", "parent_data", "variants", "metadata"]
        for key in required_keys:
            if key not in result:
                issues.append(f"Missing required key: {key}")
        
        # Check data types
        if "parent_data" in result and not isinstance(result["parent_data"], dict):
            issues.append("parent_data must be a dictionary")
        
        if "variants" in result and not isinstance(result["variants"], list):
            issues.append("variants must be a list")
        
        if "metadata" in result and not isinstance(result["metadata"], dict):
            issues.append("metadata must be a dictionary")
        
        # Check metadata structure
        metadata = result.get("metadata", {})
        required_metadata = ["total_variants", "mapping_confidence"]
        for key in required_metadata:
            if key not in metadata:
                issues.append(f"Missing required metadata key: {key}")
        
        return len(issues) == 0, issues