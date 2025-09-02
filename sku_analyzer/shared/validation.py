"""Shared validation utilities.

This module provides validation functionality used across
the entire SKU analysis pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List


class ValidationUtils:
    """Shared validation utilities for pipeline operations.
    
    Provides common validation functionality that can be used
    across different modules in the pipeline.
    """
    
    def __init__(self):
        """Initialize validation utilities."""
        self.logger = logging.getLogger(__name__)
    
    def validate_required_fields(
        self,
        data: Dict[str, Any],
        required_fields: List[str]
    ) -> List[str]:
        """Validate that required fields are present and not empty.
        
        Args:
            data: Data dictionary to validate
            required_fields: List of required field names
            
        Returns:
            List of missing or empty field names
        """
        missing_fields = []
        
        for field in required_fields:
            if field not in data:
                missing_fields.append(f"Missing field: {field}")
            elif data[field] is None:
                missing_fields.append(f"Null value for field: {field}")
            elif isinstance(data[field], str) and not data[field].strip():
                missing_fields.append(f"Empty value for field: {field}")
        
        return missing_fields
    
    def validate_data_types(
        self,
        data: Dict[str, Any],
        type_requirements: Dict[str, type]
    ) -> List[str]:
        """Validate data types match requirements.
        
        Args:
            data: Data dictionary to validate
            type_requirements: Dictionary mapping field names to expected types
            
        Returns:
            List of type validation errors
        """
        type_errors = []
        
        for field, expected_type in type_requirements.items():
            if field in data and data[field] is not None:
                if not isinstance(data[field], expected_type):
                    actual_type = type(data[field]).__name__
                    expected_type_name = expected_type.__name__
                    type_errors.append(
                        f"Field {field}: expected {expected_type_name}, got {actual_type}"
                    )
        
        return type_errors
    
    def validate_json_structure(
        self,
        data: Dict[str, Any],
        required_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate JSON structure against requirements.
        
        Args:
            data: Data to validate
            required_structure: Required structure specification
            
        Returns:
            Validation result with status and issues
        """
        issues = []
        warnings = []
        
        # Check top-level keys
        if 'required_keys' in required_structure:
            missing_keys = self.validate_required_fields(
                data, required_structure['required_keys']
            )
            issues.extend(missing_keys)
        
        # Check data types
        if 'type_requirements' in required_structure:
            type_errors = self.validate_data_types(
                data, required_structure['type_requirements']
            )
            issues.extend(type_errors)
        
        # Check nested structures
        if 'nested_validations' in required_structure:
            for key, nested_requirements in required_structure['nested_validations'].items():
                if key in data and isinstance(data[key], dict):
                    nested_result = self.validate_json_structure(
                        data[key], nested_requirements
                    )
                    if nested_result['issues']:
                        issues.extend([f"{key}.{issue}" for issue in nested_result['issues']])
                    if nested_result['warnings']:
                        warnings.extend([f"{key}.{warning}" for warning in nested_result['warnings']])
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def validate_confidence_scores(
        self,
        data: Dict[str, Any],
        min_confidence: float = 0.0,
        max_confidence: float = 1.0
    ) -> List[str]:
        """Validate confidence scores are within acceptable range.
        
        Args:
            data: Data containing confidence scores
            min_confidence: Minimum acceptable confidence
            max_confidence: Maximum acceptable confidence
            
        Returns:
            List of confidence validation errors
        """
        errors = []
        
        def check_confidence(value: Any, field_path: str) -> None:
            if isinstance(value, (int, float)):
                if not (min_confidence <= value <= max_confidence):
                    errors.append(
                        f"Confidence score {field_path} out of range: {value} "
                        f"(expected: {min_confidence}-{max_confidence})"
                    )
        
        # Recursively check for confidence fields
        self._find_confidence_fields(data, check_confidence)
        
        return errors
    
    def _find_confidence_fields(
        self,
        data: Any,
        callback,
        path: str = ""
    ) -> None:
        """Recursively find confidence fields in data structure.
        
        Args:
            data: Data to search
            callback: Callback function to call for each confidence field
            path: Current path in the data structure
        """
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if 'confidence' in key.lower():
                    callback(value, current_path)
                else:
                    self._find_confidence_fields(value, callback, current_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                self._find_confidence_fields(item, callback, current_path)
    
    def validate_field_mappings(
        self,
        mappings: List[Dict[str, Any]],
        required_mapping_fields: List[str] = None
    ) -> Dict[str, Any]:
        """Validate field mapping structures.
        
        Args:
            mappings: List of mapping dictionaries
            required_mapping_fields: Required fields in each mapping
            
        Returns:
            Validation result
        """
        if required_mapping_fields is None:
            required_mapping_fields = ['source_field', 'target_field', 'confidence']
        
        issues = []
        warnings = []
        
        if not isinstance(mappings, list):
            issues.append("Mappings must be a list")
            return {'valid': False, 'issues': issues, 'warnings': warnings}
        
        for i, mapping in enumerate(mappings):
            if not isinstance(mapping, dict):
                issues.append(f"Mapping {i} must be a dictionary")
                continue
            
            # Check required fields
            missing_fields = self.validate_required_fields(mapping, required_mapping_fields)
            if missing_fields:
                issues.extend([f"Mapping {i}: {error}" for error in missing_fields])
            
            # Check confidence scores
            confidence_errors = self.validate_confidence_scores(mapping)
            if confidence_errors:
                issues.extend([f"Mapping {i}: {error}" for error in confidence_errors])
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_mappings': len(mappings)
        }
    
    def create_validation_summary(
        self,
        validation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create summary of multiple validation results.
        
        Args:
            validation_results: List of validation result dictionaries
            
        Returns:
            Validation summary
        """
        total_validations = len(validation_results)
        valid_count = sum(1 for result in validation_results if result.get('valid', False))
        
        all_issues = []
        all_warnings = []
        
        for result in validation_results:
            all_issues.extend(result.get('issues', []))
            all_warnings.extend(result.get('warnings', []))
        
        return {
            'total_validations': total_validations,
            'valid_count': valid_count,
            'invalid_count': total_validations - valid_count,
            'success_rate': valid_count / total_validations if total_validations > 0 else 0.0,
            'total_issues': len(all_issues),
            'total_warnings': len(all_warnings),
            'issues': all_issues,
            'warnings': all_warnings
        }