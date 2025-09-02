"""Field processing utilities for template generation.

This module provides utilities for field validation, relationship analysis,
and template structure processing.
"""

from __future__ import annotations

from typing import Any, Dict, List


class FieldProcessor:
    """Utility class for field processing and analysis.
    
    Handles field validation rules creation, inheritance analysis,
    and relationship definitions for template generation.
    """
    
    @staticmethod
    def determines_child_inheritance(field_data: Dict[str, Any]) -> bool:
        """Determine if parent field value should be inherited by children.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Whether field determines child inheritance
        """
        # Fields with limited values typically define inheritance
        value_count = field_data.get('constraints', {}).get('value_count', 0)
        return value_count <= 5 and value_count > 0

    @staticmethod
    def create_validation_rules(field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation rules for field.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Validation rules
        """
        constraints = field_data.get('constraints', {})
        data_type = field_data.get('data_type', 'string')
        
        rules = {
            'required': FieldProcessor.is_required_field(field_data),
            'data_type': data_type
        }
        
        if constraints.get('max_length'):
            rules['max_length'] = constraints['max_length']
        
        if field_data.get('valid_values'):
            rules['allowed_values'] = field_data['valid_values']
        
        return rules

    @staticmethod
    def is_required_field(field_data: Dict[str, Any]) -> bool:
        """Determine if field is required.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Whether field is required
        """
        # Fields with specific valid values are typically required
        valid_values = field_data.get('valid_values', [])
        return len(valid_values) > 0 and len(valid_values) < 20

    @staticmethod
    def determine_variation_type(field_data: Dict[str, Any]) -> str:
        """Determine type of variation for variant field.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Variation type string
        """
        field_name = field_data.get('display_name', '').lower()
        
        if 'color' in field_name or 'farbe' in field_name:
            return 'color'
        elif 'size' in field_name or 'größe' in field_name:
            return 'size' 
        elif 'material' in field_name:
            return 'material'
        else:
            return 'attribute'

    @staticmethod
    def is_variable_field(field_data: Dict[str, Any]) -> bool:
        """Check if field varies between product variants.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Whether field is variable
        """
        # Fields with many possible values typically vary
        value_count = field_data.get('constraints', {}).get('value_count', 0)
        return value_count > 5 or value_count == 0

    @staticmethod
    def can_inherit_from_parent(field_data: Dict[str, Any]) -> bool:
        """Check if field can inherit value from parent.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Whether field can inherit from parent
        """
        # Most fields can potentially inherit, except unique identifiers
        field_name = field_data.get('display_name', '').lower()
        unique_indicators = ['sku', 'id', 'identifier', 'number']
        
        return not any(indicator in field_name for indicator in unique_indicators)

    @staticmethod
    def create_field_relationships(
        parent_fields: List[str],
        variant_fields: List[str],
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create field relationship definitions.
        
        Args:
            parent_fields: Parent field names
            variant_fields: Variant field names
            mandatory_fields: All mandatory field data
            
        Returns:
            Field relationships structure
        """
        parent_defines = []
        variant_overrides = []
        shared_constraints = {}
        
        # Parent fields that define product family
        for field_name in parent_fields:
            field_data = mandatory_fields[field_name]
            if FieldProcessor.determines_child_inheritance(field_data):
                parent_defines.append({
                    'field': field_name,
                    'inheritance_type': 'mandatory',
                    'override_allowed': False
                })
        
        # Variant fields that can override parent values
        for field_name in variant_fields:
            field_data = mandatory_fields[field_name]
            if FieldProcessor.can_inherit_from_parent(field_data):
                variant_overrides.append({
                    'field': field_name,
                    'default_source': 'parent',
                    'variation_required': FieldProcessor.is_variable_field(field_data)
                })
        
        # Shared constraints across field types
        for field_name, field_data in mandatory_fields.items():
            constraints = field_data.get('constraints', {})
            if constraints.get('max_length') and constraints['max_length'] > 50:
                shared_constraints[field_name] = {
                    'max_length': constraints['max_length'],
                    'applies_to': 'all_levels'
                }
        
        return {
            'parent_defines': parent_defines,
            'variant_overrides': variant_overrides,
            'shared_constraints': shared_constraints
        }
