"""Template validation logic for quality assessment.

This module provides comprehensive template validation including
structure validation, field distribution analysis, and quality scoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional


class TemplateValidator:
    """Validates template structure and quality.
    
    Provides comprehensive validation of generated templates including
    structural integrity, field distribution, and quality metrics.
    """
    
    def __init__(self):
        """Initialize template validator."""
        self.logger = logging.getLogger(__name__)
    
    def validate_template(self, template_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template structure and calculate quality metrics.
        
        Args:
            template_structure: Template structure to validate
            
        Returns:
            Validation result with quality metrics
        """
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'quality_score': 0.0,
            'field_distribution': {}
        }
        
        try:
            # Validate basic structure
            structure_issues = self._validate_basic_structure(template_structure)
            validation_result['issues'].extend(structure_issues)
            
            # Validate field distribution
            distribution = self._analyze_field_distribution(template_structure)
            validation_result['field_distribution'] = distribution
            
            # Check field balance
            balance_warnings = self._check_field_balance(distribution)
            validation_result['warnings'].extend(balance_warnings)
            
            # Calculate quality score
            validation_result['quality_score'] = self._calculate_quality_score(
                template_structure, distribution, structure_issues
            )
            
            # Determine overall validity
            validation_result['valid'] = (
                len(structure_issues) == 0 and 
                validation_result['quality_score'] >= 0.6
            )
            
        except Exception as e:
            self.logger.error(f"Template validation failed: {e}")
            validation_result['valid'] = False
            validation_result['issues'].append(f"Validation error: {e}")
        
        return validation_result
    
    def _validate_basic_structure(self, template: Dict[str, Any]) -> List[str]:
        """Validate basic template structure.
        
        Args:
            template: Template structure
            
        Returns:
            List of structural issues
        """
        issues = []
        
        # Check required top-level keys
        required_keys = ['parent_product', 'child_variants', 'field_relationships']
        for key in required_keys:
            if key not in template:
                issues.append(f"Missing required section: {key}")
        
        # Validate parent product section
        if 'parent_product' in template:
            parent = template['parent_product']
            if 'fields' not in parent:
                issues.append("Missing 'fields' in parent_product")
            elif not isinstance(parent['fields'], dict):
                issues.append("Parent fields must be a dictionary")
        
        # Validate child variants section
        if 'child_variants' in template:
            variants = template['child_variants']
            if 'fields' not in variants:
                issues.append("Missing 'fields' in child_variants")
            elif not isinstance(variants['fields'], dict):
                issues.append("Variant fields must be a dictionary")
        
        return issues
    
    def _analyze_field_distribution(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze field distribution between parent and variants.
        
        Args:
            template: Template structure
            
        Returns:
            Field distribution metrics
        """
        parent_fields = template.get('parent_product', {}).get('fields', {})
        variant_fields = template.get('child_variants', {}).get('fields', {})
        
        total_fields = len(parent_fields) + len(variant_fields)
        
        return {
            'parent_field_count': len(parent_fields),
            'variant_field_count': len(variant_fields),
            'total_field_count': total_fields,
            'parent_ratio': len(parent_fields) / total_fields if total_fields > 0 else 0.0,
            'variant_ratio': len(variant_fields) / total_fields if total_fields > 0 else 0.0
        }
    
    def _check_field_balance(self, distribution: Dict[str, Any]) -> List[str]:
        """Check field balance and generate warnings.
        
        Args:
            distribution: Field distribution metrics
            
        Returns:
            List of balance warnings
        """
        warnings = []
        
        parent_ratio = distribution['parent_ratio']
        variant_ratio = distribution['variant_ratio']
        total_fields = distribution['total_field_count']
        
        # Check for extreme imbalances
        if parent_ratio > 0.8:
            warnings.append("Heavy parent field bias - consider moving fields to variants")
        elif variant_ratio > 0.9:
            warnings.append("Heavy variant field bias - consider parent-level fields")
        
        # Check minimum field counts
        if distribution['parent_field_count'] < 2:
            warnings.append("Very few parent fields - may indicate poor categorization")
        
        if distribution['variant_field_count'] < 1:
            warnings.append("No variant fields - may indicate missing variation data")
        
        # Check total field count
        if total_fields < 5:
            warnings.append("Low total field count - template may be incomplete")
        elif total_fields > 50:
            warnings.append("High field count - consider field grouping")
        
        return warnings
    
    def _calculate_quality_score(
        self, 
        template: Dict[str, Any], 
        distribution: Dict[str, Any], 
        issues: List[str]
    ) -> float:
        """Calculate overall template quality score.
        
        Args:
            template: Template structure
            distribution: Field distribution metrics
            issues: Structural issues
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        score = 1.0
        
        # Penalize structural issues
        score -= len(issues) * 0.2
        
        # Penalize extreme field imbalances
        parent_ratio = distribution['parent_ratio']
        if parent_ratio < 0.1 or parent_ratio > 0.8:
            score -= 0.15
        
        # Reward field relationship complexity
        relationships = template.get('field_relationships', {})
        if relationships.get('parent_defines'):
            score += 0.1
        if relationships.get('variant_overrides'):
            score += 0.1
        
        # Consider field metadata richness
        parent_fields = template.get('parent_product', {}).get('fields', {})
        variant_fields = template.get('child_variants', {}).get('fields', {})
        
        metadata_richness = 0
        for field_data in list(parent_fields.values()) + list(variant_fields.values()):
            if field_data.get('validation_rules'):
                metadata_richness += 1
        
        total_fields = distribution['total_field_count']
        if total_fields > 0:
            metadata_ratio = metadata_richness / total_fields
            score += metadata_ratio * 0.1
        
        # Ensure score bounds
        return max(0.0, min(1.0, score))
