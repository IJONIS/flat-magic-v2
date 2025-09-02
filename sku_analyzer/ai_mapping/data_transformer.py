"""Data transformation utilities for AI mapping system."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class DataVarianceAnalyzer:
    """Analyzes compressed product data to identify parent vs variant fields."""
    
    def analyze_product_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze product data to separate parent and variance fields.
        
        Args:
            product_data: Compressed product data with parent_data and data_rows
            
        Returns:
            Analysis results with parent fields and variance field mappings
        """
        if "data_rows" not in product_data or not product_data["data_rows"]:
            logger.warning("No data_rows found in product data")
            return {
                "parent_fields": product_data.get("parent_data", {}),
                "variance_fields": {},
                "field_analysis": {}
            }
        
        data_rows = product_data["data_rows"]
        parent_data = product_data.get("parent_data", {})
        
        # Analyze variance in data_rows
        field_variance = self._analyze_field_variance(data_rows)
        
        # Identify meaningful variance fields
        variance_fields = self._identify_variance_fields(field_variance)
        
        # Generate field mappings for common variant fields
        field_mappings = self._generate_field_mappings(variance_fields)
        
        return {
            "parent_fields": parent_data,
            "variance_fields": variance_fields,
            "field_mappings": field_mappings,
            "field_analysis": field_variance
        }
    
    def _analyze_field_variance(self, data_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze variance across data rows for each field.
        
        Args:
            data_rows: List of product variant data
            
        Returns:
            Variance analysis for each field
        """
        field_values = defaultdict(set)
        field_counts = defaultdict(int)
        
        # Collect all values for each field
        for row in data_rows:
            for field, value in row.items():
                if value is not None:
                    field_values[field].add(str(value))
                    field_counts[field] += 1
        
        # Analyze variance
        analysis = {}
        total_rows = len(data_rows)
        
        for field, values in field_values.items():
            unique_count = len(values)
            coverage = field_counts[field] / total_rows
            
            analysis[field] = {
                "unique_values": sorted(list(values)),
                "unique_count": unique_count,
                "total_occurrences": field_counts[field],
                "coverage": coverage,
                "is_variant": unique_count > 1 and coverage > 0.5,
                "is_size_like": self._is_size_like_field(field, values),
                "is_color_like": self._is_color_like_field(field, values)
            }
        
        return analysis
    
    def _identify_variance_fields(self, field_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Identify fields that represent meaningful product variance.
        
        Args:
            field_analysis: Field variance analysis results
            
        Returns:
            Dictionary mapping field types to field names
        """
        variance_fields = {}
        
        # Group by common patterns
        size_fields = []
        color_fields = []
        other_variant_fields = []
        
        for field, analysis in field_analysis.items():
            if not analysis["is_variant"]:
                continue
                
            if analysis["is_size_like"]:
                size_fields.append(field)
            elif analysis["is_color_like"]:
                color_fields.append(field)
            elif analysis["unique_count"] <= 20:  # Reasonable number of variants
                other_variant_fields.append(field)
        
        if size_fields:
            variance_fields["size_fields"] = size_fields
        if color_fields:
            variance_fields["color_fields"] = color_fields
        if other_variant_fields:
            variance_fields["other_fields"] = other_variant_fields
            
        return variance_fields
    
    def _generate_field_mappings(self, variance_fields: Dict[str, List[str]]) -> Dict[str, str]:
        """Generate mappings from source fields to Amazon field names.
        
        Args:
            variance_fields: Identified variance fields by type
            
        Returns:
            Field mapping dictionary
        """
        mappings = {}
        
        # Map size fields
        for field in variance_fields.get("size_fields", []):
            if "size" in field.lower() or "größe" in field.lower() or "FVALUE_3_2" in field:
                mappings[field] = "size_name"
                break
        
        # Map color fields  
        for field in variance_fields.get("color_fields", []):
            if "color" in field.lower() or "farbe" in field.lower() or "FVALUE_3_3" in field:
                mappings[field] = "color_name" 
                break
        
        return mappings
    
    def _is_size_like_field(self, field: str, values: Set[str]) -> bool:
        """Check if field represents size information.
        
        Args:
            field: Field name
            values: Set of field values
            
        Returns:
            True if field appears to be size-related
        """
        field_lower = field.lower()
        size_keywords = ["size", "größe", "fvalue_3_2"]
        
        # Check field name
        if any(keyword in field_lower for keyword in size_keywords):
            return True
        
        # Check if values look like sizes (numbers, size codes)
        if len(values) <= 30:  # Reasonable number of sizes
            numeric_values = 0
            for value in values:
                if value.strip().isdigit() or self._is_size_code(value):
                    numeric_values += 1
            
            # If most values are numeric or size codes
            return numeric_values / len(values) > 0.7
        
        return False
    
    def _is_color_like_field(self, field: str, values: Set[str]) -> bool:
        """Check if field represents color information.
        
        Args:
            field: Field name
            values: Set of field values
            
        Returns:
            True if field appears to be color-related
        """
        field_lower = field.lower()
        color_keywords = ["color", "colour", "farbe", "fvalue_3_3"]
        
        # Check field name
        if any(keyword in field_lower for keyword in color_keywords):
            return True
        
        # Check if values look like colors
        if len(values) <= 20:  # Reasonable number of colors
            color_values = 0
            for value in values:
                if self._is_color_name(value):
                    color_values += 1
                    
            return color_values / len(values) > 0.5
        
        return False
    
    def _is_size_code(self, value: str) -> bool:
        """Check if value looks like a size code.
        
        Args:
            value: Value to check
            
        Returns:
            True if value appears to be a size code
        """
        value_clean = value.strip().upper()
        
        # Common size patterns
        size_patterns = [
            lambda v: v in ["XS", "S", "M", "L", "XL", "XXL", "XXXL"],
            lambda v: v.isdigit() and 10 <= int(v) <= 200,  # Numeric sizes
            lambda v: len(v) <= 5 and (v.isalnum() or "/" in v)  # Size codes
        ]
        
        return any(pattern(value_clean) for pattern in size_patterns)
    
    def _is_color_name(self, value: str) -> bool:
        """Check if value looks like a color name.
        
        Args:
            value: Value to check
            
        Returns:
            True if value appears to be a color name
        """
        value_lower = value.lower().strip()
        
        # Common color names (German and English)
        color_names = {
            "schwarz", "black", "weiß", "white", "rot", "red", 
            "blau", "blue", "grün", "green", "gelb", "yellow",
            "braun", "brown", "grau", "gray", "grey", "rosa", "pink",
            "lila", "purple", "orange", "oliv", "olive", "beige",
            "navy", "khaki", "bordeaux", "türkis", "turquoise"
        }
        
        return any(color in value_lower for color in color_names)


class ConstraintValidator:
    """Validates transformation results against mandatory field constraints."""
    
    def validate_transformation(
        self, 
        transformed_data: Dict[str, Any],
        mandatory_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate transformed data against mandatory field constraints.
        
        Args:
            transformed_data: AI transformation result
            mandatory_fields: Constraint definitions
            
        Returns:
            Validation results with compliance scores
        """
        parent_data = transformed_data.get("parent_data", {})
        variance_data = transformed_data.get("variance_data", {})
        
        validation_results = {
            "compliant_fields": [],
            "constraint_violations": [],
            "missing_mandatory": [],
            "compliance_score": 0.0
        }
        
        total_fields = len(mandatory_fields)
        compliant_count = 0
        
        for field_name, field_config in mandatory_fields.items():
            if field_name in parent_data:
                # Validate parent data field
                violations = self._validate_field_value(
                    parent_data[field_name], field_config
                )
                if violations:
                    validation_results["constraint_violations"].extend([
                        {"field": field_name, "location": "parent_data", "violations": violations}
                    ])
                else:
                    validation_results["compliant_fields"].append(field_name)
                    compliant_count += 1
                    
            elif field_name in variance_data:
                # Validate variance data field
                for value in variance_data[field_name]:
                    violations = self._validate_field_value(value, field_config)
                    if violations:
                        validation_results["constraint_violations"].extend([
                            {"field": field_name, "location": "variance_data", "violations": violations}
                        ])
                        break
                else:
                    validation_results["compliant_fields"].append(field_name)
                    compliant_count += 1
            else:
                validation_results["missing_mandatory"].append(field_name)
        
        validation_results["compliance_score"] = compliant_count / total_fields if total_fields > 0 else 0.0
        
        return validation_results
    
    def _validate_field_value(self, value: Any, field_config: Dict[str, Any]) -> List[str]:
        """Validate single field value against constraints.
        
        Args:
            value: Field value to validate
            field_config: Field configuration with constraints
            
        Returns:
            List of constraint violations
        """
        violations = []
        constraints = field_config.get("constraints", {})
        
        # Check max length
        if "max_length" in constraints and constraints["max_length"]:
            if len(str(value)) > constraints["max_length"]:
                violations.append(f"Exceeds max length of {constraints['max_length']}")
        
        # Check valid values
        valid_values = field_config.get("unique_values", [])
        if valid_values and len(valid_values) > 1:
            # Filter out description text (usually longer)
            actual_values = [v for v in valid_values if len(v) < 100]
            if actual_values and str(value) not in actual_values:
                violations.append(f"Not in valid values: {actual_values[:3]}")
        
        return violations