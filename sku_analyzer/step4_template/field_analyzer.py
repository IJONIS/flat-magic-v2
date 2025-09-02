"""Field analysis and categorization logic.

This module handles the analysis of mandatory fields to determine
which fields should be categorized as parent-level vs variant-level.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from sku_analyzer.shared.gemini_client import GeminiClient, AIProcessingConfig


class FieldCategorizationError(Exception):
    """Raised when field categorization fails."""
    pass


class FieldAnalyzer:
    """Analyzes and categorizes mandatory fields for template generation.
    
    This class provides both AI-powered and deterministic approaches
    for categorizing fields into parent and variant levels.
    """
    
    def __init__(self, enable_ai_categorization: bool = True) -> None:
        """Initialize field analyzer.
        
        Args:
            enable_ai_categorization: Whether to use AI-powered categorization
        """
        self.enable_ai = enable_ai_categorization
        self.logger = logging.getLogger(__name__)
        
        # AI configuration for field categorization
        if self.enable_ai:
            self.ai_config = AIProcessingConfig(
                model_name="gemini-2.5-flash",
                temperature=0.1,
                max_tokens=4096,
                timeout_seconds=30,
                max_concurrent=1
            )
            self.ai_client = None  # Lazy initialization
        
        # Deterministic categorization rules
        self._parent_level_indicators = {
            'brand', 'category', 'manufacturer', 'product_type', 'family',
            'series', 'collection', 'line', 'group', 'classification'
        }
        
        self._variant_level_indicators = {
            'size', 'color', 'color_name', 'material', 'style', 'variant',
            'configuration', 'option', 'specification', 'dimension'
        }
    
    async def categorize_field_levels(
        self, 
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Categorize fields into parent and variant levels.
        
        Args:
            mandatory_fields: Dictionary of mandatory field data
            
        Returns:
            Tuple of (parent_fields, variant_fields) lists
        """
        if self.enable_ai:
            try:
                # Use AI-powered categorization
                result = await self._ai_categorize_fields(mandatory_fields)
                self._last_categorization_method = "ai"
                return result
            except Exception as e:
                self.logger.warning(
                    f"AI categorization failed: {e}. Falling back to deterministic approach."
                )
                # Fall through to deterministic approach
        
        # Deterministic categorization
        result = self._deterministic_categorize_fields(mandatory_fields)
        self._last_categorization_method = "deterministic"
        return result
    
    async def _ai_categorize_fields(
        self, 
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Use AI to categorize fields into parent and variant levels.
        
        Args:
            mandatory_fields: Dictionary of mandatory field data
            
        Returns:
            Tuple of (parent_fields, variant_fields) lists
            
        Raises:
            FieldCategorizationError: When AI categorization fails
        """
        self.logger.info("Starting AI-powered field categorization")
        
        try:
            # Initialize AI client if needed
            if self.ai_client is None:
                self.ai_client = GeminiClient(
                    config=self.ai_config
                )
            
            # Create AI prompt for field categorization
            prompt = self._create_categorization_prompt(mandatory_fields)
            
            # Execute AI categorization
            response = await self.ai_client.generate_mapping(
                prompt=prompt,
                operation_name="field_categorization"
            )
            
            # Parse and validate AI response
            categorization_result = await self.ai_client.validate_json_response(response)
            validated_result = self._validate_ai_categorization(
                categorization_result, mandatory_fields
            )
            
            parent_fields = validated_result['parent_fields']
            variant_fields = validated_result['variant_fields']
            
            # Apply critical field placement rules
            parent_fields, variant_fields = self._ensure_critical_field_placement(
                parent_fields, variant_fields, mandatory_fields
            )
            
            self.logger.info(
                f"AI categorization completed - "
                f"{len(parent_fields)} parent, {len(variant_fields)} variant fields"
            )
            
            # Store AI confidence for metadata
            self._last_ai_confidence = validated_result.get('confidence', 0.8)
            
            return parent_fields, variant_fields
            
        except Exception as e:
            self.logger.error(f"AI field categorization failed: {e}")
            raise FieldCategorizationError(f"AI categorization failed: {e}") from e
    
    def _deterministic_categorize_fields(
        self, 
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Deterministic field categorization using rule-based logic.
        
        Args:
            mandatory_fields: Dictionary of mandatory field data
            
        Returns:
            Tuple of (parent_fields, variant_fields) lists
        """
        parent_fields = []
        variant_fields = []
        
        for field_name, field_data in mandatory_fields.items():
            display_name = field_data.get('display_name', '').lower()
            technical_name = field_name.lower()
            
            # Check for parent-level indicators
            is_parent_field = any(
                indicator in technical_name or indicator in display_name
                for indicator in self._parent_level_indicators
            )
            
            # Check for variant-level indicators
            is_variant_field = any(
                indicator in technical_name or indicator in display_name
                for indicator in self._variant_level_indicators
            )
            
            # Analyze field characteristics
            field_characteristics = self._analyze_field_characteristics(field_data)
            
            # Decision logic based on multiple factors
            if is_parent_field and not is_variant_field:
                parent_fields.append(field_name)
            elif is_variant_field and not is_parent_field:
                variant_fields.append(field_name)
            else:
                # Use field characteristics for ambiguous cases
                if field_characteristics['is_likely_parent']:
                    parent_fields.append(field_name)
                else:
                    variant_fields.append(field_name)
        
        # Ensure critical fields are properly categorized
        parent_fields, variant_fields = self._ensure_critical_field_placement(
            parent_fields, variant_fields, mandatory_fields
        )
        
        self.logger.info(
            f"Deterministic categorization: {len(parent_fields)} parent, {len(variant_fields)} variant"
        )
        
        return parent_fields, variant_fields
    
    def _create_categorization_prompt(self, mandatory_fields: Dict[str, Dict[str, Any]]) -> str:
        """Create AI prompt for field categorization.
        
        Args:
            mandatory_fields: Dictionary of mandatory field data
            
        Returns:
            Formatted prompt for AI categorization
        """
        import json
        
        prompt = f"""You are an expert e-commerce field categorization AI. Analyze these mandatory fields and categorize them into parent-level (shared across variants) vs variant-level (unique per variant).

MANDATORY FIELDS TO ANALYZE:
{json.dumps(mandatory_fields, indent=2)}

CATEGORIZATION RULES:
- Parent-level: brand, material, category, gender, age group, country of origin, product type, department
- Variant-level: size, color, SKU, price, individual identifiers, specific measurements

Return ONLY valid JSON in this exact format (no comments, no extra text):
{{
  "parent_fields": ["field1", "field2"],
  "variant_fields": ["field3", "field4"],
  "confidence": 0.95,
  "reasoning": "Brief explanation of categorization logic"
}}

Analyze and categorize now:"""
        
        return prompt
    
    def _validate_ai_categorization(
        self, 
        ai_result: Dict[str, Any], 
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate AI categorization response.
        
        Args:
            ai_result: AI categorization response
            mandatory_fields: Original mandatory fields data
            
        Returns:
            Validated categorization result
            
        Raises:
            FieldCategorizationError: If validation fails
        """
        try:
            # Extract field lists from different possible formats
            if 'parent_level_fields' in ai_result and 'variant_level_fields' in ai_result:
                # New format - extract field names from nested objects
                parent_fields = list(ai_result['parent_level_fields'].keys())
                variant_fields = list(ai_result['variant_level_fields'].keys())
            elif 'parent_fields' in ai_result and 'variant_fields' in ai_result:
                # Standard format - simple lists
                parent_fields = ai_result['parent_fields']
                variant_fields = ai_result['variant_fields']
            else:
                raise FieldCategorizationError("Missing required keys in AI response")
            
            # Validate field lists
            if not isinstance(parent_fields, list) or not isinstance(variant_fields, list):
                raise FieldCategorizationError("Field categories must be lists")
            
            # Check all fields are categorized
            all_mandatory_fields = set(mandatory_fields.keys())
            categorized_fields = set(parent_fields + variant_fields)
            
            missing_fields = all_mandatory_fields - categorized_fields
            if missing_fields:
                self.logger.warning(f"AI missed fields: {missing_fields}. Adding to variant level.")
                variant_fields.extend(list(missing_fields))
            
            extra_fields = categorized_fields - all_mandatory_fields
            if extra_fields:
                self.logger.warning(f"AI added unknown fields: {extra_fields}. Removing.")
                parent_fields = [f for f in parent_fields if f in all_mandatory_fields]
                variant_fields = [f for f in variant_fields if f in all_mandatory_fields]
            
            # Check for field duplicates
            duplicates = set(parent_fields) & set(variant_fields)
            if duplicates:
                # Remove from variant, keep in parent
                self.logger.warning(f"Duplicate fields found: {duplicates}. Keeping in parent.")
                variant_fields = [f for f in variant_fields if f not in duplicates]
            
            # Extract confidence and reasoning
            confidence = ai_result.get('confidence', 0.8)
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                confidence = 0.8
            
            reasoning = ai_result.get('reasoning', 'AI-powered categorization')
            
            return {
                'parent_fields': parent_fields,
                'variant_fields': variant_fields,
                'reasoning': reasoning,
                'confidence': confidence
            }
            
        except Exception as e:
            raise FieldCategorizationError(f"AI response validation failed: {e}") from e
    
    def _analyze_field_characteristics(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze field characteristics to determine parent/variant placement.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Analysis results
        """
        valid_values = field_data.get('valid_values', [])
        constraints = field_data.get('constraints', {})
        data_type = field_data.get('data_type', 'string')
        
        # Calculate uniqueness ratio
        value_count = constraints.get('value_count', len(valid_values))
        max_length = constraints.get('max_length', 0)
        
        # Characteristics that suggest parent-level field
        is_likely_parent = (
            value_count <= 10 or  # Limited options suggest categorization
            data_type in ['boolean'] or  # Boolean fields often define product categories
            max_length > 100 or  # Long text fields often describe product families
            any(keyword in str(field_data.get('display_name', '')).lower() 
                for keyword in ['typ', 'kategorie', 'klasse', 'gruppe'])
        )
        
        return {
            'is_likely_parent': is_likely_parent,
            'value_count': value_count,
            'max_length': max_length,
            'data_type_category': data_type
        }
    
    def _ensure_critical_field_placement(
        self,
        parent_fields: List[str],
        variant_fields: List[str],
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Ensure critical fields are placed in appropriate levels.
        
        Args:
            parent_fields: Current parent field assignments
            variant_fields: Current variant field assignments
            mandatory_fields: All mandatory field data
            
        Returns:
            Adjusted field assignments
        """
        # Critical parent fields that should never be variants
        critical_parent_keywords = ['feed_product_type', 'brand_name', 'manufacturer']
        
        # Critical variant fields that should never be parent
        critical_variant_keywords = ['item_sku', 'color_name', 'size_name']
        
        # Move critical parent fields
        for field_name in list(variant_fields):
            if any(keyword in field_name.lower() for keyword in critical_parent_keywords):
                variant_fields.remove(field_name)
                if field_name not in parent_fields:
                    parent_fields.append(field_name)
                    self.logger.debug(f"Moved {field_name} to parent fields (critical)")
        
        # Move critical variant fields  
        for field_name in list(parent_fields):
            if any(keyword in field_name.lower() for keyword in critical_variant_keywords):
                parent_fields.remove(field_name)
                if field_name not in variant_fields:
                    variant_fields.append(field_name)
                    self.logger.debug(f"Moved {field_name} to variant fields (critical)")
        
        return parent_fields, variant_fields
    
    def get_categorization_method(self) -> str:
        """Get the method used for the last categorization."""
        return getattr(self, '_last_categorization_method', 'unknown')
    
    def get_ai_confidence(self) -> float:
        """Get AI confidence for the last categorization."""
        return getattr(self, '_last_ai_confidence', 0.0)