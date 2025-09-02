"""Categorization prompts for field analysis in template generation.

This module provides specialized prompts for AI-powered field
categorization in step 3 of the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base_prompt import BasePromptManager


class CategorizationPromptManager(BasePromptManager):
    """Manages prompts for AI field categorization.
    
    Provides specialized prompts for categorizing mandatory fields
    into parent-level vs variant-level categories.
    """
    
    def get_system_prompt(self) -> str:
        """Get system prompt for categorization operations.
        
        Returns:
            System prompt for AI field categorization
        """
        return (
            "You are an expert e-commerce field categorization AI. "
            "Your task is to analyze mandatory fields and categorize them "
            "into parent-level (shared across variants) vs variant-level "
            "(unique per variant) based on e-commerce best practices."
        )
    
    def create_categorization_prompt(self, mandatory_fields: Dict[str, Dict[str, Any]]) -> str:
        """Create AI prompt for field categorization.
        
        Args:
            mandatory_fields: Dictionary of mandatory field data
            
        Returns:
            Formatted prompt for AI categorization
        """
        # Limit fields for prompt efficiency
        limited_fields = self.limit_data_size(
            mandatory_fields, 
            20,  # Allow more fields for categorization
            key_priority=['brand_name', 'feed_product_type', 'item_sku', 'color_name']
        )
        
        prompt = f"""{self.get_system_prompt()}

TASK: Analyze these mandatory fields and categorize them into parent-level vs variant-level.

MANDATORY FIELDS TO ANALYZE:
{self.format_json_data(limited_fields)}
{self._get_categorization_rules()}
{self._get_categorization_instructions()}
{self._get_categorization_output_format()}
Analyze and categorize now:"""
        
        return prompt
    
    def _get_categorization_rules(self) -> str:
        """Get categorization rules section.
        
        Returns:
            Formatted categorization rules
        """
        return """
CATEGORIZATION RULES:
- Parent-level: brand, material, category, gender, age group, country of origin, product type, department
- Variant-level: size, color, SKU, price, individual identifiers, specific measurements
- Parent fields define the product family and are shared across all variants
- Variant fields differentiate individual products within the family
- Consider German e-commerce context and Amazon marketplace standards"""
    
    def _get_categorization_instructions(self) -> str:
        """Get categorization instructions.
        
        Returns:
            Formatted instructions section
        """
        instructions = [
            "Analyze each field's role in product hierarchy",
            "Consider whether the field varies between product variants",
            "Parent fields should be shared across all variants of a product",
            "Variant fields should differentiate individual SKUs",
            "Use German language context for field name interpretation",
            "Provide confidence score based on certainty of categorization",
            "Include brief reasoning for categorization decisions"
        ]
        
        return self.create_instruction_section(instructions)
    
    def _get_categorization_output_format(self) -> str:
        """Get categorization output format.
        
        Returns:
            Formatted output format specification
        """
        format_example = {
            "parent_fields": ["brand_name", "feed_product_type", "manufacturer"],
            "variant_fields": ["item_sku", "color_name", "size_name"],
            "confidence": 0.95,
            "reasoning": "Brand and product type define the family, while SKU and color differentiate variants"
        }
        
        return self.create_output_format_section(
            format_example, 
            "REQUIRED JSON OUTPUT (no comments, no extra text)"
        )
    
    def create_validation_prompt(
        self,
        categorization_result: Dict[str, Any],
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> str:
        """Create prompt for validating categorization results.
        
        Args:
            categorization_result: Previous categorization result
            mandatory_fields: Original mandatory fields
            
        Returns:
            Validation prompt
        """
        prompt = f"""{self.get_system_prompt()}

TASK: Validate and potentially improve this field categorization.

ORIGINAL CATEGORIZATION:
{self.format_json_data(categorization_result)}

ORIGINAL FIELDS:
{self.format_json_data(self.limit_data_size(mandatory_fields, 15))}

VALIDATION CRITERIA:
- Are parent fields truly shared across all variants?
- Are variant fields unique to individual products?
- Is the categorization consistent with e-commerce best practices?
- Are there any obvious misclassifications?

Provide either the original categorization (if correct) or an improved version:
{self._get_categorization_output_format()}
Validate and respond:"""
        
        return prompt