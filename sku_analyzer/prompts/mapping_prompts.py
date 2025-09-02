"""Mapping prompts for AI product data mapping operations.

This module provides specialized prompts for the AI mapping
workflow in step 4 of the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_prompt import BasePromptManager


class MappingPromptManager(BasePromptManager):
    """Manages prompts for AI product data mapping.
    
    Provides specialized prompts for mapping German Amazon product data
    to mandatory fields using template guidance.
    """
    
    def get_system_prompt(self) -> str:
        """Get system prompt for mapping operations.
        
        Returns:
            System prompt for AI mapping
        """
        return (
            "You are an expert German Amazon product data mapper. "
            "Your task is to map product data to mandatory Amazon fields "
            "using template guidance and semantic understanding."
        )
    
    def render_mapping_prompt(self, context: Dict[str, Any]) -> str:
        """Render complete mapping prompt with context.
        
        Args:
            context: Mapping context including parent_sku, mandatory_fields, etc.
            
        Returns:
            Complete mapping prompt
        """
        # Validate required context
        required_fields = ['parent_sku', 'mandatory_fields', 'product_data']
        missing = self.validate_prompt_data(required_fields, context)
        if missing:
            raise ValueError(f"Missing required context fields: {missing}")
        
        # Limit data sizes for prompt efficiency
        limited_fields = self.limit_data_size(
            context['mandatory_fields'], 
            self.default_data_limits['mandatory_fields']
        )
        
        product_data = context['product_data'].get('parent_data', {})
        limited_product = self.limit_data_size(
            product_data, 
            self.default_data_limits['product_data']
        ) if product_data else {}
        
        # Create template guidance section
        template_guidance = self._create_template_guidance(
            context.get('template_structure')
        )
        
        # Build complete prompt
        prompt = f"""{self.get_system_prompt()}

TASK: Map product data for parent SKU {context['parent_sku']} to mandatory Amazon fields.
{template_guidance}
MANDATORY FIELDS (showing first {len(limited_fields)}):
{self.format_json_data(limited_fields)}

PRODUCT DATA (showing first {len(limited_product)}):
{self.format_json_data(limited_product)}
{self._get_mapping_instructions()}
{self._get_output_format(context['parent_sku'])}
Map the product data now:"""
        
        return prompt
    
    def _create_template_guidance(self, template_structure: Optional[Dict[str, Any]]) -> str:
        """Create template guidance section.
        
        Args:
            template_structure: Template structure for guidance
            
        Returns:
            Formatted template guidance section
        """
        if not template_structure:
            return ""
        
        parent_fields = template_structure.get('parent_product', {}).get('fields', {})
        variant_fields = template_structure.get('child_variants', {}).get('fields', {})
        
        if not parent_fields and not variant_fields:
            return ""
        
        return f"""
TEMPLATE GUIDANCE:
Parent-level fields (shared across variants): {list(parent_fields.keys())[:10]}
Variant-level fields (unique per variant): {list(variant_fields.keys())[:10]}
"""
    
    def _get_mapping_instructions(self) -> str:
        """Get mapping instructions.
        
        Returns:
            Formatted instructions section
        """
        instructions = [
            "Map source fields to mandatory fields based on semantic meaning",
            "Consider German language context and Amazon marketplace requirements",
            "Use template guidance to categorize parent vs variant level data",
            "Only map with confidence >70%",
            "Provide clear reasoning for each mapping decision",
            "Prioritize exact matches over approximate matches"
        ]
        
        return self.create_instruction_section(instructions)
    
    def _get_output_format(self, parent_sku: str) -> str:
        """Get output format specification.
        
        Args:
            parent_sku: Parent SKU for example
            
        Returns:
            Formatted output format section
        """
        format_example = {
            "parent_sku": parent_sku,
            "parent_data": {
                "brand_name": "EIKO",
                "feed_product_type": "LIGHTING_FIXTURE",
                "manufacturer": "EIKO Global LLC"
            },
            "variance_data": {
                "variant_1": {
                    "item_sku": f"{parent_sku}-001",
                    "color_name": "White",
                    "size_name": "Medium"
                },
                "variant_2": {
                    "item_sku": f"{parent_sku}-002",
                    "color_name": "Black",
                    "size_name": "Large"
                }
            },
            "metadata": {
                "confidence": 0.87,
                "total_mapped_fields": 8,
                "unmapped_mandatory": ["field1", "field2"],
                "processing_notes": "Mapped using template guidance"
            }
        }
        
        return self.create_output_format_section(format_example)
    
    def create_simple_mapping_prompt(
        self,
        parent_sku: str,
        mandatory_fields: Dict[str, Any],
        product_data: Dict[str, Any]
    ) -> str:
        """Create simplified mapping prompt without template guidance.
        
        Args:
            parent_sku: Parent SKU identifier
            mandatory_fields: Mandatory fields dictionary
            product_data: Product data dictionary
            
        Returns:
            Simplified mapping prompt
        """
        context = {
            'parent_sku': parent_sku,
            'mandatory_fields': mandatory_fields,
            'product_data': product_data
        }
        
        return self.render_mapping_prompt(context)
    
    def create_retry_mapping_prompt(
        self,
        context: Dict[str, Any],
        previous_error: str,
        attempt_number: int
    ) -> str:
        """Create retry prompt with error context.
        
        Args:
            context: Original mapping context
            previous_error: Error from previous attempt
            attempt_number: Current attempt number
            
        Returns:
            Retry mapping prompt with error guidance
        """
        base_prompt = self.render_mapping_prompt(context)
        
        retry_guidance = f"""

RETRY ATTEMPT #{attempt_number}:
Previous attempt failed with error: {previous_error}
Please ensure your response is valid JSON and follows the exact format specified.
Focus on providing a complete, well-structured response.
"""
        
        return base_prompt + retry_guidance