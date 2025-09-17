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
        """Get comprehensive system prompt for mapping operations.
        
        Returns:
            System prompt for AI mapping
        """
        return (
            "You are an expert e-commerce data transformation specialist with deep knowledge of German marketplace operations and Amazon's complex product taxonomy. "
            "Your task is to perform a sophisticated mapping of authentic German workwear product data to Amazon's stringent marketplace format, "
            "ensuring complete compliance with all field requirements and maintaining the authentic product characteristics."
        )
    
    def render_mapping_prompt(self, context: Dict[str, Any]) -> str:
        """Render comprehensive mapping prompt with full context.
        
        Args:
            context: Mapping context including parent_sku, mandatory_fields, product_data, template_structure
            
        Returns:
            Comprehensive mapping prompt following optimal Gemini format
        """
        # Validate required context
        required_fields = ['parent_sku', 'mandatory_fields', 'product_data']
        missing = self.validate_prompt_data(required_fields, context)
        if missing:
            raise ValueError(f"Missing required context fields: {missing}")
        
        # Get template structure if available
        template_structure = context.get('template_structure', {})
        
        # Build comprehensive prompt optimized for structured output
        prompt = f"""### **MISSION STATEMENT**
{self.get_system_prompt()}

### **CRITICAL REQUIREMENTS**
1. **ZERO TRUNCATION**: Process ALL product variants - every single variant must be mapped
2. **AUTHENTIC DATA**: Use only real source values from the compressed product data - NO placeholder or mock data
3. **COMPLETE FIELD COVERAGE**: Map all 23 mandatory Amazon marketplace fields (14 parent + 9 variant fields)
4. **STRUCTURED OUTPUT**: Data will be automatically formatted using the predefined schema structure
5. **INHERITANCE ACCURACY**: Properly implement parent-child field inheritance
6. **BUSINESS LOGIC**: Apply intelligent mapping decisions based on product characteristics
7. **OUTPUT LANGUAGE**: German

---

## 📋 FIELD MAPPING REQUIREMENTS

### Parent Data Fields (14 required):
- age_range_description: Target age range
- bottoms_size_class: Size classification system
- bottoms_size_system: Specific size system (e.g., DE/NL/SE/PL)
- brand_name: Product brand name
- country_of_origin: Manufacturing country
- department_name: Product department/category
- external_product_id_type: ID type (EAN, UPC, etc.)
- fabric_type: Primary fabric material
- feed_product_type: Product type for feeds
- item_name: Product name/title
- main_image_url: Primary product image URL
- outer_material_type: Outer layer material
- recommended_browse_nodes: Category/browse node ID
- target_gender: Target gender (Männlich/Weiblich/Unisex)

### Variant Data Fields (9 required per variant):
- color_map: Standardized color name
- color_name: Specific color name
- external_product_id: Unique variant identifier (EAN)
- item_sku: Variant SKU
- list_price_with_tax: Price including tax
- quantity: Available quantity
- size_map: Standardized size name
- size_name: Specific size name
- standard_price: Base price excluding tax

---

## 📊 SOURCE PRODUCT DATA

```json
{self.format_json_data(context['product_data'])}
```

---

## 🧠 MAPPING INSTRUCTIONS

Transform the authentic German workwear product data above to populate all required fields:

1. **Extract Parent Information**: Identify common attributes shared across all variants
2. **Map Individual Variants**: Process each unique size/color combination
3. **Apply Business Logic**: Use intelligent field derivation and German marketplace knowledge
4. **Ensure Data Quality**: Validate all mappings and apply appropriate defaults where needed

**Key Mapping Guidelines:**
- Use authentic source values wherever possible
- Apply German marketplace conventions
- Derive missing values using business logic
- Maintain consistency across variants
- Follow Amazon's field validation requirements

The structured output schema will automatically format your response correctly."""
        
        return prompt
    
    def _create_template_guidance(self, template_structure: Optional[Dict[str, Any]]) -> str:
        """Create template guidance section - now returns empty string.
        
        Args:
            template_structure: Template structure for guidance (unused)
            
        Returns:
            Empty string (no template guidance)
        """
        return ""
    
    def _get_mapping_instructions(self) -> str:
        """Get mapping instructions.
        
        Returns:
            Formatted instructions section
        """
        instructions = [
            "Map source fields to mandatory fields based on semantic meaning",
            "Consider German language context and Amazon marketplace requirements",
            "Categorize fields as parent-level (shared) or variant-level (unique per item)",
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
            "variant_data": {
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
    
    def create_comprehensive_mapping_prompt(
        self,
        parent_sku: str,
        mandatory_fields: Dict[str, Any],
        product_data: Dict[str, Any],
        template_structure: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create comprehensive mapping prompt with full context.
        
        Args:
            parent_sku: Parent SKU identifier
            mandatory_fields: Mandatory fields dictionary
            product_data: Product data dictionary
            template_structure: Template structure from step4_template.json
            
        Returns:
            Comprehensive mapping prompt
        """
        context = {
            'parent_sku': parent_sku,
            'mandatory_fields': mandatory_fields,
            'product_data': product_data,
            'template_structure': template_structure
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
    
