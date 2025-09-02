"""AI mapping execution logic for template-driven transformations.

This module provides AI mapping capabilities including prompt generation,
response parsing, and retry logic for robust mapping operations.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from sku_analyzer.shared.gemini_client import GeminiClient
from .models import MappingInput, TransformationResult, MappingResult, ProcessingConfig


class AIMapper:
    """Handles AI mapping operations with retry logic and error handling.
    
    Provides comprehensive AI mapping capabilities including prompt generation,
    response parsing, and quality validation with retry mechanisms.
    """
    
    def __init__(self, ai_client: GeminiClient, config: ProcessingConfig, result_formatter):
        """Initialize AI mapper.
        
        Args:
            ai_client: Configured Gemini AI client
            config: Processing configuration
            result_formatter: Result formatter for statistics tracking
        """
        self.ai_client = ai_client
        self.config = config
        self.result_formatter = result_formatter
        self.logger = logging.getLogger(__name__)
    
    async def execute_mapping_with_retry(
        self,
        mapping_input: MappingInput,
        max_retries: Optional[int] = None
    ) -> TransformationResult:
        """Execute mapping with retry logic and comprehensive error handling.
        
        Args:
            mapping_input: Input for mapping
            max_retries: Maximum retry attempts (uses config default if None)
            
        Returns:
            Mapping result
        """
        max_retries = max_retries or self.config.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                # Execute AI mapping
                result = await self._execute_ai_mapping(mapping_input)
                
                # Validate result quality
                if result.metadata.get("confidence", 0.0) >= self.config.confidence_threshold:
                    self.result_formatter.processing_stats["successful_mappings"] += 1
                    return result
                else:
                    self.logger.warning(
                        f"Low confidence result ({result.metadata.get('confidence', 0.0)}) "
                        f"for {mapping_input.parent_sku} (attempt {attempt + 1})"
                    )
                    
                    # On final attempt, return even low-confidence result
                    if attempt == max_retries:
                        self.result_formatter.processing_stats["successful_mappings"] += 1
                        return result
                
            except Exception as e:
                self.logger.warning(
                    f"AI mapping attempt {attempt + 1} failed for "
                    f"{mapping_input.parent_sku}: {e}"
                )
                
                if attempt == max_retries:
                    # Return minimal error result
                    self.result_formatter.processing_stats["failed_mappings"] += 1
                    return TransformationResult(
                        parent_sku=mapping_input.parent_sku,
                        parent_data={},
                        variance_data={},
                        metadata={
                            "confidence": 0.0,
                            "unmapped_mandatory": list(mapping_input.mandatory_fields.keys()),
                            "processing_notes": f"All mapping attempts failed: {e}"
                        }
                    )
        
        # Should not reach here
        raise RuntimeError("Max retries exceeded without result")
    
    async def _execute_ai_mapping(self, mapping_input: MappingInput) -> TransformationResult:
        """Execute AI mapping using Gemini client.
        
        Args:
            mapping_input: Input for mapping
            
        Returns:
            Transformation result
        """
        # Create mapping prompt
        prompt = self._create_mapping_prompt(mapping_input)
        
        # Make AI request
        response = await self.ai_client.generate_mapping(
            prompt=prompt,
            operation_name=f"map_parent_{mapping_input.parent_sku}"
        )
        
        # Parse and validate response
        json_data = await self.ai_client.validate_json_response(response)
        
        # Convert to TransformationResult
        return self._parse_ai_response(json_data, mapping_input.parent_sku)
    
    def _create_mapping_prompt(self, mapping_input: MappingInput) -> str:
        """Create AI mapping prompt using template guidance.
        
        Args:
            mapping_input: Mapping input data
            
        Returns:
            Formatted prompt string
        """
        # Limit data size for prompt efficiency
        limited_fields = dict(list(mapping_input.mandatory_fields.items())[:15])
        product_data = mapping_input.product_data.get('parent_data', {})
        limited_product = dict(list(product_data.items())[:20]) if product_data else {}
        
        # Extract template guidance if available
        template_guidance = ""
        if mapping_input.template_structure:
            parent_fields = mapping_input.template_structure.get('parent_product', {}).get('fields', {})
            variant_fields = mapping_input.template_structure.get('child_variants', {}).get('fields', {})
            
            template_guidance = f"""
TEMPLATE GUIDANCE:
Parent-level fields (shared across variants): {list(parent_fields.keys())[:10]}
Variant-level fields (unique per variant): {list(variant_fields.keys())[:10]}
"""
        
        return f"""You are an expert German Amazon product data mapper. Map product data to mandatory fields using template guidance.

TASK: Map product data for parent SKU {mapping_input.parent_sku} to mandatory Amazon fields.
{template_guidance}
MANDATORY FIELDS (showing first 15):
{json.dumps(limited_fields, indent=2, ensure_ascii=False)}

PRODUCT DATA (showing first 20):
{json.dumps(limited_product, indent=2, ensure_ascii=False)}

INSTRUCTIONS:
1. Map source fields to mandatory fields based on semantic meaning
2. Consider German language context and Amazon marketplace requirements
3. Use template guidance to categorize parent vs variant level data
4. Only map with confidence >70%
5. Provide clear reasoning for each mapping

REQUIRED JSON OUTPUT:
{{
  "parent_sku": "{mapping_input.parent_sku}",
  "parent_data": {{
    "brand_name": "EIKO",
    "feed_product_type": "LIGHTING_FIXTURE"
  }},
  "variance_data": {{
    "variant_1": {{
      "item_sku": "4301-001",
      "color_name": "White"
    }}
  }},
  "metadata": {{
    "confidence": 0.87,
    "total_mapped_fields": 8,
    "unmapped_mandatory": ["field1", "field2"],
    "processing_notes": "Mapped using template guidance"
  }}
}}

Map the product data now:"""
    
    def _parse_ai_response(
        self, 
        json_data: Dict[str, Any], 
        parent_sku: str
    ) -> TransformationResult:
        """Parse AI response into TransformationResult.
        
        Args:
            json_data: Parsed JSON response from AI
            parent_sku: Parent SKU identifier
            
        Returns:
            TransformationResult object
        """
        # Extract main components
        parent_data = json_data.get("parent_data", {})
        variance_data = json_data.get("variance_data", {})
        metadata = json_data.get("metadata", {})
        
        # Create mapped fields list (if available)
        mapped_fields = []
        if "mapped_fields" in json_data:
            for field_mapping in json_data["mapped_fields"]:
                if isinstance(field_mapping, dict):
                    mapped_fields.append(MappingResult(
                        source_field=field_mapping.get("source_field", "unknown"),
                        target_field=field_mapping.get("target_field", "unknown"),
                        mapped_value=field_mapping.get("mapped_value", ""),
                        confidence=field_mapping.get("confidence", 0.0),
                        reasoning=field_mapping.get("reasoning", "AI mapping")
                    ))
        
        return TransformationResult(
            parent_sku=parent_sku,
            parent_data=parent_data,
            variance_data=variance_data,
            mapped_fields=mapped_fields,
            metadata=metadata
        )
