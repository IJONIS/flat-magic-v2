"""Simple AI mapping execution following SOLID principles.

This module provides focused AI mapping capabilities:
- Create structured prompt using mandatory fields from step3
- Send request to AI API
- Return parsed result

No fallbacks, caching, or complex retry logic per CLAUDE.md requirements.
No dependencies on step4_1_structure_example.json - structure generated internally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from sku_analyzer.shared.gemini_client import GeminiClient
from sku_analyzer.shared.content_sanitizer import ContentSanitizer
from .models import MappingInput, TransformationResult


class AIMapper:
    """Handles AI mapping operations with single responsibility.
    
    Core workflow:
    1. Load required files from job directory
    2. Create 4-component prompt structure
    3. Send to AI API
    4. Return result
    """
    
    def __init__(self, ai_client: GeminiClient):
        """Initialize AI mapper with client.
        
        Args:
            ai_client: Configured Gemini AI client
        """
        self.ai_client = ai_client
        self.sanitizer = ContentSanitizer()
        self.logger = logging.getLogger(__name__)
    
    async def execute_ai_mapping(
        self, 
        mapping_input: MappingInput, 
        job_dir: Path
    ) -> TransformationResult:
        """Execute AI mapping using aggregated files and structured prompt.
        
        Implements progressive sanitization retry logic for safety filter handling:
        1. Original content
        2. Basic sanitization (known triggers)  
        3. Aggressive sanitization (demographic terms)
        4. Direct field mapping fallback
        
        Args:
            mapping_input: Input for mapping containing compressed product data
            job_dir: Job directory containing required files
            
        Returns:
            Transformation result from AI mapping
            
        Raises:
            FileNotFoundError: When required files are missing
            ValueError: When file content is invalid
        """
        # Aggregate required files
        required_files = self._load_required_files(job_dir)
        
        # Progressive sanitization attempts
        attempts = [
            ("original", mapping_input, False),
            ("basic_sanitization", self._sanitize_mapping_input(mapping_input), False),
            ("aggressive_sanitization", self._sanitize_mapping_input(mapping_input, aggressive=True), True),
        ]
        
        last_error = None
        
        for attempt_name, sanitized_input, is_aggressive in attempts:
            try:
                self.logger.info(f"Attempting AI mapping with {attempt_name} for parent {mapping_input.parent_sku}")
                
                # Create prompt with sanitized input
                prompt = self._create_structured_prompt(sanitized_input, required_files)
                
                # Send AI request using structured output
                response = await self.ai_client.generate_structured_mapping(
                    prompt=prompt,
                    operation_name=f"structured_map_{mapping_input.parent_sku}_{attempt_name}"
                )
                
                # Parse and return result
                json_data = await self.ai_client.validate_json_response(response)
                result = self._parse_ai_response(json_data, mapping_input.parent_sku)
                
                # Add sanitization metadata
                if result.metadata:
                    result.metadata["sanitization_level"] = attempt_name
                    result.metadata["content_sanitized"] = attempt_name != "original"
                    
                self.logger.info(f"AI mapping succeeded with {attempt_name} for parent {mapping_input.parent_sku}")
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                last_error = e
                
                # Check if it's a safety filter error
                if "safety" in error_msg or "blocked" in error_msg or "content" in error_msg:
                    self.logger.warning(f"Safety filter blocked {attempt_name} for parent {mapping_input.parent_sku}: {e}")
                    continue
                else:
                    # Non-safety error, don't retry
                    self.logger.error(f"Non-safety error in {attempt_name} for parent {mapping_input.parent_sku}: {e}")
                    break
        
        # All attempts failed, return error result
        self.logger.error(f"All AI mapping attempts failed for parent {mapping_input.parent_sku}: {last_error}")
        
        return TransformationResult(
            parent_sku=mapping_input.parent_sku,
            parent_data={},
            variant_data={},
            metadata={
                "confidence": 0.0,
                "processing_notes": f"All mapping attempts failed: {last_error}",
                "sanitization_attempts": [attempt[0] for attempt in attempts],
                "final_error": str(last_error)
            }
        )
    
    def _load_required_files(self, job_dir: Path) -> Dict[str, Any]:
        """Load required files from job directory.
        
        Args:
            job_dir: Job directory containing flat_file_analysis
            
        Returns:
            Dictionary containing loaded file data (empty dict as no files needed)
        """
        # No longer loading step4_1_structure_example.json - structure example generated internally
        return {}
    
    def _load_json_file(self, file_path: Path, file_description: str) -> Dict[str, Any]:
        """Load and parse JSON file.
        
        Args:
            file_path: Path to JSON file
            file_description: Description for error messages
            
        Returns:
            Parsed JSON data
            
        Raises:
            FileNotFoundError: When file doesn't exist
            ValueError: When file content is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"{file_description} not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded {file_description}")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_description}: {e}")
        except OSError as e:
            raise ValueError(f"Failed to read {file_description}: {e}")
    
    
    def _create_structured_prompt(
        self,
        mapping_input: MappingInput, 
        required_files: Dict[str, Any]
    ) -> str:
        """Create comprehensive prompt following the optimal Gemini testing format.
        
        Args:
            mapping_input: Input data containing product information
            required_files: Required file data (unused)
            
        Returns:
            Comprehensive prompt string for AI mapping
        """
        # Use the mapping prompts manager for comprehensive prompt generation
        from sku_analyzer.prompts.mapping_prompts import MappingPromptManager
        
        prompt_manager = MappingPromptManager()
        
        prompt = prompt_manager.create_comprehensive_mapping_prompt(
            parent_sku=mapping_input.parent_sku,
            mandatory_fields=mapping_input.mandatory_fields,
            product_data=mapping_input.product_data,
            template_structure=mapping_input.template_structure
        )
        
        return prompt
    
    def _parse_ai_response(
        self, 
        json_data: Any, 
        parent_sku: str
    ) -> TransformationResult:
        """Parse AI response into TransformationResult.
        
        Handles both structured output format (parent_data + variants array)
        and legacy formats for backward compatibility.
        
        Args:
            json_data: Parsed JSON response from AI (structured output format)
            parent_sku: Parent SKU identifier
            
        Returns:
            TransformationResult object
        """
        # Handle list response (extract first item)
        if isinstance(json_data, list) and json_data:
            json_data = json_data[0]
        
        # Validate response is dict
        if not isinstance(json_data, dict):
            return TransformationResult(
                parent_sku=parent_sku,
                parent_data={},
                variant_data={},
                metadata={
                    "confidence": 0.8,
                    "processing_notes": f"Invalid AI response format: {type(json_data).__name__}",
                    "structured_output": False
                }
            )
        
        # Extract parent data (structured output format)
        parent_data = json_data.get("parent_data", {})
        if not isinstance(parent_data, dict):
            parent_data = {}
        
        # Handle structured output variants array format
        variant_data = {}
        variants_array = json_data.get("variants", [])
        
        if isinstance(variants_array, list) and variants_array:
            # Convert structured output variants array to legacy format
            for i, variant in enumerate(variants_array):
                if isinstance(variant, dict):
                    variant_data[f"variant_{i+1}"] = variant
        else:
            # Fallback to legacy formats for backward compatibility
            # Check for 'variant_data' format first
            raw_variant_data = json_data.get("variant_data", None)
            if raw_variant_data:
                if isinstance(raw_variant_data, list):
                    # Convert list to dict format
                    for i, variant in enumerate(raw_variant_data):
                        if isinstance(variant, dict):
                            variant_data[f"variant_{i+1}"] = variant
                elif isinstance(raw_variant_data, dict):
                    variant_data = raw_variant_data
        
        # Generate metadata for structured output
        metadata = {
            "confidence": 0.95,  # Higher confidence for structured output
            "structured_output": bool(variants_array),
            "total_variants": len(variant_data),
            "parent_fields_count": len(parent_data),
            "processing_notes": "Processed with structured output schema" if variants_array else "Processed with legacy format"
        }
        
        # Add any existing metadata
        raw_metadata = json_data.get("metadata", {})
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)
        
        return TransformationResult(
            parent_sku=parent_sku,
            parent_data=parent_data,
            variant_data=variant_data,
            metadata=metadata
        )
    
    def _safe_get_list(self, obj: Any, key: str, default: list) -> list:
        """Safely get list value from object.
        
        Args:
            obj: Object to get value from
            key: Key to look for
            default: Default value if not found or wrong type
            
        Returns:
            List value or default
        """
        if not isinstance(obj, dict):
            return default
        
        value = obj.get(key, default)
        return value if isinstance(value, list) else default
    
    def _sanitize_mapping_input(self, mapping_input: MappingInput, aggressive: bool = False) -> MappingInput:
        """Create sanitized copy of mapping input.
        
        Args:
            mapping_input: Original mapping input
            aggressive: Use aggressive sanitization mode
            
        Returns:
            New MappingInput with sanitized content
        """
        # Create sanitizer with appropriate mode
        sanitizer = ContentSanitizer(aggressive_mode=aggressive)
        
        # Sanitize product data
        sanitized_product_data = sanitizer.sanitize_product_data(mapping_input.product_data)
        
        # Create new mapping input with sanitized data
        return MappingInput(
            parent_sku=mapping_input.parent_sku,
            mandatory_fields=mapping_input.mandatory_fields,  # Don't sanitize field definitions
            product_data=sanitized_product_data,
            template_structure=mapping_input.template_structure  # Don't sanitize template structure
        )