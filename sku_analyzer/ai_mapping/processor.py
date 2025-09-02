"""Main AI mapping processor coordinating Pydantic AI and Gemini integration."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import AIProcessingConfig, MappingInput, TransformationResult, AITransformationOutput
from .pydantic_agent import ModernPydanticAgent
from .gemini_client import ModernGeminiClient
from .example_loader import ExampleFormatLoader, FormatEnforcer


class AIMappingProcessor:
    """Main processor for AI-powered product data mapping."""
    
    def __init__(
        self, 
        config: Optional[AIProcessingConfig] = None,
        enable_fallback: bool = True
    ):
        """Initialize AI mapping processor.
        
        Args:
            config: AI processing configuration
            enable_fallback: Whether to enable fallback to Gemini client
        """
        self.config = config or AIProcessingConfig()
        self.enable_fallback = enable_fallback
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI clients
        self.pydantic_agent = ModernPydanticAgent(self.config)
        self.gemini_client = ModernGeminiClient(self.config) if enable_fallback else None
        
        # Initialize format validation system
        self.example_loader = ExampleFormatLoader()
        self.format_enforcer = FormatEnforcer(self.example_loader)
        
        # Performance tracking
        self.processing_stats = {
            "total_processed": 0,
            "successful_mappings": 0,
            "failed_mappings": 0,
            "fallback_used": 0,
            "average_confidence": 0.0,
            "total_processing_time": 0.0
        }
    
    async def process_parent_directory(
        self,
        parent_sku: str,
        step4_template_path: Path,
        step2_path: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Process single parent directory with template-driven AI mapping.
        
        Args:
            parent_sku: Parent SKU identifier
            step4_template_path: Path to step4_template.json
            step2_path: Path to step2_compressed.json
            output_dir: Output directory for results
            
        Returns:
            Processing result summary
        """
        start_time = time.time()
        self.logger.info(f"Processing parent {parent_sku}")
        
        try:
            # Load input data
            template_data = await self._load_json_async(step4_template_path)
            product_data = await self._load_json_async(step2_path)
            
            # Extract template structure and field definitions
            template_structure = template_data.get("template_structure", {})
            
            # Create enhanced mapping input with template guidance
            mapping_input = MappingInput(
                parent_sku=parent_sku,
                mandatory_fields=self._extract_template_fields(template_structure),
                product_data=product_data,
                business_context="German Amazon marketplace product",
                template_structure=template_structure  # Add template for AI guidance
            )
            
            # Attempt mapping with Pydantic AI
            mapping_result = await self._execute_mapping_with_retry(mapping_input)
            
            # Enforce format compliance
            compliant_result, format_warnings = self.format_enforcer.enforce_format(
                mapping_result.model_dump(), parent_sku, strict=False
            )
            
            if format_warnings:
                self.logger.warning(
                    f"Format warnings for {parent_sku}: {format_warnings}"
                )
            
            # Save result in compliant format
            output_file = output_dir / f"step5_ai_mapping.json"
            await self._save_compliant_result(compliant_result, output_file)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(mapping_result, processing_time)
            
            return {
                "parent_sku": parent_sku,
                "success": True,
                "mapped_fields_count": compliant_result.get("metadata", {}).get("total_variants", 0),
                "unmapped_count": 0,  # Compliant format doesn't track unmapped fields
                "confidence": compliant_result.get("metadata", {}).get("mapping_confidence", 0.0),
                "processing_time_ms": processing_time * 1000,
                "output_file": str(output_file),
                "format_warnings": len(format_warnings),
                "format_compliant": True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process parent {parent_sku}: {e}")
            self.processing_stats["failed_mappings"] += 1
            
            return {
                "parent_sku": parent_sku,
                "success": False,
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }
    
    async def process_all_parents(
        self,
        base_output_dir: Path,
        starting_parent: str = "4301"
    ) -> Dict[str, Any]:
        """Process all parent directories with AI mapping.
        
        Args:
            base_output_dir: Base output directory containing parent folders
            starting_parent: Starting parent for validation
            
        Returns:
            Complete processing summary
        """
        self.logger.info("Starting AI mapping for all parents")
        
        # Find all parent directories with required files
        parent_dirs = self._find_parent_directories(base_output_dir)
        
        if not parent_dirs:
            raise ValueError(f"No parent directories found in {base_output_dir}")
        
        # Start with specified parent if exists
        results = []
        if starting_parent in parent_dirs:
            self.logger.info(f"Processing starting parent {starting_parent} first")
            
            result = await self.process_parent_directory(
                starting_parent,
                base_output_dir / "flat_file_analysis" / "step4_template.json",
                base_output_dir / f"parent_{starting_parent}" / "step2_compressed.json",
                base_output_dir / f"parent_{starting_parent}"
            )
            
            results.append(result)
            
            # If successful, continue with remaining parents
            if result["success"]:
                remaining_parents = [p for p in parent_dirs if p != starting_parent]
                
                # Process remaining parents in batches
                remaining_results = await self._process_parents_batch(
                    remaining_parents, base_output_dir
                )
                results.extend(remaining_results)
        else:
            # Process all parents
            all_results = await self._process_parents_batch(parent_dirs, base_output_dir)
            results.extend(all_results)
        
        # Generate summary
        return self._generate_processing_summary(results)
    
    async def _execute_mapping_with_retry(
        self,
        mapping_input: MappingInput,
        max_retries: int = 2
    ) -> TransformationResult:
        """Execute mapping with retry logic and fallback.
        
        Args:
            mapping_input: Input for mapping
            max_retries: Maximum retry attempts
            
        Returns:
            Mapping result
        """
        for attempt in range(max_retries + 1):
            try:
                # Try Pydantic AI agent first
                result = await self.pydantic_agent.map_product_data(mapping_input)
                
                # Validate result quality
                if result.metadata.get("confidence", 0.0) > 0.5:
                    self.processing_stats["successful_mappings"] += 1
                    return result
                else:
                    self.logger.warning(
                        f"Low confidence result ({result.metadata.get('confidence', 0.0)}) "
                        f"for {mapping_input.parent_sku}"
                    )
                
            except Exception as e:
                self.logger.warning(
                    f"Pydantic AI attempt {attempt + 1} failed for "
                    f"{mapping_input.parent_sku}: {e}"
                )
                
                if attempt == max_retries:
                    # Try fallback to direct Gemini client
                    if self.gemini_client:
                        self.logger.info("Attempting fallback to Gemini client")
                        return await self._fallback_to_gemini(mapping_input)
                    else:
                        raise e
        
        # Should not reach here
        raise RuntimeError("Max retries exceeded without result")
    
    async def _fallback_to_gemini(self, mapping_input: MappingInput) -> TransformationResult:
        """Fallback to direct Gemini client.
        
        Args:
            mapping_input: Input for mapping
            
        Returns:
            Mapping result from Gemini
        """
        try:
            # Generate prompt (if prompt_manager exists)
            if hasattr(self, 'prompt_manager') and self.prompt_manager:
                prompt_context = {
                    "parent_sku": mapping_input.parent_sku,
                    "mandatory_fields": mapping_input.mandatory_fields,
                    "product_data": mapping_input.product_data,
                    "business_context": mapping_input.business_context
                }
                
                prompt = self.prompt_manager.render_mapping_prompt(prompt_context)
                
                # Make request
                response = await self.gemini_client.generate_mapping(prompt)
                
                # Parse JSON response
                json_data = await self.gemini_client.validate_json_response(response)
                
                # Convert to TransformationResult
                transformation_result = TransformationResult.model_validate(json_data)
                
                self.processing_stats["fallback_used"] += 1
                self.processing_stats["successful_mappings"] += 1
                
                return transformation_result
            else:
                # No prompt manager available, create basic fallback
                raise RuntimeError("Prompt manager not available for Gemini fallback")
            
        except Exception as e:
            self.logger.error(f"Gemini fallback failed: {e}")
            self.processing_stats["failed_mappings"] += 1
            
            # Return minimal error result
            return TransformationResult(
                parent_sku=mapping_input.parent_sku,
                parent_data={},
                variance_data={},
                metadata={
                    "total_mapped_fields": 0,
                    "confidence": 0.0,
                    "unmapped_mandatory": list(mapping_input.mandatory_fields.keys()),
                    "processing_notes": f"All mapping attempts failed: {e}"
                }
            )
    
    async def _process_parents_batch(
        self,
        parent_skus: List[str],
        base_output_dir: Path
    ) -> List[Dict[str, Any]]:
        """Process parents in batches with controlled concurrency.
        
        Args:
            parent_skus: List of parent SKUs to process
            base_output_dir: Base output directory
            
        Returns:
            List of processing results
        """
        # Create semaphore for batch processing
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_semaphore(parent_sku: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.process_parent_directory(
                    parent_sku,
                    base_output_dir / "flat_file_analysis" / "step4_template.json",
                    base_output_dir / f"parent_{parent_sku}" / "step2_compressed.json",
                    base_output_dir / f"parent_{parent_sku}"
                )
        
        # Execute all tasks
        tasks = [process_with_semaphore(sku) for sku in parent_skus]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def _find_parent_directories(self, base_dir: Path) -> List[str]:
        """Find parent directories with required files.
        
        Args:
            base_dir: Base directory to search
            
        Returns:
            List of parent SKU identifiers
        """
        parent_skus = []
        
        for parent_dir in base_dir.glob("parent_*"):
            if parent_dir.is_dir():
                # Check if required files exist
                step2_file = parent_dir / "step2_compressed.json"
                if step2_file.exists():
                    # Extract parent SKU from directory name
                    parent_sku = parent_dir.name.replace("parent_", "")
                    parent_skus.append(parent_sku)
        
        return parent_skus
    
    async def _load_json_async(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file asynchronously.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded JSON data
        """
        def load_json():
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        
        return await asyncio.to_thread(load_json)
    
    async def _save_mapping_result(
        self,
        transformation_result: TransformationResult,
        output_file: Path
    ) -> None:
        """Save transformation result to file.
        
        Args:
            transformation_result: Result to save
            output_file: Output file path
        """
        def save_json():
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open('w', encoding='utf-8') as f:
                json.dump(transformation_result.model_dump(), f, indent=2, ensure_ascii=False)
        
        await asyncio.to_thread(save_json)
    
    async def _save_compliant_result(
        self,
        compliant_result: Dict[str, Any],
        output_file: Path
    ) -> None:
        """Save compliant result to file.
        
        Args:
            compliant_result: Compliant result dictionary to save
            output_file: Output file path
        """
        def save_json():
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open('w', encoding='utf-8') as f:
                json.dump(compliant_result, f, indent=2, ensure_ascii=False)
        
        await asyncio.to_thread(save_json)
    
    def _extract_template_fields(self, template_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Extract field definitions from template structure for AI mapping.
        
        Args:
            template_structure: Template structure from step 4
            
        Returns:
            Field definitions compatible with existing MappingInput format
        """
        extracted_fields = {}
        
        # Extract parent fields
        parent_product = template_structure.get('parent_product', {})
        parent_fields = parent_product.get('fields', {})
        
        for field_name, field_info in parent_fields.items():
            extracted_fields[field_name] = {
                'display_name': field_info.get('display_name', field_name),
                'data_type': field_info.get('data_type', 'string'),
                'constraints': field_info.get('constraints', {}),
                'level': 'parent',
                'validation_rules': field_info.get('validation_rules', {})
            }
        
        # Extract variant fields
        child_variants = template_structure.get('child_variants', {})
        variant_fields = child_variants.get('fields', {})
        
        for field_name, field_info in variant_fields.items():
            extracted_fields[field_name] = {
                'display_name': field_info.get('display_name', field_name),
                'data_type': field_info.get('data_type', 'string'),
                'constraints': field_info.get('constraints', {}),
                'level': 'variant',
                'variation_type': field_info.get('variation_type', 'attribute'),
                'validation_rules': field_info.get('validation_rules', {})
            }
        
        return extracted_fields
    
    def _update_processing_stats(
        self,
        transformation_result: TransformationResult,
        processing_time: float
    ) -> None:
        """Update processing statistics.
        
        Args:
            transformation_result: Completed transformation result
            processing_time: Processing duration in seconds
        """
        self.processing_stats["total_processed"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        # Update confidence average
        current_avg = self.processing_stats["average_confidence"]
        total_processed = self.processing_stats["total_processed"]
        confidence = transformation_result.metadata.get("confidence", 0.0)
        
        self.processing_stats["average_confidence"] = (
            (current_avg * (total_processed - 1) + confidence) 
            / total_processed
        )
    
    def _generate_processing_summary(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive processing summary.
        
        Args:
            results: List of processing results
            
        Returns:
            Summary statistics
        """
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        total_confidence = sum(r.get("confidence", 0.0) for r in successful_results)
        avg_confidence = (
            total_confidence / len(successful_results) 
            if successful_results else 0.0
        )
        
        return {
            "summary": {
                "total_parents": len(results),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "success_rate": len(successful_results) / len(results) if results else 0.0,
                "average_confidence": avg_confidence
            },
            "performance": {
                "total_processing_time": sum(r.get("processing_time_ms", 0) for r in results),
                "average_processing_time": (
                    sum(r.get("processing_time_ms", 0) for r in results) / len(results)
                    if results else 0.0
                ),
                "agent_stats": self.pydantic_agent.get_performance_stats(),
                "processor_stats": self.processing_stats
            },
            "results": results
        }