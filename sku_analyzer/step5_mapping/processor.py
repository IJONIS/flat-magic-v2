"""Main AI mapping processor with template-driven intelligence.

This module coordinates the AI mapping process using templates
generated in step 3, with comprehensive error handling and
performance monitoring.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sku_analyzer.shared.gemini_client import GeminiClient, AIProcessingConfig
from sku_analyzer.shared.performance import PerformanceMonitor
from .models import (
    MappingInput, TransformationResult, ProcessingConfig, 
    ProcessingResult, BatchProcessingResult, MappingResult
)
from .format_enforcer import FormatEnforcer
from .batch_processor import BatchProcessor
from .result_formatter import ResultFormatter
from .ai_mapper import AIMapper


class MappingProcessor:
    """Main processor for AI-powered product data mapping.
    
    This processor coordinates the entire AI mapping workflow including:
    - Template-guided field mapping
    - Multi-strategy AI processing (with fallback)
    - Format validation and enforcement
    - Performance monitoring and statistics
    - Batch processing with concurrency control
    """
    
    def __init__(
        self, 
        config: Optional[ProcessingConfig] = None,
        ai_config: Optional[AIProcessingConfig] = None,
        enable_performance_monitoring: bool = True
    ):
        """Initialize AI mapping processor.
        
        Args:
            config: Processing configuration
            ai_config: AI client configuration
            enable_performance_monitoring: Whether to enable performance tracking
        """
        self.config = config or ProcessingConfig()
        self.ai_config = ai_config or AIProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(enable_performance_monitoring)
        self.format_enforcer = FormatEnforcer()
        self.batch_processor = BatchProcessor(self.config)
        self.result_formatter = ResultFormatter()
        
        # Initialize AI client
        self.ai_client = GeminiClient(
            config=self.ai_config,
            performance_monitor=self.performance_monitor
        )
        
        # Initialize AI mapper
        self.ai_mapper = AIMapper(self.ai_client, self.config, self.result_formatter)
    
    async def process_parent_directory(
        self,
        parent_sku: str,
        step4_template_path: Path,
        step2_path: Path,
        output_dir: Path
    ) -> ProcessingResult:
        """Process single parent directory with template-driven AI mapping.
        
        Args:
            parent_sku: Parent SKU identifier
            step4_template_path: Path to step4_template.json
            step2_path: Path to step2_compressed.json
            output_dir: Output directory for results
            
        Returns:
            Processing result
        """
        start_time = time.time()
        self.logger.info(f"Processing parent {parent_sku}")
        
        try:
            with self.performance_monitor.measure_performance(
                f"process_parent_{parent_sku}"
            ) as perf:
                # Load input data
                template_data = await self.result_formatter.load_json_async(step4_template_path)
                product_data = await self.result_formatter.load_json_async(step2_path)
                
                # Extract template structure and field definitions
                template_structure = template_data.get("template_structure", {})
                
                # Create enhanced mapping input with template guidance
                mapping_input = MappingInput(
                    parent_sku=parent_sku,
                    mandatory_fields=self.result_formatter.extract_template_fields(template_structure),
                    product_data=product_data,
                    business_context="German Amazon marketplace product",
                    template_structure=template_structure
                )
                
                # Execute mapping with retry logic
                mapping_result = await self.ai_mapper.execute_mapping_with_retry(mapping_input)
                
                # Enforce format compliance
                compliant_result, format_warnings = self.format_enforcer.enforce_format(
                    mapping_result.model_dump(), parent_sku, strict=False
                )
                
                if format_warnings:
                    self.logger.warning(
                        f"Format warnings for {parent_sku}: {format_warnings}"
                    )
                
                # Save result
                output_file = output_dir / "step5_ai_mapping.json"
                await self.result_formatter.save_compliant_result(compliant_result, output_file)
                
                # Update statistics
                processing_time = time.time() - start_time
                self.result_formatter.update_processing_stats(mapping_result, processing_time)
                
                return ProcessingResult(
                    parent_sku=parent_sku,
                    success=True,
                    mapped_fields_count=compliant_result.get("metadata", {}).get("total_variants", 0),
                    unmapped_count=len(compliant_result.get("metadata", {}).get("unmapped_mandatory_fields", [])),
                    confidence=compliant_result.get("metadata", {}).get("mapping_confidence", 0.0),
                    processing_time_ms=processing_time * 1000,
                    output_file=str(output_file),
                    format_warnings=len(format_warnings),
                    format_compliant=True
                )
                
        except Exception as e:
            self.logger.error(f"Failed to process parent {parent_sku}: {e}")
            self.result_formatter.processing_stats["failed_mappings"] += 1
            
            return ProcessingResult(
                parent_sku=parent_sku,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def process_all_parents(
        self,
        base_output_dir: Path,
        starting_parent: str = "4301"
    ) -> BatchProcessingResult:
        """Process all parent directories with AI mapping.
        
        Args:
            base_output_dir: Base output directory containing parent folders
            starting_parent: Starting parent for validation
            
        Returns:
            Complete processing summary
        """
        self.logger.info("Starting AI mapping for all parents")
        
        # Find all parent directories with required files
        parent_dirs = self.batch_processor.find_parent_directories(base_output_dir)
        
        if not parent_dirs:
            raise ValueError(f"No parent directories found in {base_output_dir}")
        
        # Process parents starting with specified one
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
            if result.success:
                remaining_parents = [p for p in parent_dirs if p != starting_parent]
                
                # Process remaining parents in batches
                remaining_results = await self.batch_processor.process_parents_batch(
                    remaining_parents, base_output_dir, self.process_parent_directory
                )
                results.extend(remaining_results)
        else:
            # Process all parents
            all_results = await self.batch_processor.process_parents_batch(
                parent_dirs, base_output_dir, self.process_parent_directory
            )
            results.extend(all_results)
        
        # Generate comprehensive summary
        return self.result_formatter.generate_processing_summary(
            results, self.ai_client.get_performance_summary()
        )
    
    
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.result_formatter.get_performance_stats(
            self.ai_client.get_performance_summary()
        )