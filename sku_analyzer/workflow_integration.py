"""Workflow integration for AI mapping with existing SKU analysis."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.analyzer import SkuPatternAnalyzer
from .utils.performance_monitor import PerformanceMonitor
from .shared.gemini_client import AIProcessingConfig
from .step5_mapping.processor import MappingProcessor


class AIWorkflowIntegration:
    """Integrates AI mapping into existing SKU analysis workflow."""
    
    def __init__(
        self,
        analyzer: Optional[SkuPatternAnalyzer] = None,
        ai_config: Optional[AIProcessingConfig] = None,
        enable_ai: bool = True
    ):
        """Initialize workflow integration.
        
        Args:
            analyzer: Existing SKU analyzer instance (optional)
            ai_config: AI processing configuration
            enable_ai: Whether to enable AI processing
        """
        self.analyzer = analyzer
        self.ai_config = ai_config or AIProcessingConfig()
        self.enable_ai = enable_ai
        self.performance_monitor = PerformanceMonitor()
        self.logger = logging.getLogger(__name__)
    
    async def run_enhanced_workflow(
        self,
        input_path: Path,
        output_path: Path,
        enable_ai_mapping: bool = True
    ) -> Dict[str, Any]:
        """Run the complete workflow with AI mapping integration.
        
        Args:
            input_path: Path to input data
            output_path: Path for output
            enable_ai_mapping: Whether to enable AI mapping step
            
        Returns:
            Processing results and performance metrics
        """
        with self.performance_monitor.track_operation("enhanced_workflow"):
            results = {}
            
            # Step 1-2: Standard SKU analysis
            self.logger.info("Running standard SKU analysis...")
            analysis_results = await self._run_standard_analysis(input_path)
            results.update(analysis_results)
            
            # Step 3-5: AI-enhanced processing
            if enable_ai_mapping:
                self.logger.info("Running AI mapping...")
                ai_processor = AIMappingProcessor(self.ai_config)
                ai_results = await ai_processor.process_data(input_path, output_path)
                results.update(ai_results)
            
            # Performance summary
            results['performance'] = self.performance_monitor.get_summary()
            
            return results
    
    async def process_ai_mapping_step(
        self,
        output_dir: Path,
        starting_parent: str = "4301"
    ) -> Dict[str, Any]:
        """Process AI mapping step for all parents.
        
        Args:
            output_dir: Base output directory containing parent folders
            starting_parent: Starting parent for validation
            
        Returns:
            Processing results with performance metrics
        """
        if not self.enable_ai:
            raise ValueError("AI processing is disabled")
        
        self.logger.info(f"Starting AI mapping step in {output_dir}")
        
        # Initialize AI mapping processor
        ai_processor = MappingProcessor(ai_config=self.ai_config)
        
        # Process all parents
        batch_result = await ai_processor.process_all_parents(
            base_output_dir=output_dir,
            starting_parent=starting_parent
        )
        
        # Format results for compatibility
        results = {
            "ai_mapping_completed": True,
            "summary": {
                "total_parents": len(batch_result.results),
                "successful": sum(1 for r in batch_result.results if r.success),
                "failed": sum(1 for r in batch_result.results if not r.success),
                "success_rate": sum(1 for r in batch_result.results if r.success) / max(len(batch_result.results), 1),
                "average_confidence": sum(r.confidence for r in batch_result.results) / max(len(batch_result.results), 1)
            },
            "performance": {
                "total_processing_time_ms": sum(r.processing_time_ms for r in batch_result.results),
                "average_time_per_parent": sum(r.processing_time_ms for r in batch_result.results) / max(len(batch_result.results), 1)
            },
            "details": [
                {
                    "parent_sku": r.parent_sku,
                    "success": r.success,
                    "mapped_fields_count": r.mapped_fields_count,
                    "confidence": r.confidence,
                    "output_file": r.output_file,
                    "error": r.error
                }
                for r in batch_result.results
            ]
        }
        
        return results
    
    async def _run_standard_analysis(self, input_path: Path) -> Dict[str, Any]:
        """Run the standard SKU analysis steps."""
        # This would integrate with existing analyzer logic
        return {
            "standard_analysis": "completed",
            "files_processed": 0  # Placeholder
        }


async def run_ai_enhanced_workflow(
    input_path: Path,
    output_path: Path,
    ai_config: Optional[AIProcessingConfig] = None
) -> Dict[str, Any]:
    """Convenience function to run AI-enhanced workflow.
    
    Args:
        input_path: Path to input data
        output_path: Path for output
        ai_config: AI processing configuration
        
    Returns:
        Processing results
    """
    # Create analyzer instance (placeholder - would use real analyzer)
    analyzer = None  # SkuPatternAnalyzer would be initialized here
    
    integration = AIWorkflowIntegration(analyzer, ai_config)
    return await integration.run_enhanced_workflow(input_path, output_path)