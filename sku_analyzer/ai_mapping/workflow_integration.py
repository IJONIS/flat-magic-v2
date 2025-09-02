"""Integration module for existing workflow and AI mapping."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.analyzer import SkuAnalyzer
from ..utils.performance_monitor import PerformanceMonitor
from .models import AIProcessingConfig
from .processor import AIMappingProcessor


class AIWorkflowIntegration:
    """Integrates AI mapping into existing SKU analysis workflow."""
    
    def __init__(
        self,
        analyzer: SkuAnalyzer,
        ai_config: Optional[AIProcessingConfig] = None
    ):
        """Initialize workflow integration.
        
        Args:
            analyzer: Existing SKU analyzer instance
            ai_config: AI processing configuration
        """
        self.analyzer = analyzer
        self.ai_config = ai_config or AIProcessingConfig()
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize AI processor
        self.ai_processor = AIMappingProcessor(self.ai_config)
    
    async def run_complete_workflow_with_ai(
        self,
        input_file: Path,
        output_dir: Path,
        enable_ai_mapping: bool = True,
        starting_parent: str = "4301"
    ) -> Dict[str, Any]:
        """Run complete workflow including AI mapping step.
        
        Args:
            input_file: Input Excel/CSV file
            output_dir: Output directory
            enable_ai_mapping: Whether to run AI mapping
            starting_parent: Starting parent for AI processing
            
        Returns:
            Complete workflow results
        """
        workflow_start = asyncio.get_event_loop().time()
        self.logger.info("Starting complete workflow with AI mapping")
        
        try:
            # Step 1-3: Run existing analysis pipeline
            analysis_result = await self._run_base_analysis(input_file, output_dir)
            
            if not analysis_result.get("success", False):
                return {
                    "success": False,
                    "error": "Base analysis failed",
                    "details": analysis_result
                }
            
            # Step 4: AI Mapping (if enabled)
            ai_results = None
            if enable_ai_mapping:
                ai_results = await self._run_ai_mapping_step(
                    output_dir, starting_parent
                )
            
            # Generate complete summary
            total_time = asyncio.get_event_loop().time() - workflow_start
            
            return {
                "success": True,
                "workflow_time_seconds": total_time,
                "base_analysis": analysis_result,
                "ai_mapping": ai_results,
                "output_directory": str(output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_time_seconds": asyncio.get_event_loop().time() - workflow_start
            }
    
    async def _run_base_analysis(
        self,
        input_file: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Run base analysis steps (1-3).
        
        Args:
            input_file: Input file path
            output_dir: Output directory
            
        Returns:
            Base analysis results
        """
        try:
            # Use existing analyzer methods - run synchronously then wrap
            def run_sync_analysis():
                return self.analyzer.analyze_sku_data(
                    input_file=str(input_file),
                    output_dir=str(output_dir),
                    enable_compression=True,
                    enable_csv_export=True
                )
            
            # Run in thread to maintain async interface
            analysis_result = await asyncio.to_thread(run_sync_analysis)
            
            return {
                "success": True,
                "analysis_data": analysis_result
            }
            
        except Exception as e:
            self.logger.error(f"Base analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_ai_mapping_step(
        self,
        output_dir: Path,
        starting_parent: str
    ) -> Dict[str, Any]:
        """Run AI mapping step on analysis results.
        
        Args:
            output_dir: Output directory with analysis results
            starting_parent: Starting parent SKU
            
        Returns:
            AI mapping results
        """
        try:
            # Process AI mapping
            ai_summary = await self.ai_processor.process_all_parents(
                output_dir, starting_parent
            )
            
            self.logger.info(
                f"AI mapping completed: {ai_summary['summary']['successful']}/"
                f"{ai_summary['summary']['total_parents']} parents processed"
            )
            
            return {
                "success": True,
                "summary": ai_summary
            }
            
        except Exception as e:
            self.logger.error(f"AI mapping step failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_ai_mapping_config(
        self,
        temperature: float = 0.1,
        batch_size: int = 10,
        max_concurrent: int = 5,
        timeout_seconds: int = 30
    ) -> AIProcessingConfig:
        """Create AI processing configuration.
        
        Args:
            temperature: Model temperature
            batch_size: Batch processing size
            max_concurrent: Maximum concurrent requests
            timeout_seconds: Request timeout
            
        Returns:
            AI processing configuration
        """
        return AIProcessingConfig(
            model_name="gemini-2.5-flash",
            temperature=temperature,
            max_tokens=4096,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            max_concurrent=max_concurrent
        )


async def run_ai_enhanced_workflow(
    input_file: str,
    output_dir: str,
    enable_ai: bool = True,
    starting_parent: str = "4301",
    **ai_config_kwargs
) -> Dict[str, Any]:
    """Convenience function for running AI-enhanced workflow.
    
    Args:
        input_file: Input file path
        output_dir: Output directory path
        enable_ai: Enable AI mapping step
        starting_parent: Starting parent SKU
        **ai_config_kwargs: AI configuration parameters
        
    Returns:
        Complete workflow results
    """
    # Initialize components
    analyzer = SkuAnalyzer()
    
    # Create AI config
    ai_config = AIProcessingConfig(
        model_name="gemini-2.5-flash",
        temperature=ai_config_kwargs.get('temperature', 0.1),
        batch_size=ai_config_kwargs.get('batch_size', 10),
        max_concurrent=ai_config_kwargs.get('max_concurrent', 5),
        timeout_seconds=ai_config_kwargs.get('timeout_seconds', 30)
    )
    
    # Initialize workflow integration
    workflow = AIWorkflowIntegration(analyzer, ai_config)
    
    # Run complete workflow
    return await workflow.run_complete_workflow_with_ai(
        Path(input_file),
        Path(output_dir),
        enable_ai,
        starting_parent
    )