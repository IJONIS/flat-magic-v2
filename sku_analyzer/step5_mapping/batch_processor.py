"""Batch processing utilities for AI mapping operations.

This module provides batch processing capabilities with concurrency control,
performance monitoring, and comprehensive error handling.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from .models import ProcessingResult, ProcessingConfig


class BatchProcessor:
    """Handles batch processing of multiple parent directories.
    
    Provides controlled concurrency, error handling, and performance
    monitoring for processing multiple parent SKUs.
    """
    
    def __init__(self, config: ProcessingConfig):
        """Initialize batch processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def process_parents_batch(
        self,
        parent_skus: List[str],
        base_output_dir: Path,
        processor_func
    ) -> List[ProcessingResult]:
        """Process parents in batches with controlled concurrency.
        
        Args:
            parent_skus: List of parent SKUs to process
            base_output_dir: Base output directory
            processor_func: Function to process individual parent
            
        Returns:
            List of processing results
        """
        # Create semaphore for batch processing
        semaphore = asyncio.Semaphore(self.config.batch_size)
        
        async def process_with_semaphore(parent_sku: str) -> ProcessingResult:
            async with semaphore:
                return await processor_func(
                    parent_sku,
                    base_output_dir / "flat_file_analysis" / "step4_template.json",
                    base_output_dir / f"parent_{parent_sku}" / "step2_compressed.json",
                    base_output_dir / f"parent_{parent_sku}"
                )
        
        # Execute all tasks
        tasks = [process_with_semaphore(sku) for sku in parent_skus]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    parent_sku=parent_skus[i],
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def find_parent_directories(self, base_dir: Path) -> List[str]:
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
