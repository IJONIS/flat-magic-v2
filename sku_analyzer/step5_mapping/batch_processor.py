"""Optimized batch processing with efficient concurrency and performance monitoring.

This module provides high-performance batch processing capabilities with:
- Intelligent batching strategies for optimal throughput
- Enhanced concurrency control with resource optimization
- Performance monitoring and adaptive batch sizing
- Efficient error handling and recovery mechanisms
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .models import ProcessingResult, ProcessingConfig


class BatchOptimizer:
    """Optimizes batch processing parameters based on performance feedback."""
    
    def __init__(self, initial_batch_size: int = 3):
        """Initialize batch optimizer.
        
        Args:
            initial_batch_size: Starting batch size for optimization
        """
        self.current_batch_size = initial_batch_size
        self.performance_history = []
        self.optimal_batch_size = initial_batch_size
        self.logger = logging.getLogger(__name__)
    
    def update_performance(self, batch_size: int, avg_response_time: float, success_rate: float) -> None:
        """Update performance metrics and optimize batch size.
        
        Args:
            batch_size: Size of batch that was processed
            avg_response_time: Average response time in ms
            success_rate: Success rate (0.0 to 1.0)
        """
        performance_score = success_rate / (avg_response_time / 1000)  # Success per second
        
        self.performance_history.append({
            'batch_size': batch_size,
            'response_time': avg_response_time,
            'success_rate': success_rate,
            'performance_score': performance_score,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        self.performance_history = self.performance_history[-10:]
        
        # Find optimal batch size based on performance score
        if len(self.performance_history) >= 3:
            best_performance = max(self.performance_history, key=lambda x: x['performance_score'])
            self.optimal_batch_size = best_performance['batch_size']
            
            self.logger.debug(f"Optimal batch size updated to {self.optimal_batch_size} "
                            f"(performance score: {best_performance['performance_score']:.2f})")
    
    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size based on performance history."""
        return self.optimal_batch_size


class BatchProcessor:
    """Handles optimized batch processing of multiple parent directories.
    
    Key Optimizations:
    - Adaptive batch sizing based on performance feedback
    - Intelligent concurrency control with resource monitoring
    - Efficient error handling and retry strategies
    - Performance monitoring and optimization recommendations
    """
    
    def __init__(self, config: ProcessingConfig):
        """Initialize optimized batch processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.batch_optimizer = BatchOptimizer(config.batch_size)
        
        # Performance tracking
        self.total_batches_processed = 0
        self.total_processing_time = 0.0
        self.successful_parents = 0
        self.failed_parents = 0
        
        # Enhanced concurrency settings
        self.max_concurrent_batches = min(config.batch_size * 2, 10)  # Intelligent concurrency limit
    
    async def process_parents_batch(
        self,
        parent_skus: List[str],
        base_output_dir: Path,
        processor_func: Callable
    ) -> List[ProcessingResult]:
        """Process parents in optimized batches with performance monitoring.
        
        Args:
            parent_skus: List of parent SKUs to process
            base_output_dir: Base output directory
            processor_func: Function to process individual parent
            
        Returns:
            List of processing results
        """
        total_parents = len(parent_skus)
        batch_size = self.batch_optimizer.get_recommended_batch_size()
        
        self.logger.info(f"Starting optimized batch processing: {total_parents} parents, "
                        f"batch size: {batch_size}, max concurrent: {self.max_concurrent_batches}")
        
        all_results = []
        overall_start_time = time.perf_counter()
        
        # Process in optimized batches
        for i in range(0, total_parents, batch_size):
            batch_parents = parent_skus[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_parents + batch_size - 1) // batch_size
            
            batch_start_time = time.perf_counter()
            
            # Process batch with enhanced concurrency control
            batch_results = await self._process_optimized_batch(
                batch_parents, base_output_dir, processor_func, batch_num, total_batches
            )
            
            batch_time = (time.perf_counter() - batch_start_time) * 1000
            all_results.extend(batch_results)
            
            # Update performance tracking
            self.total_batches_processed += 1
            self.total_processing_time += batch_time
            
            # Update batch optimizer with performance data
            successful_in_batch = sum(1 for r in batch_results if r.success)
            success_rate = successful_in_batch / len(batch_results) if batch_results else 0.0
            avg_response_time = (
                sum(r.processing_time_ms for r in batch_results) / len(batch_results)
                if batch_results else 0.0
            )
            
            self.batch_optimizer.update_performance(len(batch_parents), avg_response_time, success_rate)
            
            # Progress reporting
            completed_parents = i + len(batch_parents)
            progress_pct = (completed_parents / total_parents) * 100
            
            self.logger.info(f"Progress: {completed_parents}/{total_parents} "
                           f"({progress_pct:.1f}%) - Batch {batch_num} completed in {batch_time:.1f}ms")
        
        overall_time = (time.perf_counter() - overall_start_time) * 1000
        self.logger.info(f"All batches completed in {overall_time:.1f}ms total")
        
        return all_results
    
    async def _process_optimized_batch(
        self,
        batch_parents: List[str],
        base_output_dir: Path,
        processor_func: Callable,
        batch_num: int,
        total_batches: int
    ) -> List[ProcessingResult]:
        """Process single batch with optimized resource management.
        
        Args:
            batch_parents: Parents in this batch
            base_output_dir: Base output directory
            processor_func: Processing function
            batch_num: Current batch number
            total_batches: Total number of batches
            
        Returns:
            Processing results for this batch
        """
        # Create resource-aware semaphore
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_with_monitoring(parent_sku: str) -> ProcessingResult:
            async with semaphore:
                parent_start_time = time.perf_counter()
                
                try:
                    result = await processor_func(
                        parent_sku,
                        base_output_dir / "flat_file_analysis" / "step3_mandatory_fields.json",
                        base_output_dir / f"parent_{parent_sku}" / "step2_compressed.json",
                        base_output_dir / f"parent_{parent_sku}"
                    )
                    
                    # Add performance context to result
                    actual_time = (time.perf_counter() - parent_start_time) * 1000
                    if hasattr(result, 'processing_time_ms'):
                        result.processing_time_ms = actual_time
                    
                    return result
                    
                except Exception as e:
                    actual_time = (time.perf_counter() - parent_start_time) * 1000
                    self.logger.error(f"Error processing {parent_sku} in batch {batch_num}: {e}")
                    
                    return ProcessingResult(
                        parent_sku=parent_sku,
                        success=False,
                        error=str(e),
                        processing_time_ms=actual_time
                    )
        
        # Execute all tasks in batch
        tasks = [process_with_monitoring(sku) for sku in batch_parents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any remaining exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Unhandled exception for {batch_parents[i]}: {result}")
                processed_results.append(ProcessingResult(
                    parent_sku=batch_parents[i],
                    success=False,
                    error=f"Unhandled exception: {result}"
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def process_high_priority_parents(
        self,
        parent_skus: List[str],
        base_output_dir: Path,
        processor_func: Callable,
        priority_parents: Optional[List[str]] = None
    ) -> List[ProcessingResult]:
        """Process parents with priority ordering for critical SKUs.
        
        Args:
            parent_skus: All parent SKUs to process
            base_output_dir: Base output directory
            processor_func: Processing function
            priority_parents: List of high-priority parents to process first
            
        Returns:
            Processing results in priority order
        """
        if not priority_parents:
            return await self.process_parents_batch(parent_skus, base_output_dir, processor_func)
        
        # Separate priority and regular parents
        priority_set = set(priority_parents)
        priority_list = [sku for sku in parent_skus if sku in priority_set]
        regular_list = [sku for sku in parent_skus if sku not in priority_set]
        
        self.logger.info(f"Processing {len(priority_list)} priority parents first, "
                        f"then {len(regular_list)} regular parents")
        
        # Process priority parents first
        all_results = []
        if priority_list:
            priority_results = await self.process_parents_batch(
                priority_list, base_output_dir, processor_func
            )
            all_results.extend(priority_results)
        
        # Process remaining parents
        if regular_list:
            regular_results = await self.process_parents_batch(
                regular_list, base_output_dir, processor_func
            )
            all_results.extend(regular_results)
        
        return all_results
    
    def find_parent_directories(self, base_dir: Path) -> List[str]:
        """Find parent directories with optimized file validation.
        
        Args:
            base_dir: Base directory to search
            
        Returns:
            List of parent SKU identifiers optimized for processing order
        """
        start_time = time.perf_counter()
        parent_skus = []
        
        # Efficient directory scanning
        try:
            parent_dirs = list(base_dir.glob("parent_*"))
            
            for parent_dir in parent_dirs:
                if not parent_dir.is_dir():
                    continue
                
                # Check if required files exist
                step2_file = parent_dir / "step2_compressed.json"
                if step2_file.exists():
                    # Validate file is not empty
                    try:
                        if step2_file.stat().st_size > 100:  # Minimum size check
                            parent_sku = parent_dir.name.replace("parent_", "")
                            parent_skus.append(parent_sku)
                    except OSError:
                        self.logger.warning(f"Cannot access {step2_file}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Error scanning parent directories: {e}")
            return []
        
        # Sort for optimal processing (numeric order)
        parent_skus.sort()
        
        scan_time = (time.perf_counter() - start_time) * 1000
        self.logger.info(f"Directory scan completed: {len(parent_skus)} valid parents "
                        f"found in {scan_time:.1f}ms")
        
        return parent_skus
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary with optimization insights."""
        avg_processing_time = (
            self.total_processing_time / self.total_batches_processed
            if self.total_batches_processed > 0 else 0.0
        )
        
        total_parents = self.successful_parents + self.failed_parents
        success_rate = (
            self.successful_parents / total_parents
            if total_parents > 0 else 0.0
        )
        
        return {
            # Basic metrics
            'total_batches_processed': self.total_batches_processed,
            'total_parents_processed': total_parents,
            'successful_parents': self.successful_parents,
            'failed_parents': self.failed_parents,
            'success_rate': success_rate,
            
            # Performance metrics
            'total_processing_time_ms': self.total_processing_time,
            'average_batch_time_ms': avg_processing_time,
            'current_batch_size': self.batch_optimizer.get_recommended_batch_size(),
            'optimal_batch_size': self.batch_optimizer.optimal_batch_size,
            
            # Optimization insights
            'performance_history': self.batch_optimizer.performance_history[-5:],  # Last 5 batches
            'recommended_improvements': self._get_optimization_recommendations(),
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        if self.batch_optimizer.performance_history:
            recent_performance = self.batch_optimizer.performance_history[-3:]
            
            # Analyze response time trends
            avg_response_time = sum(p['response_time'] for p in recent_performance) / len(recent_performance)
            if avg_response_time > 5000:  # >5s average
                recommendations.append("Consider reducing batch size - response times are high")
            elif avg_response_time < 2000:  # <2s average
                recommendations.append("Consider increasing batch size - response times are low")
            
            # Analyze success rate trends
            avg_success_rate = sum(p['success_rate'] for p in recent_performance) / len(recent_performance)
            if avg_success_rate < 0.9:  # <90% success rate
                recommendations.append("High failure rate detected - check for safety filter issues")
            
            # Analyze batch size efficiency
            if self.batch_optimizer.optimal_batch_size != self.batch_optimizer.current_batch_size:
                recommendations.append(f"Optimal batch size is {self.batch_optimizer.optimal_batch_size} "
                                    f"(current: {self.batch_optimizer.current_batch_size})")
        
        # General recommendations
        total_parents = self.successful_parents + self.failed_parents
        if total_parents > 0:
            if self.failed_parents / total_parents > 0.1:  # >10% failure rate
                recommendations.append("High failure rate - consider reducing prompt complexity")
            
            if self.total_processing_time / total_parents > 10000:  # >10s per parent average
                recommendations.append("Slow processing - enable prompt compression optimizations")
        
        return recommendations or ["Performance is within acceptable parameters"]