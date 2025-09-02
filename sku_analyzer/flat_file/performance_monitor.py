"""Performance monitoring utilities for template analysis.

This module provides focused functionality for tracking and measuring
performance metrics during template analysis operations.
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from contextlib import contextmanager
from typing import Dict, Generator, Optional

from .data_structures import PerformanceMetrics


class PerformanceMonitor:
    """Context manager for performance monitoring."""
    
    def __init__(self, enable_monitoring: bool = True) -> None:
        """Initialize performance monitor.
        
        Args:
            enable_monitoring: Whether to enable performance tracking
        """
        self.enable_monitoring = enable_monitoring
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def measure_performance(
        self, 
        operation_name: str, 
        row_count: int = 0
    ) -> Generator[Dict[str, Optional[PerformanceMetrics]], None, None]:
        """Context manager for measuring operation performance.
        
        Args:
            operation_name: Name of the operation being measured
            row_count: Number of rows being processed
            
        Yields:
            Dictionary container for metrics (populated on exit)
        """
        if not self.enable_monitoring:
            yield {'metrics': None}
            return
            
        tracemalloc.start()
        start_memory = self._get_current_memory_mb()
        start_time = time.perf_counter()
        
        metrics_container = {'metrics': None}
        
        try:
            yield metrics_container
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_current_memory_mb()
            peak_memory = self._get_peak_memory_mb()
            
            duration_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                duration_ms=duration_ms,
                peak_memory_mb=peak_memory,
                memory_delta_mb=memory_delta,
                rows_processed=row_count
            )
            
            metrics_container['metrics'] = metrics
            
            self.logger.info(
                f"⚡ {operation_name}: {duration_ms:.1f}ms, "
                f"peak: {peak_memory:.1f}MB, delta: {memory_delta:+.1f}MB"
            )
            
            tracemalloc.stop()
    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB.
        
        Returns:
            Current memory usage in megabytes
        """
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            try:
                current, _ = tracemalloc.get_traced_memory()
                return current / 1024 / 1024
            except Exception:
                return 0.0
    
    def _get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB.
        
        Returns:
            Peak memory usage in megabytes
        """
        try:
            _, peak = tracemalloc.get_traced_memory()
            return peak / 1024 / 1024
        except Exception:
            return 0.0
    
    def validate_performance_targets(
        self, 
        duration_ms: float, 
        memory_mb: float,
        max_duration_ms: float = 1500,
        max_memory_mb: float = 50.0
    ) -> Dict[str, bool]:
        """Validate performance against targets.
        
        Args:
            duration_ms: Actual duration in milliseconds
            memory_mb: Actual memory usage in MB
            max_duration_ms: Maximum allowed duration
            max_memory_mb: Maximum allowed memory usage
            
        Returns:
            Dictionary of validation results
        """
        results = {
            'duration_target_met': duration_ms <= max_duration_ms,
            'memory_target_met': memory_mb <= max_memory_mb
        }
        
        if not results['duration_target_met']:
            self.logger.warning(
                f"Performance: Duration {duration_ms:.1f}ms exceeds target {max_duration_ms}ms"
            )
        
        if not results['memory_target_met']:
            self.logger.warning(
                f"Performance: Memory {memory_mb:.1f}MB exceeds target {max_memory_mb}MB"
            )
        
        return results
