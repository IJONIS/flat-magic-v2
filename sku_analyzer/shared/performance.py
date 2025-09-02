"""Unified performance monitoring for all pipeline operations.

This module consolidates performance monitoring functionality
from across the pipeline into a single, consistent interface.
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    
    operation_name: str
    duration_ms: float
    peak_memory_mb: float
    memory_delta_mb: float
    rows_processed: int = 0
    files_processed: int = 0
    api_calls_made: int = 0
    
    @property
    def throughput_rows_per_second(self) -> float:
        """Calculate row processing throughput."""
        if self.duration_ms > 0 and self.rows_processed > 0:
            return (self.rows_processed / self.duration_ms) * 1000
        return 0.0
    
    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.duration_ms / 1000


class PerformanceMonitor:
    """Unified performance monitor for all pipeline operations.
    
    Features:
    - Memory usage tracking with tracemalloc and psutil
    - Operation duration measurement
    - Throughput calculation
    - Performance target validation
    - Structured logging
    """
    
    def __init__(self, enable_monitoring: bool = True) -> None:
        """Initialize performance monitor.
        
        Args:
            enable_monitoring: Whether to enable performance tracking
        """
        self.enable_monitoring = enable_monitoring
        self.logger = logging.getLogger(__name__)
        
        # Initialize process monitor if available
        self.process = None
        if PSUTIL_AVAILABLE:
            try:
                self.process = psutil.Process()
            except (psutil.NoSuchProcess, AttributeError):
                self.process = None
    
    @contextmanager
    def measure_performance(
        self, 
        operation_name: str, 
        row_count: int = 0,
        expected_files: int = 0
    ) -> Generator[Dict[str, Optional[PerformanceMetrics]], None, None]:
        """Context manager for measuring operation performance.
        
        Args:
            operation_name: Name of the operation being measured
            row_count: Number of rows being processed
            expected_files: Expected number of files to be created
            
        Yields:
            Dictionary container for metrics (populated on exit)
        """
        if not self.enable_monitoring:
            yield {'metrics': None}
            return
        
        # Start memory tracking
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
                rows_processed=row_count,
                files_processed=expected_files
            )
            
            metrics_container['metrics'] = metrics
            
            self.logger.info(
                f"⚡ {operation_name}: {duration_ms:.1f}ms, "
                f"peak: {peak_memory:.1f}MB, delta: {memory_delta:+.1f}MB"
            )
            
            # Stop memory tracking
            try:
                tracemalloc.stop()
            except Exception:
                pass
    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB.
        
        Returns:
            Current memory usage in megabytes
        """
        # Try psutil first (more accurate)
        if self.process:
            try:
                return self.process.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, AttributeError):
                pass
        
        # Fallback to tracemalloc
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
        metrics: PerformanceMetrics,
        max_duration_ms: float = 5000,
        max_memory_mb: float = 100.0
    ) -> Dict[str, bool]:
        """Validate performance against targets.
        
        Args:
            metrics: Performance metrics to validate
            max_duration_ms: Maximum allowed duration
            max_memory_mb: Maximum allowed memory usage
            
        Returns:
            Dictionary of validation results
        """
        results = {
            'duration_target_met': metrics.duration_ms <= max_duration_ms,
            'memory_target_met': metrics.peak_memory_mb <= max_memory_mb,
            'overall_performance_acceptable': True
        }
        
        # Overall performance check
        results['overall_performance_acceptable'] = (
            results['duration_target_met'] and 
            results['memory_target_met']
        )
        
        # Log warnings for target violations
        if not results['duration_target_met']:
            self.logger.warning(
                f"Performance: Duration {metrics.duration_ms:.1f}ms "
                f"exceeds target {max_duration_ms}ms for {metrics.operation_name}"
            )
        
        if not results['memory_target_met']:
            self.logger.warning(
                f"Performance: Memory {metrics.peak_memory_mb:.1f}MB "
                f"exceeds target {max_memory_mb}MB for {metrics.operation_name}"
            )
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system resource information."""
        system_info = {
            'psutil_available': PSUTIL_AVAILABLE,
            'current_memory_mb': self._get_current_memory_mb(),
            'tracemalloc_active': tracemalloc.is_tracing()
        }
        
        if self.process:
            try:
                system_info.update({
                    'cpu_percent': self.process.cpu_percent(),
                    'memory_percent': self.process.memory_percent(),
                    'num_threads': self.process.num_threads()
                })
            except (psutil.NoSuchProcess, AttributeError):
                pass
        
        return system_info