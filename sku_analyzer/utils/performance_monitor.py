"""Performance monitoring and benchmarking for CSV export operations."""

import asyncio
import logging
import time
import tracemalloc
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..models import ParentChildRelationship

import pandas as pd


@dataclass
class PerformanceMetrics:
    """Performance measurement data structure."""
    operation_name: str
    duration_seconds: float
    peak_memory_mb: float
    memory_delta_mb: float
    rows_processed: int
    files_created: int
    throughput_rows_per_second: float = field(init=False)
    memory_efficiency_mb_per_1k_rows: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.throughput_rows_per_second = (
            self.rows_processed / self.duration_seconds if self.duration_seconds > 0 else 0
        )
        self.memory_efficiency_mb_per_1k_rows = (
            (self.peak_memory_mb / self.rows_processed) * 1000 if self.rows_processed > 0 else 0
        )


class PerformanceMonitor:
    """Monitor and benchmark CSV export performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[PerformanceMetrics] = []
    
    @asynccontextmanager
    async def measure_operation(self, operation_name: str):
        """Context manager for measuring operation performance."""
        # Start memory tracing
        tracemalloc.start()
        start_memory = self._get_memory_usage_mb()
        start_time = time.perf_counter()
        
        self.logger.info(f"🔄 Starting performance measurement: {operation_name}")
        
        try:
            # Yield control to measured operation
            yield self
            
        finally:
            # Calculate final metrics
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage_mb()
            peak_memory = self._get_peak_memory_mb()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.logger.info(
                f"⚡ {operation_name} completed: "
                f"{duration:.2f}s, peak memory: {peak_memory:.1f}MB, "
                f"delta: {memory_delta:+.1f}MB"
            )
            
            tracemalloc.stop()
    
    def record_metrics(
        self,
        operation_name: str,
        duration_seconds: float,
        peak_memory_mb: float,
        memory_delta_mb: float,
        rows_processed: int,
        files_created: int = 0
    ) -> PerformanceMetrics:
        """Record performance metrics for an operation."""
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            duration_seconds=duration_seconds,
            peak_memory_mb=peak_memory_mb,
            memory_delta_mb=memory_delta_mb,
            rows_processed=rows_processed,
            files_created=files_created
        )
        
        self.metrics_history.append(metrics)
        
        # Log performance summary
        self.logger.info(
            f"📊 Performance: {operation_name} | "
            f"{metrics.throughput_rows_per_second:.0f} rows/s | "
            f"{metrics.memory_efficiency_mb_per_1k_rows:.1f} MB/1k rows"
        )
        
        return metrics
    
    def validate_performance_targets(
        self,
        metrics: PerformanceMetrics,
        max_duration_seconds: float = 5.0,
        max_memory_mb: float = 100.0
    ) -> Dict[str, bool]:
        """Validate performance against target thresholds."""
        validation_results = {
            'duration_target_met': metrics.duration_seconds <= max_duration_seconds,
            'memory_target_met': metrics.peak_memory_mb <= max_memory_mb,
            'overall_pass': True
        }
        
        # Overall pass requires both targets to be met
        validation_results['overall_pass'] = (
            validation_results['duration_target_met'] and 
            validation_results['memory_target_met']
        )
        
        # Log validation results
        if validation_results['overall_pass']:
            self.logger.info(f"✅ Performance targets met for {metrics.operation_name}")
        else:
            self.logger.warning(
                f"⚠️ Performance targets not met for {metrics.operation_name}: "
                f"duration={metrics.duration_seconds:.2f}s (target: {max_duration_seconds}s), "
                f"memory={metrics.peak_memory_mb:.1f}MB (target: {max_memory_mb}MB)"
            )
        
        return validation_results
    
    def get_performance_summary(self) -> Dict:
        """Get summary of all recorded performance metrics."""
        if not self.metrics_history:
            return {'message': 'No performance data recorded'}
        
        latest = self.metrics_history[-1]
        
        return {
            'latest_operation': latest.operation_name,
            'duration_seconds': latest.duration_seconds,
            'peak_memory_mb': latest.peak_memory_mb,
            'throughput_rows_per_second': latest.throughput_rows_per_second,
            'memory_efficiency_mb_per_1k_rows': latest.memory_efficiency_mb_per_1k_rows,
            'files_created': latest.files_created,
            'total_operations_measured': len(self.metrics_history)
        }
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback to tracemalloc if psutil not available
            current, peak = tracemalloc.get_traced_memory()
            return current / 1024 / 1024
    
    def _get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB from tracemalloc."""
        try:
            current, peak = tracemalloc.get_traced_memory()
            return peak / 1024 / 1024
        except Exception:
            return 0.0


class CsvPerformanceBenchmark:
    """Benchmark CSV export operations against performance targets."""
    
    @staticmethod
    async def benchmark_csv_export(
        original_df: pd.DataFrame,
        relationships: Dict[str, ParentChildRelationship],
        job_output_dir: Path
    ) -> PerformanceMetrics:
        """Benchmark CSV export performance with detailed metrics."""
        monitor = PerformanceMonitor()
        
        async with monitor.measure_operation("CSV_Export_Benchmark"):
            # Import here to avoid circular imports
            from ..output.csv_writer import OptimizedCsvWriter, CsvExportProgressTracker
            
            # Create temporary job structure for benchmarking
            from ..models import ProcessingJob
            
            benchmark_job = ProcessingJob(
                job_id="benchmark",
                input_path=Path("benchmark.xlsx"),
                output_dir=job_output_dir / "benchmark"
            )
            
            # Initialize CSV writer and progress tracker
            csv_writer = OptimizedCsvWriter(max_workers=4)
            progress_tracker = CsvExportProgressTracker()
            
            start_time = time.perf_counter()
            start_memory = monitor._get_memory_usage_mb()
            
            # Execute CSV export
            csv_files = await csv_writer.split_and_export_csv(
                benchmark_job,
                relationships,
                original_df,
                progress_tracker
            )
            
            end_time = time.perf_counter()
            peak_memory = monitor._get_peak_memory_mb()
            end_memory = monitor._get_memory_usage_mb()
            
            # Record metrics
            metrics = monitor.record_metrics(
                operation_name="CSV_Export_Complete",
                duration_seconds=end_time - start_time,
                peak_memory_mb=peak_memory,
                memory_delta_mb=end_memory - start_memory,
                rows_processed=len(original_df),
                files_created=len(csv_files)
            )
            
            # Validate against targets
            validation = monitor.validate_performance_targets(metrics)
            
            return metrics