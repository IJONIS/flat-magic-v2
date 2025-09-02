"""Comprehensive performance monitoring for AI mapping workflow end-to-end test."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import tracemalloc
import psutil
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import pandas as pd


@dataclass
class PipelineStageMetrics:
    """Performance metrics for individual pipeline stages."""
    stage_name: str
    start_timestamp: float
    end_timestamp: float
    duration_ms: float
    memory_start_mb: float
    memory_peak_mb: float
    memory_end_mb: float
    cpu_percent: float
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    rows_processed: int = 0
    files_created: int = 0
    api_calls_made: int = 0
    
    @property
    def memory_delta_mb(self) -> float:
        """Calculate memory change during stage."""
        return self.memory_end_mb - self.memory_start_mb
    
    @property
    def throughput_rows_per_second(self) -> float:
        """Calculate row processing throughput."""
        if self.duration_ms > 0 and self.rows_processed > 0:
            return (self.rows_processed / self.duration_ms) * 1000
        return 0.0


@dataclass
class APIPerformanceMetrics:
    """Performance metrics for Gemini API interactions."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    rate_limited_calls: int = 0
    total_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0
    total_tokens_used: int = 0
    total_prompt_tokens: int = 0
    total_response_tokens: int = 0
    retry_attempts: int = 0
    
    @property
    def average_response_time_ms(self) -> float:
        """Calculate average API response time."""
        return self.total_response_time_ms / self.total_calls if self.total_calls > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate API success rate."""
        return self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0
    
    @property
    def tokens_per_request(self) -> float:
        """Calculate average tokens per request."""
        return self.total_tokens_used / self.total_calls if self.total_calls > 0 else 0.0


@dataclass
class ResourceUtilizationMetrics:
    """System resource utilization during workflow execution."""
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    average_cpu_percent: float = 0.0
    total_disk_read_mb: float = 0.0
    total_disk_write_mb: float = 0.0
    concurrent_operations_peak: int = 0
    memory_samples: List[float] = field(default_factory=list)
    cpu_samples: List[float] = field(default_factory=list)
    
    def add_sample(self, memory_mb: float, cpu_percent: float) -> None:
        """Add resource utilization sample."""
        self.memory_samples.append(memory_mb)
        self.cpu_samples.append(cpu_percent)
        
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
        
        self.average_memory_mb = sum(self.memory_samples) / len(self.memory_samples)
        self.average_cpu_percent = sum(self.cpu_samples) / len(self.cpu_samples)


@dataclass
class E2EPerformanceResults:
    """Complete end-to-end performance test results."""
    test_name: str
    test_timestamp: str
    total_duration_ms: float
    pipeline_stages: Dict[str, PipelineStageMetrics] = field(default_factory=dict)
    api_performance: APIPerformanceMetrics = field(default_factory=APIPerformanceMetrics)
    resource_utilization: ResourceUtilizationMetrics = field(default_factory=ResourceUtilizationMetrics)
    performance_targets_met: Dict[str, bool] = field(default_factory=dict)
    bottlenecks_identified: List[str] = field(default_factory=list)
    optimization_recommendations: List[str] = field(default_factory=list)
    baseline_comparison: Optional[Dict[str, float]] = None


class APIPerformanceTracker:
    """Tracks Gemini API performance metrics."""
    
    def __init__(self):
        self.metrics = APIPerformanceMetrics()
        self.logger = logging.getLogger(__name__)
    
    @asynccontextmanager
    async def track_api_call(self, operation_name: str = "gemini_request"):
        """Context manager for tracking individual API calls."""
        start_time = time.perf_counter()
        call_successful = False
        
        try:
            self.metrics.total_calls += 1
            yield self
            call_successful = True
            self.metrics.successful_calls += 1
            
        except asyncio.TimeoutError:
            self.metrics.timeout_calls += 1
            self.metrics.failed_calls += 1
            raise
        except Exception as e:
            if "rate limit" in str(e).lower():
                self.metrics.rate_limited_calls += 1
            self.metrics.failed_calls += 1
            raise
        finally:
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            
            self.metrics.total_response_time_ms += response_time_ms
            self.metrics.min_response_time_ms = min(self.metrics.min_response_time_ms, response_time_ms)
            self.metrics.max_response_time_ms = max(self.metrics.max_response_time_ms, response_time_ms)
            
            self.logger.info(
                f"API {operation_name}: {response_time_ms:.1f}ms, "
                f"{'success' if call_successful else 'failed'}"
            )
    
    def record_token_usage(self, prompt_tokens: int, response_tokens: int) -> None:
        """Record token usage from API response."""
        self.metrics.total_prompt_tokens += prompt_tokens
        self.metrics.total_response_tokens += response_tokens
        self.metrics.total_tokens_used += prompt_tokens + response_tokens
    
    def record_retry_attempt(self) -> None:
        """Record API retry attempt."""
        self.metrics.retry_attempts += 1


class ResourceUtilizationTracker:
    """Tracks system resource utilization during workflow execution."""
    
    def __init__(self, sampling_interval: float = 0.5):
        self.sampling_interval = sampling_interval
        self.metrics = ResourceUtilizationMetrics()
        self.logger = logging.getLogger(__name__)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False
        
        # Initialize process monitor
        try:
            self.process = psutil.Process()
            self.initial_io = self.process.io_counters()
        except (psutil.NoSuchProcess, AttributeError):
            self.process = None
            self.initial_io = None
    
    async def start_monitoring(self) -> None:
        """Start continuous resource monitoring."""
        self._stop_monitoring = False
        self._monitoring_task = asyncio.create_task(self._monitor_resources())
    
    async def stop_monitoring(self) -> None:
        """Stop resource monitoring and finalize metrics."""
        self._stop_monitoring = True
        if self._monitoring_task and not self._monitoring_task.done():
            await self._monitoring_task
        
        # Calculate final disk I/O
        if self.process and self.initial_io:
            try:
                final_io = self.process.io_counters()
                self.metrics.total_disk_read_mb = (final_io.read_bytes - self.initial_io.read_bytes) / 1024 / 1024
                self.metrics.total_disk_write_mb = (final_io.write_bytes - self.initial_io.write_bytes) / 1024 / 1024
            except (psutil.NoSuchProcess, AttributeError):
                pass
    
    async def _monitor_resources(self) -> None:
        """Continuous resource monitoring loop."""
        while not self._stop_monitoring:
            try:
                if self.process:
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    cpu_percent = self.process.cpu_percent()
                    
                    self.metrics.add_sample(memory_mb, cpu_percent)
                
                await asyncio.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.warning(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.sampling_interval)


class E2EPerformanceMonitor:
    """Comprehensive end-to-end performance monitoring for AI mapping workflow."""
    
    def __init__(self, enable_detailed_monitoring: bool = True):
        self.enable_detailed_monitoring = enable_detailed_monitoring
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking components
        self.api_tracker = APIPerformanceTracker()
        self.resource_tracker = ResourceUtilizationTracker()
        
        # Performance data storage
        self.results = E2EPerformanceResults(
            test_name="ai_mapping_e2e_test",
            test_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_duration_ms=0.0
        )
        
        # Stage tracking
        self.current_stage: Optional[str] = None
        self.stage_start_time: Optional[float] = None
        self.workflow_start_time: Optional[float] = None
    
    async def start_workflow_monitoring(self) -> None:
        """Start comprehensive workflow performance monitoring."""
        self.workflow_start_time = time.perf_counter()
        
        # Start memory tracing
        tracemalloc.start()
        
        # Start resource monitoring
        if self.enable_detailed_monitoring:
            await self.resource_tracker.start_monitoring()
        
        self.logger.info("Started E2E performance monitoring")
    
    async def stop_workflow_monitoring(self) -> None:
        """Stop workflow monitoring and finalize results."""
        if self.workflow_start_time:
            total_duration = time.perf_counter() - self.workflow_start_time
            self.results.total_duration_ms = total_duration * 1000
        
        # Stop resource monitoring
        if self.enable_detailed_monitoring:
            await self.resource_tracker.stop_monitoring()
            self.results.resource_utilization = self.resource_tracker.metrics
        
        # Stop memory tracing
        try:
            tracemalloc.stop()
        except Exception:
            pass
        
        # Analyze performance and generate insights
        self._analyze_performance_targets()
        self._identify_bottlenecks()
        self._generate_optimization_recommendations()
        
        self.logger.info(f"E2E monitoring complete: {self.results.total_duration_ms:.1f}ms total")
    
    @asynccontextmanager
    async def monitor_pipeline_stage(
        self, 
        stage_name: str, 
        expected_rows: int = 0,
        expected_files: int = 0
    ):
        """Context manager for monitoring individual pipeline stages."""
        # Finalize previous stage if any
        if self.current_stage:
            await self._finalize_current_stage()
        
        # Start new stage
        self.current_stage = stage_name
        self.stage_start_time = time.perf_counter()
        
        # Capture starting metrics
        start_memory = self._get_current_memory_mb()
        start_cpu = self._get_current_cpu_percent()
        
        stage_context = {
            'start_memory': start_memory,
            'expected_rows': expected_rows,
            'expected_files': expected_files
        }
        
        self.logger.info(f"Starting stage: {stage_name}")
        
        try:
            yield stage_context
        finally:
            # Finalize stage on exit
            await self._finalize_current_stage()
    
    async def _finalize_current_stage(self) -> None:
        """Finalize metrics for current pipeline stage."""
        if not self.current_stage or not self.stage_start_time:
            return
        
        end_time = time.perf_counter()
        duration_ms = (end_time - self.stage_start_time) * 1000
        
        # Capture ending metrics
        end_memory = self._get_current_memory_mb()
        end_cpu = self._get_current_cpu_percent()
        
        # Get peak memory from tracemalloc
        try:
            _, peak_memory_bytes = tracemalloc.get_traced_memory()
            peak_memory_mb = peak_memory_bytes / 1024 / 1024
        except Exception:
            peak_memory_mb = end_memory
        
        # Create stage metrics
        stage_metrics = PipelineStageMetrics(
            stage_name=self.current_stage,
            start_timestamp=self.stage_start_time,
            end_timestamp=end_time,
            duration_ms=duration_ms,
            memory_start_mb=self._get_current_memory_mb(),
            memory_peak_mb=peak_memory_mb,
            memory_end_mb=end_memory,
            cpu_percent=end_cpu
        )
        
        # Store stage metrics
        self.results.pipeline_stages[self.current_stage] = stage_metrics
        
        self.logger.info(
            f"Stage {self.current_stage} complete: {duration_ms:.1f}ms, "
            f"peak memory: {peak_memory_mb:.1f}MB"
        )
        
        # Reset current stage
        self.current_stage = None
        self.stage_start_time = None
    
    def get_api_performance_tracker(self) -> APIPerformanceTracker:
        """Get API performance tracker for external use."""
        return self.api_tracker
    
    def record_stage_details(
        self, 
        rows_processed: int = 0, 
        files_created: int = 0,
        api_calls_made: int = 0
    ) -> None:
        """Record additional details for current stage."""
        if self.current_stage and self.current_stage in self.results.pipeline_stages:
            stage = self.results.pipeline_stages[self.current_stage]
            stage.rows_processed = rows_processed
            stage.files_created = files_created
            stage.api_calls_made = api_calls_made
    
    def _analyze_performance_targets(self) -> None:
        """Analyze performance against defined targets."""
        targets = {}
        
        # Overall pipeline target: <60 seconds
        targets['overall_pipeline_60s'] = self.results.total_duration_ms < 60000
        
        # AI mapping per parent target: <5 seconds
        ai_mapping_stage = self.results.pipeline_stages.get('ai_mapping')
        if ai_mapping_stage:
            # Estimate per-parent time (assuming 6 parents)
            per_parent_ms = ai_mapping_stage.duration_ms / 6
            targets['ai_mapping_per_parent_5s'] = per_parent_ms < 5000
        
        # API response time target: <2 seconds per call
        targets['api_response_2s'] = self.api_tracker.metrics.average_response_time_ms < 2000
        
        # Memory usage target: <500MB peak
        targets['memory_500mb'] = self.results.resource_utilization.peak_memory_mb < 500
        
        # API success rate target: >95%
        targets['api_success_95pct'] = self.api_tracker.metrics.success_rate > 0.95
        
        self.results.performance_targets_met = targets
    
    def _identify_bottlenecks(self) -> None:
        """Identify performance bottlenecks in the pipeline."""
        bottlenecks = []
        
        # Analyze stage durations
        stage_durations = {
            name: metrics.duration_ms 
            for name, metrics in self.results.pipeline_stages.items()
        }
        
        if stage_durations:
            # Find slowest stage
            slowest_stage = max(stage_durations, key=stage_durations.get)
            slowest_duration = stage_durations[slowest_stage]
            
            # If one stage takes >40% of total time, it's a bottleneck
            if slowest_duration > (self.results.total_duration_ms * 0.4):
                bottlenecks.append(f"Stage '{slowest_stage}' bottleneck: {slowest_duration:.1f}ms ({slowest_duration/self.results.total_duration_ms*100:.1f}% of total)")
        
        # API performance bottlenecks
        api_metrics = self.api_tracker.metrics
        if api_metrics.average_response_time_ms > 2000:
            bottlenecks.append(f"API response time bottleneck: {api_metrics.average_response_time_ms:.1f}ms average")
        
        if api_metrics.success_rate < 0.95:
            bottlenecks.append(f"API reliability bottleneck: {api_metrics.success_rate:.1%} success rate")
        
        # Memory bottlenecks
        if self.results.resource_utilization.peak_memory_mb > 400:
            bottlenecks.append(f"Memory usage bottleneck: {self.results.resource_utilization.peak_memory_mb:.1f}MB peak")
        
        # CPU bottlenecks
        if self.results.resource_utilization.peak_cpu_percent > 80:
            bottlenecks.append(f"CPU utilization bottleneck: {self.results.resource_utilization.peak_cpu_percent:.1f}% peak")
        
        self.results.bottlenecks_identified = bottlenecks
    
    def _generate_optimization_recommendations(self) -> None:
        """Generate actionable optimization recommendations."""
        recommendations = []
        
        # API optimization recommendations
        api_metrics = self.api_tracker.metrics
        if api_metrics.average_response_time_ms > 1500:
            recommendations.append("Optimize Gemini API prompts to reduce response time")
            recommendations.append("Implement request batching to reduce API overhead")
        
        if api_metrics.rate_limited_calls > 0:
            recommendations.append("Implement exponential backoff for rate limit handling")
            recommendations.append("Reduce concurrent API requests to stay within rate limits")
        
        # Memory optimization recommendations
        if self.results.resource_utilization.peak_memory_mb > 300:
            recommendations.append("Implement streaming processing to reduce memory footprint")
            recommendations.append("Use memory-efficient data structures for large datasets")
        
        # Performance bottleneck recommendations
        stage_durations = {
            name: metrics.duration_ms 
            for name, metrics in self.results.pipeline_stages.items()
        }
        
        if stage_durations:
            # CSV processing optimization
            csv_duration = stage_durations.get('csv_splitting', 0)
            if csv_duration > 10000:  # >10 seconds
                recommendations.append("Optimize CSV splitting with parallel processing")
                recommendations.append("Use faster DataFrame operations for large datasets")
            
            # Compression optimization
            compression_duration = stage_durations.get('json_compression', 0)
            if compression_duration > 8000:  # >8 seconds
                recommendations.append("Use orjson library for faster JSON serialization")
                recommendations.append("Implement parallel compression for multiple files")
            
            # AI mapping optimization
            ai_duration = stage_durations.get('ai_mapping', 0)
            if ai_duration > 20000:  # >20 seconds total
                recommendations.append("Optimize AI mapping prompts for faster processing")
                recommendations.append("Implement caching for similar product mappings")
        
        # Concurrency recommendations
        concurrent_peak = self.results.resource_utilization.concurrent_operations_peak
        if concurrent_peak < 3:
            recommendations.append("Increase concurrency for parallel operations")
        elif concurrent_peak > 8:
            recommendations.append("Reduce concurrency to prevent resource contention")
        
        self.results.optimization_recommendations = recommendations
    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            if hasattr(self, 'process') and self.process:
                return self.process.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, AttributeError):
            pass
        
        try:
            current, _ = tracemalloc.get_traced_memory()
            return current / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_current_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            if hasattr(self, 'process') and self.process:
                return self.process.cpu_percent()
        except (psutil.NoSuchProcess, AttributeError):
            pass
        return 0.0
    
    async def save_performance_report(self, output_dir: Path) -> Path:
        """Save comprehensive performance report."""
        # Finalize API metrics
        self.results.api_performance = self.api_tracker.metrics
        
        report_file = output_dir / "e2e_performance_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        report_data = {
            'test_metadata': {
                'test_name': self.results.test_name,
                'timestamp': self.results.test_timestamp,
                'total_duration_ms': self.results.total_duration_ms,
                'total_duration_seconds': self.results.total_duration_ms / 1000
            },
            'pipeline_stages': {
                name: {
                    'duration_ms': metrics.duration_ms,
                    'duration_seconds': metrics.duration_ms / 1000,
                    'memory_peak_mb': metrics.memory_peak_mb,
                    'memory_delta_mb': metrics.memory_delta_mb,
                    'throughput_rows_per_second': metrics.throughput_rows_per_second,
                    'rows_processed': metrics.rows_processed,
                    'files_created': metrics.files_created,
                    'api_calls_made': metrics.api_calls_made
                }
                for name, metrics in self.results.pipeline_stages.items()
            },
            'api_performance': {
                'total_calls': self.results.api_performance.total_calls,
                'success_rate': self.results.api_performance.success_rate,
                'average_response_time_ms': self.results.api_performance.average_response_time_ms,
                'min_response_time_ms': self.results.api_performance.min_response_time_ms,
                'max_response_time_ms': self.results.api_performance.max_response_time_ms,
                'total_tokens_used': self.results.api_performance.total_tokens_used,
                'tokens_per_request': self.results.api_performance.tokens_per_request,
                'failed_calls': self.results.api_performance.failed_calls,
                'timeout_calls': self.results.api_performance.timeout_calls,
                'rate_limited_calls': self.results.api_performance.rate_limited_calls,
                'retry_attempts': self.results.api_performance.retry_attempts
            },
            'resource_utilization': {
                'peak_memory_mb': self.results.resource_utilization.peak_memory_mb,
                'average_memory_mb': self.results.resource_utilization.average_memory_mb,
                'peak_cpu_percent': self.results.resource_utilization.peak_cpu_percent,
                'average_cpu_percent': self.results.resource_utilization.average_cpu_percent,
                'total_disk_read_mb': self.results.resource_utilization.total_disk_read_mb,
                'total_disk_write_mb': self.results.resource_utilization.total_disk_write_mb,
                'memory_samples_count': len(self.results.resource_utilization.memory_samples)
            },
            'performance_analysis': {
                'targets_met': self.results.performance_targets_met,
                'bottlenecks_identified': self.results.bottlenecks_identified,
                'optimization_recommendations': self.results.optimization_recommendations
            }
        }
        
        # Save report
        with report_file.open('w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Performance report saved: {report_file}")
        return report_file
    
    def compare_with_baseline(self, baseline_file: Optional[Path] = None) -> None:
        """Compare current performance with baseline measurements."""
        if not baseline_file or not baseline_file.exists():
            self.logger.info("No baseline file provided - current run will serve as baseline")
            return
        
        try:
            with baseline_file.open('r') as f:
                baseline_data = json.load(f)
            
            # Compare key metrics
            current_total = self.results.total_duration_ms
            baseline_total = baseline_data.get('test_metadata', {}).get('total_duration_ms', 0)
            
            if baseline_total > 0:
                performance_ratio = current_total / baseline_total
                
                comparison = {
                    'baseline_duration_ms': baseline_total,
                    'current_duration_ms': current_total,
                    'performance_ratio': performance_ratio,
                    'performance_change': 'improved' if performance_ratio < 1.0 else 'degraded',
                    'change_percentage': ((performance_ratio - 1.0) * 100)
                }
                
                self.results.baseline_comparison = comparison
                
                self.logger.info(
                    f"Baseline comparison: {comparison['change_percentage']:+.1f}% "
                    f"({comparison['performance_change']})"
                )
        
        except Exception as e:
            self.logger.warning(f"Failed to compare with baseline: {e}")
    
    def get_real_time_performance_summary(self) -> Dict[str, Any]:
        """Get real-time performance summary during execution."""
        current_time = time.perf_counter()
        elapsed_ms = 0
        
        if self.workflow_start_time:
            elapsed_ms = (current_time - self.workflow_start_time) * 1000
        
        return {
            'elapsed_time_ms': elapsed_ms,
            'elapsed_time_seconds': elapsed_ms / 1000,
            'current_stage': self.current_stage,
            'stages_completed': len(self.results.pipeline_stages),
            'api_calls_made': self.api_tracker.metrics.total_calls,
            'api_success_rate': self.api_tracker.metrics.success_rate,
            'current_memory_mb': self._get_current_memory_mb(),
            'peak_memory_mb': self.results.resource_utilization.peak_memory_mb
        }


class PerformanceReportGenerator:
    """Generates comprehensive performance analysis reports."""
    
    @staticmethod
    def generate_executive_summary(results: E2EPerformanceResults) -> str:
        """Generate executive summary of performance test results."""
        total_seconds = results.total_duration_ms / 1000
        targets_met = sum(results.performance_targets_met.values())
        total_targets = len(results.performance_targets_met)
        
        summary = f"""
E2E AI Mapping Workflow Performance Summary
==========================================

Test: {results.test_name}
Timestamp: {results.test_timestamp}
Total Duration: {total_seconds:.1f}s ({results.total_duration_ms:.0f}ms)
Performance Targets: {targets_met}/{total_targets} met

API Performance:
- Total API calls: {results.api_performance.total_calls}
- Success rate: {results.api_performance.success_rate:.1%}
- Average response: {results.api_performance.average_response_time_ms:.1f}ms
- Token usage: {results.api_performance.total_tokens_used:,} tokens

Resource Utilization:
- Peak memory: {results.resource_utilization.peak_memory_mb:.1f}MB
- Average memory: {results.resource_utilization.average_memory_mb:.1f}MB
- Peak CPU: {results.resource_utilization.peak_cpu_percent:.1f}%
- Disk I/O: {results.resource_utilization.total_disk_read_mb:.1f}MB read, {results.resource_utilization.total_disk_write_mb:.1f}MB write
"""
        
        if results.bottlenecks_identified:
            summary += f"\nBottlenecks Identified:\n"
            for bottleneck in results.bottlenecks_identified:
                summary += f"- {bottleneck}\n"
        
        if results.optimization_recommendations:
            summary += f"\nOptimization Recommendations:\n"
            for i, rec in enumerate(results.optimization_recommendations[:5], 1):
                summary += f"{i}. {rec}\n"
        
        return summary
    
    @staticmethod
    def generate_stage_breakdown(results: E2EPerformanceResults) -> str:
        """Generate detailed stage-by-stage performance breakdown."""
        breakdown = "\nPipeline Stage Performance Breakdown:\n"
        breakdown += "=" * 45 + "\n"
        
        for stage_name, metrics in results.pipeline_stages.items():
            duration_s = metrics.duration_ms / 1000
            percentage = (metrics.duration_ms / results.total_duration_ms) * 100
            
            breakdown += f"\n{stage_name}:\n"
            breakdown += f"  Duration: {duration_s:.2f}s ({percentage:.1f}% of total)\n"
            breakdown += f"  Memory: {metrics.memory_peak_mb:.1f}MB peak, {metrics.memory_delta_mb:+.1f}MB delta\n"
            breakdown += f"  Throughput: {metrics.throughput_rows_per_second:.0f} rows/s\n"
            breakdown += f"  Files created: {metrics.files_created}\n"
            
            if metrics.api_calls_made > 0:
                breakdown += f"  API calls: {metrics.api_calls_made}\n"
        
        return breakdown