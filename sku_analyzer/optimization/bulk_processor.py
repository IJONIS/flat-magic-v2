"""Bulk compression engine with parallel processing and benchmarking."""

import asyncio
import logging
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np

from .redundancy_analyzer import HighPerformanceRedundancyAnalyzer, RedundancyAnalysis
from .json_compressor import OptimizedJsonCompressor, CompressionMetrics

# Check library availability
try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

try:
    import ujson
    UJSON_AVAILABLE = True
except ImportError:
    UJSON_AVAILABLE = False


@dataclass
class CompressionBenchmark:
    """Comprehensive benchmarking results for bulk compression operations."""
    total_groups: int
    total_processing_time_seconds: float
    peak_memory_usage_mb: float
    total_original_size_mb: float
    total_compressed_size_mb: float
    overall_compression_ratio: float
    avg_compression_time_per_group_ms: float
    throughput_groups_per_second: float
    memory_efficiency_mb_per_group: float
    performance_targets_met: Dict[str, bool]
    group_metrics: Dict[str, CompressionMetrics] = field(default_factory=dict)
    redundancy_metrics: Dict[str, RedundancyAnalysis] = field(default_factory=dict)


class BulkCompressionEngine:
    """High-performance bulk compression engine with parallel processing."""
    
    def __init__(self, max_workers: int = 4, enable_benchmarking: bool = True):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        self.enable_benchmarking = enable_benchmarking
        
        # Initialize components
        self.redundancy_analyzer = HighPerformanceRedundancyAnalyzer(max_workers)
        self.json_compressor = OptimizedJsonCompressor(max_workers)
        
        # Performance tracking
        self.benchmark_data: List[CompressionBenchmark] = []
    
    async def bulk_compress_all_groups(
        self,
        original_df: pd.DataFrame,
        relationships: Dict[str, Any],
        output_dir: Path,
        compression_strategy: Optional[Dict[str, Any]] = None
    ) -> CompressionBenchmark:
        """Execute bulk compression with comprehensive performance monitoring."""
        
        self.logger.info(f"🚀 Starting bulk compression for {len(relationships)} groups")
        
        # Start comprehensive performance monitoring
        start_time = time.perf_counter()
        tracemalloc.start()
        start_memory = self._get_memory_usage_mb()
        
        try:
            # Phase 1: Parallel redundancy analysis for all groups
            self.logger.info("📊 Phase 1: Parallel redundancy analysis")
            redundancy_results = await self.redundancy_analyzer.batch_analyze_all_groups(
                original_df, relationships
            )
            
            # Phase 2: Optimize compression strategy based on analysis
            if compression_strategy is None:
                compression_strategy = self._optimize_compression_strategy(redundancy_results)
            
            self.logger.info(f"🎯 Using compression strategy: {compression_strategy}")
            
            # Phase 3: Parallel compression execution
            self.logger.info("⚡ Phase 3: Parallel compression execution")
            compression_results = await self._execute_parallel_compression(
                original_df, relationships, redundancy_results, output_dir, compression_strategy
            )
            
            # Phase 4: Calculate comprehensive benchmark metrics
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            peak_memory_mb = peak / 1024 / 1024
            
            benchmark = self._calculate_benchmark_metrics(
                relationships, compression_results, redundancy_results,
                start_time, end_time, peak_memory_mb
            )
            
            # Validate performance targets
            benchmark.performance_targets_met = self._validate_performance_targets(benchmark)
            
            # Store benchmark for analysis
            self.benchmark_data.append(benchmark)
            
            self.logger.info(
                f"✅ Bulk compression complete: "
                f"{benchmark.overall_compression_ratio:.1%} compression, "
                f"{benchmark.total_processing_time_seconds:.2f}s total, "
                f"{benchmark.throughput_groups_per_second:.1f} groups/s"
            )
            
            return benchmark
            
        finally:
            tracemalloc.stop()
    
    def _optimize_compression_strategy(
        self, 
        redundancy_results: Dict[str, RedundancyAnalysis]
    ) -> Dict[str, Any]:
        """Optimize compression strategy based on redundancy analysis."""
        
        if not redundancy_results:
            return {'library': 'json', 'eliminate_redundancy': False}
        
        # Analyze redundancy patterns across all groups
        avg_compression_potential = float(np.mean([r.compression_ratio for r in redundancy_results.values()]))
        total_blank_columns = len(set().union(*[r.blank_columns for r in redundancy_results.values()]))
        total_redundant_columns = len(set().union(*[r.redundant_columns for r in redundancy_results.values()]))
        
        strategy = {
            'library': 'orjson' if ORJSON_AVAILABLE and avg_compression_potential > 0.3 else 'ujson' if UJSON_AVAILABLE else 'json',
            'eliminate_blank_columns': total_blank_columns > 10,  # If >10 blank columns globally
            'extract_parent_data': total_redundant_columns > 5,   # If >5 redundant columns globally
            'use_dictionary_compression': avg_compression_potential > 0.5,
            'enable_parallel_io': len(redundancy_results) > 2,
            'chunk_processing': any(r.total_columns > 100 for r in redundancy_results.values()) if redundancy_results else False  # Large group optimization
        }
        
        return strategy
    
    async def _execute_parallel_compression(
        self,
        original_df: pd.DataFrame,
        relationships: Dict[str, Any],
        redundancy_results: Dict[str, RedundancyAnalysis],
        output_dir: Path,
        strategy: Dict[str, Any]
    ) -> Dict[str, CompressionMetrics]:
        """Execute compression for all groups in parallel."""
        
        # Prepare parallel compression tasks (parent folders already exist from CSV export)
        compression_tasks = []
        
        for parent_sku, relationship in relationships.items():
            if parent_sku not in redundancy_results:
                continue
                
            task = self._compress_single_group_async(
                original_df,
                parent_sku,
                relationship,
                redundancy_results[parent_sku],
                output_dir,  # Use main output_dir (contains parent folders)
                strategy
            )
            compression_tasks.append((parent_sku, task))
        
        # Execute all compressions with controlled concurrency
        results = {}
        
        # Use semaphore to control concurrent I/O operations
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def run_with_semaphore(parent_sku: str, task):
            async with semaphore:
                return await task
        
        # Execute with controlled concurrency
        bounded_tasks = [
            run_with_semaphore(parent_sku, task) 
            for parent_sku, task in compression_tasks
        ]
        
        completed_results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
        
        # Process results
        for (parent_sku, _), result in zip(compression_tasks, completed_results):
            if isinstance(result, Exception):
                self.logger.error(f"Compression failed for {parent_sku}: {result}")
                continue
            results[parent_sku] = result
        
        return results
    
    async def _compress_single_group_async(
        self,
        original_df: pd.DataFrame,
        parent_sku: str,
        relationship: Any,
        redundancy_analysis: RedundancyAnalysis,
        output_dir: Path,
        strategy: Dict[str, Any]
    ) -> CompressionMetrics:
        """Compress single parent group with optimized redundancy elimination."""
        
        # Filter to group data
        child_skus = getattr(relationship, 'child_skus', set())
        all_group_skus = child_skus | {parent_sku}
        mask = original_df['SUPPLIER_PID'].isin(all_group_skus)
        group_df = original_df[mask].copy()
        
        # Generate compressed output path in the parent folder
        parent_folder = output_dir / f"parent_{parent_sku}"
        output_path = parent_folder / "step2_compressed.json"
        
        # Execute compression with redundancy elimination
        compression_metrics = await self.json_compressor.compress_with_redundancy_elimination(
            group_df,
            redundancy_analysis,
            output_path,
            use_best_library=strategy.get('library') != 'json'
        )
        
        return compression_metrics
    
    def _calculate_benchmark_metrics(
        self,
        relationships: Dict[str, Any],
        compression_results: Dict[str, CompressionMetrics],
        redundancy_results: Dict[str, RedundancyAnalysis],
        start_time: float,
        end_time: float,
        peak_memory_mb: float
    ) -> CompressionBenchmark:
        """Calculate comprehensive benchmark metrics."""
        
        total_processing_time = end_time - start_time
        total_groups = len(relationships)
        
        # Aggregate compression metrics
        if compression_results:
            total_compressed_size = sum(m.file_size_bytes for m in compression_results.values())
            avg_compression_time = float(np.mean([m.serialization_time_ms for m in compression_results.values()]))
            overall_compression_ratio = float(np.mean([m.compression_ratio for m in compression_results.values()]))
        else:
            total_compressed_size = 0
            avg_compression_time = 0
            overall_compression_ratio = 0
        
        # Estimate original size (for compression ratio calculation)
        # Rough estimate: 164 columns * avg 50 chars per field * total rows
        total_original_size = 164 * 50 * sum(
            len(getattr(rel, 'child_skus', set())) for rel in relationships.values()
        )
        
        return CompressionBenchmark(
            total_groups=total_groups,
            total_processing_time_seconds=total_processing_time,
            peak_memory_usage_mb=peak_memory_mb,
            total_original_size_mb=total_original_size / 1024 / 1024,
            total_compressed_size_mb=total_compressed_size / 1024 / 1024,
            overall_compression_ratio=overall_compression_ratio,
            avg_compression_time_per_group_ms=avg_compression_time,
            throughput_groups_per_second=total_groups / total_processing_time if total_processing_time > 0 else 0,
            memory_efficiency_mb_per_group=peak_memory_mb / total_groups if total_groups > 0 else 0,
            performance_targets_met={},  # Will be filled by validation
            group_metrics=compression_results,
            redundancy_metrics=redundancy_results
        )
    
    def _validate_performance_targets(self, benchmark: CompressionBenchmark) -> Dict[str, bool]:
        """Validate performance against specified targets."""
        targets = {
            'processing_speed_target': benchmark.avg_compression_time_per_group_ms <= 3000,  # ≤3s per group
            'memory_usage_target': benchmark.peak_memory_usage_mb <= 200,  # ≤200MB peak
            'compression_ratio_target': benchmark.overall_compression_ratio >= 0.5,  # ≥50% compression
            'throughput_target': benchmark.throughput_groups_per_second >= 0.5,  # ≥0.5 groups/s
            'memory_efficiency_target': benchmark.memory_efficiency_mb_per_group <= 35  # ≤35MB per group
        }
        
        targets['overall_pass'] = all(targets.values())
        
        # Log validation results
        if targets['overall_pass']:
            self.logger.info("✅ All performance targets met")
        else:
            failed_targets = [k for k, v in targets.items() if not v and k != 'overall_pass']
            self.logger.warning(f"⚠️ Performance targets not met: {failed_targets}")
        
        return targets
    
    def generate_performance_report(self, benchmark: CompressionBenchmark) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        
        # Performance summary
        performance_summary = {
            'overall_performance': {
                'total_groups_processed': benchmark.total_groups,
                'total_processing_time_seconds': round(benchmark.total_processing_time_seconds, 2),
                'peak_memory_usage_mb': round(benchmark.peak_memory_usage_mb, 1),
                'throughput_groups_per_second': round(benchmark.throughput_groups_per_second, 2),
                'memory_efficiency_mb_per_group': round(benchmark.memory_efficiency_mb_per_group, 1)
            },
            'compression_performance': {
                'overall_compression_ratio': round(benchmark.overall_compression_ratio, 3),
                'total_size_reduction_mb': round(benchmark.total_original_size_mb - benchmark.total_compressed_size_mb, 2),
                'avg_compression_time_per_group_ms': round(benchmark.avg_compression_time_per_group_ms, 1),
                'best_performing_group': self._find_best_performing_group(benchmark),
                'compression_efficiency': 'excellent' if benchmark.overall_compression_ratio > 0.7 else 'good' if benchmark.overall_compression_ratio > 0.5 else 'moderate'
            },
            'performance_targets': benchmark.performance_targets_met,
            'optimization_recommendations': self._generate_optimization_recommendations(benchmark)
        }
        
        return performance_summary
    
    def _find_best_performing_group(self, benchmark: CompressionBenchmark) -> Optional[Dict[str, Any]]:
        """Find the best performing group for analysis."""
        if not benchmark.group_metrics:
            return None
        
        # Best = highest compression ratio with reasonable speed
        best_group = max(
            benchmark.group_metrics.items(),
            key=lambda x: x[1].compression_ratio - (x[1].serialization_time_ms / 10000)  # Penalize slow groups
        )
        
        return {
            'parent_sku': best_group[0],
            'compression_ratio': round(best_group[1].compression_ratio, 3),
            'serialization_time_ms': round(best_group[1].serialization_time_ms, 1),
            'throughput_mb_per_second': round(best_group[1].throughput_mb_per_second, 1)
        }
    
    def _generate_optimization_recommendations(self, benchmark: CompressionBenchmark) -> List[str]:
        """Generate performance optimization recommendations based on benchmark results."""
        recommendations = []
        
        # Processing speed recommendations
        if benchmark.avg_compression_time_per_group_ms > 3000:
            recommendations.append("Consider increasing max_workers for parallel processing")
            recommendations.append("Implement DataFrame chunking for large groups (>100 rows)")
        
        # Memory optimization recommendations  
        if benchmark.peak_memory_usage_mb > 200:
            recommendations.append("Implement streaming processing for large DataFrames")
            recommendations.append("Use memory-mapped file access for reduced RAM usage")
        
        # Compression optimization recommendations
        if benchmark.overall_compression_ratio < 0.5:
            recommendations.append("Enable more aggressive redundancy elimination")
            recommendations.append("Consider LZ4 or gzip compression for JSON output")
        
        # I/O optimization recommendations
        if benchmark.throughput_groups_per_second < 0.5:
            recommendations.append("Implement async I/O for file operations")
            recommendations.append("Use SSD storage for output directory if possible")
        
        # Library optimization recommendations
        if not any('orjson' in str(m.library_name) for m in benchmark.group_metrics.values()):
            recommendations.append("Install orjson library for 2-5x JSON serialization speedup")
        
        return recommendations
    
    async def benchmark_compression_libraries(
        self,
        sample_data: Dict[str, Any],
        output_dir: Path
    ) -> Dict[str, CompressionMetrics]:
        """Benchmark JSON compression libraries with sample data."""
        self.logger.info("🔬 Benchmarking JSON compression libraries...")
        
        benchmark_dir = output_dir / "library_benchmarks"
        benchmark_dir.mkdir(exist_ok=True)
        
        try:
            # Run library benchmark
            library_results = await self.json_compressor.benchmark_json_libraries(
                sample_data, benchmark_dir, iterations=5
            )
            
            # Log results summary
            best_lib = min(library_results.items(), key=lambda x: x[1].serialization_time_ms)
            self.logger.info(
                f"🏆 Library benchmark winner: {best_lib[0]} "
                f"({best_lib[1].serialization_time_ms:.1f}ms, "
                f"{best_lib[1].throughput_mb_per_second:.1f} MB/s)"
            )
            
            return library_results
            
        finally:
            # Clean up benchmark directory
            import shutil
            if benchmark_dir.exists():
                shutil.rmtree(benchmark_dir)
    
    async def run_comprehensive_benchmark(
        self,
        original_df: pd.DataFrame,
        relationships: Dict[str, Any],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark including library comparison and bulk processing."""
        
        self.logger.info("🎯 Starting comprehensive compression benchmark...")
        
        # Create sample data for library benchmarking
        sample_group = list(relationships.keys())[0] if relationships else None
        if sample_group:
            sample_relationship = relationships[sample_group]
            child_skus = getattr(sample_relationship, 'child_skus', set())
            sample_mask = original_df['SUPPLIER_PID'].isin(child_skus | {sample_group})
            sample_df = original_df[sample_mask]
            sample_data = sample_df.to_dict('records')
        else:
            sample_data = {'sample': 'data'}
        
        # Phase 1: Library benchmarking
        library_benchmarks = await self.benchmark_compression_libraries(sample_data, output_dir)
        
        # Phase 2: Full bulk compression benchmark
        bulk_benchmark = await self.bulk_compress_all_groups(
            original_df, relationships, output_dir
        )
        
        # Phase 3: Generate comprehensive report
        comprehensive_report = {
            'library_performance': {
                lib: {
                    'serialization_time_ms': metrics.serialization_time_ms,
                    'throughput_mb_per_second': metrics.throughput_mb_per_second,
                    'memory_usage_mb': metrics.memory_usage_mb
                }
                for lib, metrics in library_benchmarks.items()
            },
            'bulk_processing_performance': self.generate_performance_report(bulk_benchmark),
            'recommendations': {
                'optimal_json_library': min(library_benchmarks.items(), key=lambda x: x[1].serialization_time_ms)[0],
                'performance_optimizations': bulk_benchmark.performance_targets_met,
                'scaling_projections': self._calculate_scaling_projections(bulk_benchmark)
            }
        }
        
        return comprehensive_report
    
    def _calculate_scaling_projections(self, benchmark: CompressionBenchmark) -> Dict[str, Any]:
        """Calculate performance projections for larger datasets."""
        
        # Linear scaling projections (conservative estimates)
        current_groups = benchmark.total_groups
        current_time = benchmark.total_processing_time_seconds
        current_memory = benchmark.peak_memory_usage_mb
        
        projections = {}
        
        for scale_factor in [2, 5, 10]:
            projected_groups = current_groups * scale_factor
            projected_time = current_time * scale_factor * 1.1  # 10% overhead for coordination
            projected_memory = current_memory * scale_factor * 0.7  # Memory reuse efficiency
            
            projections[f'{scale_factor}x_scale'] = {
                'projected_groups': projected_groups,
                'projected_time_seconds': round(projected_time, 1),
                'projected_memory_mb': round(projected_memory, 1),
                'meets_targets': projected_time <= projected_groups * 3.0 and projected_memory <= 200
            }
        
        return projections
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            current, _ = tracemalloc.get_traced_memory()
            return current / 1024 / 1024
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'redundancy_analyzer'):
            del self.redundancy_analyzer
        if hasattr(self, 'json_compressor'):
            del self.json_compressor