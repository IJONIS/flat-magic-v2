"""Comprehensive performance benchmarking suite for CSV compression engine."""

import asyncio
import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

from .bulk_processor import BulkCompressionEngine, CompressionBenchmark
from .redundancy_analyzer import HighPerformanceRedundancyAnalyzer
from .json_compressor import OptimizedJsonCompressor


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    test_name: str
    dataset_info: Dict[str, Any]
    redundancy_performance: Dict[str, Any]
    compression_performance: Dict[str, Any]
    library_comparison: Dict[str, Any]
    memory_analysis: Dict[str, Any]
    scaling_projections: Dict[str, Any]
    performance_targets_met: Dict[str, bool]
    recommendations: List[str]
    benchmark_timestamp: str


class CompressionPerformanceBenchmark:
    """Comprehensive performance benchmarking for CSV compression operations."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.bulk_engine = BulkCompressionEngine(max_workers=4)
        self.redundancy_analyzer = HighPerformanceRedundancyAnalyzer(max_workers=4)
        self.json_compressor = OptimizedJsonCompressor(max_workers=4)
    
    async def run_full_benchmark_suite(
        self,
        original_df: pd.DataFrame,
        relationships: Dict[str, Any],
        test_name: str = "compression_benchmark"
    ) -> BenchmarkSuite:
        """Execute comprehensive benchmark suite with all performance tests."""
        
        self.logger.info(f"🎯 Starting comprehensive benchmark suite: {test_name}")
        benchmark_start = time.time()
        
        # Dataset analysis
        dataset_info = self._analyze_dataset_characteristics(original_df, relationships)
        self.logger.info(f"📊 Dataset: {dataset_info['total_rows']} rows, {dataset_info['total_columns']} cols, {dataset_info['parent_groups']} groups")
        
        # Test 1: Redundancy Analysis Performance
        self.logger.info("🔍 Test 1: Redundancy analysis performance")
        redundancy_performance = await self._benchmark_redundancy_analysis(
            original_df, relationships
        )
        
        # Test 2: JSON Library Comparison
        self.logger.info("📚 Test 2: JSON library performance comparison")
        library_comparison = await self._benchmark_json_libraries(
            original_df, relationships
        )
        
        # Test 3: Full Compression Pipeline
        self.logger.info("⚡ Test 3: Full compression pipeline")
        compression_performance = await self._benchmark_compression_pipeline(
            original_df, relationships
        )
        
        # Test 4: Memory Analysis
        self.logger.info("🧠 Test 4: Memory usage analysis")
        memory_analysis = await self._benchmark_memory_usage(
            original_df, relationships
        )
        
        # Test 5: Scaling Projections
        self.logger.info("📈 Test 5: Performance scaling projections")
        scaling_projections = self._calculate_scaling_projections(
            compression_performance, dataset_info
        )
        
        # Validate against performance targets
        performance_targets = self._validate_all_targets(
            redundancy_performance,
            compression_performance,
            memory_analysis
        )
        
        # Generate optimization recommendations
        recommendations = self._generate_comprehensive_recommendations(
            redundancy_performance,
            compression_performance,
            library_comparison,
            memory_analysis,
            performance_targets
        )
        
        # Create final benchmark suite
        benchmark_suite = BenchmarkSuite(
            test_name=test_name,
            dataset_info=dataset_info,
            redundancy_performance=redundancy_performance,
            compression_performance=compression_performance,
            library_comparison=library_comparison,
            memory_analysis=memory_analysis,
            scaling_projections=scaling_projections,
            performance_targets_met=performance_targets,
            recommendations=recommendations,
            benchmark_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Save comprehensive report
        await self._save_benchmark_report(benchmark_suite)
        
        total_time = time.time() - benchmark_start
        self.logger.info(f"✅ Comprehensive benchmark completed in {total_time:.2f}s")
        
        return benchmark_suite
    
    def _analyze_dataset_characteristics(
        self, 
        original_df: pd.DataFrame, 
        relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze dataset characteristics for benchmark context."""
        
        group_sizes = []
        for parent_sku, relationship in relationships.items():
            child_count = len(getattr(relationship, 'child_skus', set()))
            group_sizes.append(child_count)
        
        return {
            'total_rows': len(original_df),
            'total_columns': len(original_df.columns),
            'parent_groups': len(relationships),
            'avg_children_per_group': float(np.mean(group_sizes)) if group_sizes else 0,
            'max_children_per_group': max(group_sizes) if group_sizes else 0,
            'min_children_per_group': min(group_sizes) if group_sizes else 0,
            'estimated_csv_size_mb': len(original_df.to_csv(index=False).encode('utf-8')) / 1024 / 1024
        }
    
    async def _benchmark_redundancy_analysis(
        self, 
        original_df: pd.DataFrame, 
        relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark redundancy analysis performance."""
        
        start_time = time.perf_counter()
        tracemalloc.start()
        
        # Execute redundancy analysis
        redundancy_results = await self.redundancy_analyzer.batch_analyze_all_groups(
            original_df, relationships
        )
        
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        
        # Calculate metrics
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_group_ms = total_time_ms / len(relationships) if relationships else 0
        memory_usage_mb = peak / 1024 / 1024
        
        # Analyze redundancy patterns
        if redundancy_results:
            avg_compression_potential = float(np.mean([r.compression_ratio for r in redundancy_results.values()]))
            total_blank_cols = len(set().union(*[r.blank_columns for r in redundancy_results.values()]))
            total_redundant_cols = len(set().union(*[r.redundant_columns for r in redundancy_results.values()]))
        else:
            avg_compression_potential = 0
            total_blank_cols = 0
            total_redundant_cols = 0
        
        tracemalloc.stop()
        
        return {
            'total_analysis_time_ms': round(total_time_ms, 1),
            'avg_time_per_group_ms': round(avg_time_per_group_ms, 1),
            'memory_usage_mb': round(memory_usage_mb, 1),
            'redundancy_patterns': {
                'avg_compression_potential': round(avg_compression_potential, 3),
                'total_blank_columns': total_blank_cols,
                'total_redundant_columns': total_redundant_cols,
                'groups_with_high_compression': sum(1 for r in redundancy_results.values() if r.compression_ratio > 0.6)
            },
            'performance_efficiency': {
                'meets_speed_target': avg_time_per_group_ms <= 100,  # ≤100ms per group for analysis
                'meets_memory_target': memory_usage_mb <= 50,  # ≤50MB for analysis
                'redundancy_detection_efficiency': 'excellent' if avg_time_per_group_ms < 50 else 'good' if avg_time_per_group_ms < 100 else 'needs_optimization'
            }
        }
    
    async def _benchmark_json_libraries(
        self, 
        original_df: pd.DataFrame, 
        relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark JSON serialization library performance."""
        
        # Create representative sample data
        if relationships:
            sample_parent = list(relationships.keys())[0]
            sample_relationship = relationships[sample_parent]
            child_skus = getattr(sample_relationship, 'child_skus', set())
            sample_mask = original_df['SUPPLIER_PID'].isin(child_skus | {sample_parent})
            sample_df = original_df[sample_mask]
            sample_data = {
                'parent_data': {'sku': sample_parent, 'child_count': len(child_skus)},
                'data_rows': sample_df.to_dict('records')
            }
        else:
            sample_data = {'test': 'data'}
        
        # Run library benchmark
        library_results = await self.json_compressor.benchmark_json_libraries(
            sample_data, self.output_dir
        )
        
        # Format results for analysis
        formatted_results = {}
        for lib, metrics in library_results.items():
            formatted_results[lib] = {
                'serialization_time_ms': round(metrics.serialization_time_ms, 2),
                'throughput_mb_per_second': round(metrics.throughput_mb_per_second, 1),
                'memory_usage_mb': round(metrics.memory_usage_mb, 1),
                'file_size_bytes': metrics.file_size_bytes,
                'relative_performance': 'fastest' if metrics == min(library_results.values(), key=lambda x: x.serialization_time_ms) else 'slower'
            }
        
        return {
            'library_results': formatted_results,
            'recommended_library': min(library_results.items(), key=lambda x: x[1].serialization_time_ms)[0],
            'performance_spread': {
                'fastest_time_ms': min(m.serialization_time_ms for m in library_results.values()),
                'slowest_time_ms': max(m.serialization_time_ms for m in library_results.values()),
                'speed_improvement_factor': max(m.serialization_time_ms for m in library_results.values()) / min(m.serialization_time_ms for m in library_results.values())
            }
        }
    
    async def _benchmark_compression_pipeline(
        self, 
        original_df: pd.DataFrame, 
        relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark complete compression pipeline performance."""
        
        # Run full compression benchmark
        bulk_benchmark = await self.bulk_engine.bulk_compress_all_groups(
            original_df, relationships, self.output_dir
        )
        
        # Generate performance report
        performance_report = self.bulk_engine.generate_performance_report(bulk_benchmark)
        
        return {
            'pipeline_metrics': {
                'total_processing_time_seconds': bulk_benchmark.total_processing_time_seconds,
                'avg_compression_time_per_group_ms': bulk_benchmark.avg_compression_time_per_group_ms,
                'throughput_groups_per_second': bulk_benchmark.throughput_groups_per_second,
                'overall_compression_ratio': bulk_benchmark.overall_compression_ratio,
                'memory_efficiency_mb_per_group': bulk_benchmark.memory_efficiency_mb_per_group
            },
            'performance_analysis': performance_report,
            'target_validation': bulk_benchmark.performance_targets_met
        }
    
    async def _benchmark_memory_usage(
        self, 
        original_df: pd.DataFrame, 
        relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detailed memory usage analysis during compression operations."""
        
        self.logger.info("🧠 Analyzing memory usage patterns...")
        
        # Memory tracking for different operations
        memory_profile = {}
        
        # Test 1: DataFrame loading and optimization
        tracemalloc.start()
        start_memory = self._get_memory_usage_mb()
        
        # Simulate optimized DataFrame loading
        optimized_df = original_df.copy()
        optimized_df['SUPPLIER_PID'] = optimized_df['SUPPLIER_PID'].astype('category')
        
        loading_memory = self._get_memory_usage_mb()
        memory_profile['dataframe_optimization'] = {
            'memory_increase_mb': round(loading_memory - start_memory, 1),
            'optimization_overhead': 'low' if (loading_memory - start_memory) < 10 else 'moderate'
        }
        
        # Test 2: Single group processing memory
        if relationships:
            sample_parent = list(relationships.keys())[0]
            sample_relationship = relationships[sample_parent]
            child_skus = getattr(sample_relationship, 'child_skus', set())
            
            before_group = self._get_memory_usage_mb()
            
            # Process single group
            mask = optimized_df['SUPPLIER_PID'].isin(child_skus | {sample_parent})
            group_df = optimized_df[mask].copy()
            
            # Simulate redundancy analysis
            blank_cols = group_df.isna().all()
            redundant_analysis = self._get_memory_usage_mb()
            
            memory_profile['single_group_processing'] = {
                'base_memory_mb': round(before_group, 1),
                'group_processing_memory_mb': round(redundant_analysis, 1),
                'memory_per_row_kb': round((redundant_analysis - before_group) * 1024 / len(group_df), 2) if len(group_df) > 0 else 0,
                'memory_efficiency': 'excellent' if (redundant_analysis - before_group) < 5 else 'good' if (redundant_analysis - before_group) < 15 else 'needs_optimization'
            }
        
        # Test 3: Parallel processing memory scaling
        current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / 1024 / 1024
        
        memory_profile['parallel_processing'] = {
            'peak_memory_mb': round(peak_memory_mb, 1),
            'parallel_overhead': 'low' if peak_memory_mb < 100 else 'moderate' if peak_memory_mb < 200 else 'high',
            'memory_scaling_factor': round(peak_memory_mb / len(relationships), 1) if relationships else 0
        }
        
        tracemalloc.stop()
        
        return {
            'memory_profile': memory_profile,
            'memory_optimization_opportunities': self._identify_memory_optimizations(memory_profile),
            'memory_target_validation': {
                'meets_200mb_target': peak_memory_mb <= 200,
                'meets_35mb_per_group_target': (peak_memory_mb / len(relationships)) <= 35 if relationships else True,
                'efficient_memory_usage': peak_memory_mb / len(relationships) < 25 if relationships else True
            }
        }
    
    def _identify_memory_optimizations(self, memory_profile: Dict[str, Any]) -> List[str]:
        """Identify specific memory optimization opportunities."""
        optimizations = []
        
        # DataFrame optimization analysis
        df_opt = memory_profile.get('dataframe_optimization', {})
        if df_opt.get('memory_increase_mb', 0) > 10:
            optimizations.append("Implement streaming DataFrame processing for large datasets")
            optimizations.append("Use memory-mapped file access to reduce RAM requirements")
        
        # Single group processing analysis
        group_proc = memory_profile.get('single_group_processing', {})
        memory_per_row = group_proc.get('memory_per_row_kb', 0)
        if memory_per_row > 5:
            optimizations.append("Optimize row-level processing to reduce per-row memory overhead")
            optimizations.append("Implement row chunking for large groups (>100 rows)")
        
        # Parallel processing analysis
        parallel_proc = memory_profile.get('parallel_processing', {})
        if parallel_proc.get('peak_memory_mb', 0) > 150:
            optimizations.append("Reduce max_workers to control memory usage")
            optimizations.append("Implement group-level memory limits with queue management")
        
        return optimizations
    
    def _calculate_scaling_projections(
        self, 
        compression_performance: Dict[str, Any],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate performance projections for larger datasets."""
        
        # Extract current performance metrics
        current_metrics = compression_performance.get('pipeline_metrics', {})
        current_time = current_metrics.get('total_processing_time_seconds', 1)
        current_groups = dataset_info.get('parent_groups', 1)
        
        projections = {}
        
        # Project for different dataset sizes
        for scale_factor in [2, 5, 10, 20]:
            projected_groups = current_groups * scale_factor
            projected_rows = dataset_info.get('total_rows', 0) * scale_factor
            
            # Time scaling (with coordination overhead)
            projected_time = current_time * scale_factor * (1 + 0.1 * scale_factor)  # Increasing overhead
            
            # Memory scaling (sub-linear due to reuse)
            base_memory = 50  # Base overhead
            projected_memory = base_memory + (projected_groups * 8)  # ~8MB per group
            
            projections[f'{scale_factor}x_dataset'] = {
                'projected_groups': projected_groups,
                'projected_rows': projected_rows,
                'projected_time_seconds': round(projected_time, 1),
                'projected_memory_mb': round(projected_memory, 1),
                'meets_3s_per_group_target': (projected_time / projected_groups) <= 3.0,
                'meets_200mb_memory_target': projected_memory <= 200,
                'processing_efficiency': 'excellent' if projected_time <= projected_groups * 2 else 'good' if projected_time <= projected_groups * 3 else 'needs_optimization'
            }
        
        return projections
    
    async def _benchmark_redundancy_analysis(
        self, 
        original_df: pd.DataFrame, 
        relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detailed redundancy analysis performance benchmark."""
        
        start_time = time.perf_counter()
        
        # Execute redundancy analysis with monitoring
        redundancy_results = await self.redundancy_analyzer.batch_analyze_all_groups(
            original_df, relationships
        )
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        
        # Analyze results
        if redundancy_results:
            analysis_times = [r.analysis_duration_ms for r in redundancy_results.values()]
            compression_ratios = [r.compression_ratio for r in redundancy_results.values()]
            
            return {
                'total_redundancy_analysis_time_ms': round(total_time_ms, 1),
                'avg_analysis_time_per_group_ms': round(float(np.mean(analysis_times)), 1),
                'fastest_group_analysis_ms': round(min(analysis_times), 1),
                'slowest_group_analysis_ms': round(max(analysis_times), 1),
                'redundancy_detection_efficiency': {
                    'avg_compression_potential': round(float(np.mean(compression_ratios)), 3),
                    'best_compression_potential': round(max(compression_ratios), 3),
                    'groups_with_high_redundancy': sum(1 for r in compression_ratios if r > 0.6)
                },
                'meets_analysis_targets': {
                    'speed_target': float(np.mean(analysis_times)) <= 100,  # ≤100ms per group
                    'consistency_target': (max(analysis_times) - min(analysis_times)) <= 50  # ≤50ms variation
                }
            }
        else:
            return {'error': 'No redundancy analysis results available'}
    
    async def _benchmark_json_libraries(
        self, 
        original_df: pd.DataFrame, 
        relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark JSON library performance with representative data."""
        
        # Create representative data for benchmarking
        if relationships:
            sample_parent = list(relationships.keys())[0]
            sample_relationship = relationships[sample_parent]
            child_skus = getattr(sample_relationship, 'child_skus', set())
            sample_mask = original_df['SUPPLIER_PID'].isin(child_skus | {sample_parent})
            sample_df = original_df[sample_mask]
            
            benchmark_data = {
                'parent_data': {'sku': sample_parent, 'child_count': len(child_skus)},
                'schema': {'columns': list(sample_df.columns)},
                'data_rows': sample_df.to_dict('records')
            }
        else:
            benchmark_data = {'test': 'data'}
        
        # Run library comparison
        library_results = await self.json_compressor.benchmark_json_libraries(
            benchmark_data, self.output_dir
        )
        
        # Analyze results
        fastest_lib = min(library_results.items(), key=lambda x: x[1].serialization_time_ms)
        slowest_lib = max(library_results.items(), key=lambda x: x[1].serialization_time_ms)
        
        speed_improvement = slowest_lib[1].serialization_time_ms / fastest_lib[1].serialization_time_ms
        
        return {
            'library_performance': {
                lib: {
                    'serialization_time_ms': round(metrics.serialization_time_ms, 2),
                    'throughput_mb_per_second': round(metrics.throughput_mb_per_second, 1),
                    'memory_usage_mb': round(metrics.memory_usage_mb, 1)
                }
                for lib, metrics in library_results.items()
            },
            'performance_comparison': {
                'fastest_library': fastest_lib[0],
                'fastest_time_ms': round(fastest_lib[1].serialization_time_ms, 2),
                'slowest_library': slowest_lib[0],
                'slowest_time_ms': round(slowest_lib[1].serialization_time_ms, 2),
                'speed_improvement_factor': round(speed_improvement, 1)
            },
            'recommendation': {
                'optimal_library': fastest_lib[0],
                'expected_speedup': f"{speed_improvement:.1f}x faster than standard json"
            }
        }
    
    async def _benchmark_compression_pipeline(
        self, 
        original_df: pd.DataFrame, 
        relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark the complete compression pipeline."""
        
        # Execute full compression pipeline
        benchmark = await self.bulk_engine.bulk_compress_all_groups(
            original_df, relationships, self.output_dir
        )
        
        # Generate detailed performance analysis
        performance_report = self.bulk_engine.generate_performance_report(benchmark)
        
        return performance_report
    
    def _validate_all_targets(
        self,
        redundancy_perf: Dict[str, Any],
        compression_perf: Dict[str, Any],
        memory_analysis: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Validate all performance targets across benchmark suite."""
        
        # Extract target validation from each benchmark
        redundancy_targets = redundancy_perf.get('meets_analysis_targets', {})
        compression_targets = compression_perf.get('target_validation', {})
        memory_targets = memory_analysis.get('memory_target_validation', {})
        
        # Comprehensive target validation
        all_targets = {
            'redundancy_analysis_speed': redundancy_targets.get('speed_target', False),
            'redundancy_analysis_consistency': redundancy_targets.get('consistency_target', False),
            'compression_ratio_target': compression_targets.get('compression_ratio_target', False),
            'processing_speed_target': compression_targets.get('processing_speed_target', False),
            'memory_usage_target': memory_targets.get('meets_200mb_target', False),
            'memory_efficiency_target': memory_targets.get('meets_35mb_per_group_target', False),
            'throughput_target': compression_targets.get('throughput_target', False)
        }
        
        all_targets['overall_performance_pass'] = all(all_targets.values())
        
        return all_targets
    
    def _generate_comprehensive_recommendations(
        self,
        redundancy_perf: Dict[str, Any],
        compression_perf: Dict[str, Any], 
        library_comparison: Dict[str, Any],
        memory_analysis: Dict[str, Any],
        performance_targets: Dict[str, bool]
    ) -> List[str]:
        """Generate comprehensive optimization recommendations."""
        
        recommendations = []
        
        # Library optimization recommendations
        fastest_lib = library_comparison.get('recommendation', {}).get('optimal_library')
        if fastest_lib and fastest_lib != 'json':
            recommendations.append(f"Install {fastest_lib} library for {library_comparison.get('performance_comparison', {}).get('speed_improvement_factor', 1):.1f}x JSON serialization speedup")
        
        # Memory optimization recommendations
        memory_opts = memory_analysis.get('memory_optimization_opportunities', [])
        recommendations.extend(memory_opts)
        
        # Performance target recommendations
        if not performance_targets.get('compression_ratio_target', True):
            recommendations.append("Enable more aggressive redundancy elimination for better compression")
        
        if not performance_targets.get('processing_speed_target', True):
            recommendations.append("Increase parallel processing workers or implement DataFrame chunking")
        
        if not performance_targets.get('memory_usage_target', True):
            recommendations.append("Implement streaming processing to reduce peak memory usage")
        
        # Algorithm-specific recommendations
        redundancy_efficiency = redundancy_perf.get('performance_efficiency', {}).get('redundancy_detection_efficiency')
        if redundancy_efficiency == 'needs_optimization':
            recommendations.append("Optimize vectorized operations for redundancy detection")
            recommendations.append("Consider column chunking for 164-column DataFrames")
        
        # I/O optimization recommendations
        throughput = compression_perf.get('pipeline_metrics', {}).get('throughput_groups_per_second', 0)
        if throughput < 1.0:
            recommendations.append("Implement async I/O with proper buffering for file operations")
            recommendations.append("Consider SSD storage for output directory to improve I/O performance")
        
        return recommendations
    
    async def _save_benchmark_report(self, benchmark_suite: BenchmarkSuite) -> None:
        """Save comprehensive benchmark report to file."""
        
        report_file = self.output_dir / f"benchmark_report_{benchmark_suite.test_name}.json"
        
        # Convert dataclass to dict for JSON serialization
        report_data = asdict(benchmark_suite)
        
        # Use best available JSON library for report generation
        try:
            if hasattr(self.json_compressor, 'available_libraries'):
                best_lib = self.json_compressor.available_libraries[0]
                if best_lib == 'orjson':
                    import orjson
                    report_content = orjson.dumps(report_data, option=orjson.OPT_INDENT_2)
                    with open(report_file, 'wb') as f:
                        f.write(report_content)
                else:
                    with open(report_file, 'w') as f:
                        json.dump(report_data, f, indent=2)
            else:
                with open(report_file, 'w') as f:
                    json.dump(report_data, f, indent=2)
                    
            self.logger.info(f"📄 Benchmark report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save benchmark report: {e}")
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            current, _ = tracemalloc.get_traced_memory()
            return current / 1024 / 1024