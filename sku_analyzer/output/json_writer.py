"""Optimized JSON output writer with high-performance compression capabilities."""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict

import pandas as pd

from ..models import ProcessingJob, ParentChildRelationship
from .csv_writer import OptimizedCsvWriter, CsvExportProgressTracker
from ..optimization import (
    BulkCompressionEngine, 
    HighPerformanceRedundancyAnalyzer, 
    OptimizedJsonCompressor,
    CompressionPerformanceBenchmark
)

# Try to import high-performance JSON libraries
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


class JsonWriter:
    """Handle JSON output for SKU analysis results with high-performance compression."""
    
    def __init__(self, enable_compression: bool = True, max_workers: int = 4):
        self.logger = logging.getLogger(__name__)
        self.csv_writer = OptimizedCsvWriter(max_workers=max_workers)
        self.enable_compression = enable_compression
        
        # High-performance compression components
        if enable_compression:
            self.compression_engine = BulkCompressionEngine(max_workers=max_workers)
            self.redundancy_analyzer = HighPerformanceRedundancyAnalyzer(max_workers=max_workers)
            self.json_compressor = OptimizedJsonCompressor(max_workers=max_workers)
        
        # Determine optimal JSON library
        self.json_library = self._select_optimal_json_library()
    
    def _select_optimal_json_library(self) -> str:
        """Select the best available JSON library for performance."""
        if ORJSON_AVAILABLE:
            return 'orjson'  # Fastest option
        elif UJSON_AVAILABLE:
            return 'ujson'   # Second fastest
        else:
            return 'json'    # Standard library fallback
    
    async def save_results(
        self, 
        job: ProcessingJob, 
        relationships: Dict[str, ParentChildRelationship],
        original_df: pd.DataFrame,
        export_csv: bool = True,
        enable_compression_benchmark: bool = False
    ) -> None:
        """Save analysis results with optional high-performance compression."""
        self.logger.info(f"Saving results to {job.output_dir}")
        
        # Export CSV splits first if requested
        csv_files = {}
        csv_summary = {}
        
        if export_csv and relationships:
            progress_tracker = CsvExportProgressTracker()
            csv_files = await self.csv_writer.split_and_export_csv(
                job, relationships, original_df, progress_tracker
            )
            csv_summary = await self.csv_writer.get_split_summary(csv_files)
        
        # Build standard analysis metadata
        metadata = await self._build_analysis_metadata(
            job, relationships, original_df, csv_files, csv_summary
        )
        
        # Save standard JSON metadata
        await self._save_json_metadata(metadata, job.output_dir / f"analysis_{job.job_id}.json")
        
        # Optional: High-performance compression analysis and output
        if self.enable_compression and relationships:
            await self._save_compressed_results(
                job, relationships, original_df, enable_compression_benchmark
            )
    
    async def _build_analysis_metadata(
        self,
        job: ProcessingJob,
        relationships: Dict[str, ParentChildRelationship],
        original_df: pd.DataFrame,
        csv_files: Dict[str, Path],
        csv_summary: Dict[str, int]
    ) -> Dict[str, Any]:
        """Build comprehensive analysis metadata."""
        
        # Build complete analysis data for JSON
        analysis_data = []
        for parent_sku, relationship in relationships.items():
            analysis_entry = {
                'parent_sku': parent_sku,
                'child_count': len(relationship.child_skus),
                'child_skus': sorted(list(relationship.child_skus)),
                'pattern_confidence': relationship.pattern_confidence,
                'base_pattern': relationship.metadata.get('base_pattern', ''),
                'pattern_count': relationship.metadata.get('pattern_count', 0),
                'subcategories': relationship.metadata.get('subcategories', []),
                'hierarchy_depth': relationship.metadata.get('hierarchy_depth', 1)
            }
            
            # Add CSV export information if available
            if parent_sku in csv_files:
                analysis_entry['csv_file'] = str(csv_files[parent_sku].relative_to(job.output_dir))
                analysis_entry['csv_row_count'] = csv_summary.get(parent_sku, 0)
            
            analysis_data.append(analysis_entry)
        
        # Sort for deterministic output
        analysis_data = sorted(analysis_data, key=lambda x: (x['parent_sku'], x['child_count']))
        
        # Extract parent SKUs for metadata
        parent_skus = sorted(relationships.keys())
        
        # Calculate summary statistics
        total_child_skus = sum(len(r.child_skus) for r in relationships.values())
        avg_children = total_child_skus / len(relationships) if relationships else 0
        
        metadata = {
            'job_id': job.job_id,
            'input_file': str(job.input_path),
            'output_dir': str(job.output_dir),
            'created_at': job.created_at.isoformat(),
            'status': 'completed',
            'summary': {
                'total_skus': len(original_df),
                'parent_child_groups': len(relationships),
                'total_child_skus': total_child_skus,
                'avg_children_per_parent': round(avg_children, 2),
                'parent_skus': parent_skus
            },
            'csv_export': {
                'enabled': len(csv_files) > 0,
                'files_created': len(csv_files),
                'total_exported_rows': sum(csv_summary.values()) if csv_summary else 0,
                'csv_directory': 'csv_splits' if csv_files else None
            },
            'analysis': analysis_data
        }
        
        return metadata
    
    async def _save_compressed_results(
        self,
        job: ProcessingJob,
        relationships: Dict[str, ParentChildRelationship],
        original_df: pd.DataFrame,
        enable_benchmark: bool = False
    ) -> None:
        """Save compressed results with performance optimization."""
        
        self.logger.info("⚡ Generating high-performance compressed output...")
        
        try:
            if enable_benchmark:
                # Run comprehensive benchmark
                benchmark_tool = CompressionPerformanceBenchmark(job.output_dir / "compression_benchmarks")
                benchmark_results = await benchmark_tool.run_full_benchmark_suite(
                    original_df, relationships, f"job_{job.job_id}"
                )
                
                self.logger.info(
                    f"📊 Compression benchmark completed: "
                    f"{benchmark_results.compression_performance.get('pipeline_metrics', {}).get('overall_compression_ratio', 0):.1%} compression, "
                    f"{len(benchmark_results.recommendations)} optimizations identified"
                )
                
                # Save benchmark summary
                benchmark_summary_file = job.output_dir / "compression_benchmark_summary.json"
                await self._save_json_metadata(asdict(benchmark_results), benchmark_summary_file)
            
            else:
                # Standard compression without full benchmark
                compression_benchmark = await self.compression_engine.bulk_compress_all_groups(
                    original_df, relationships, job.output_dir
                )
                
                # Save compression summary
                compression_summary = {
                    'compression_results': {
                        'total_groups': compression_benchmark.total_groups,
                        'overall_compression_ratio': round(compression_benchmark.overall_compression_ratio, 3),
                        'total_processing_time_seconds': round(compression_benchmark.total_processing_time_seconds, 2),
                        'peak_memory_usage_mb': round(compression_benchmark.peak_memory_usage_mb, 1),
                        'throughput_groups_per_second': round(compression_benchmark.throughput_groups_per_second, 1)
                    },
                    'performance_targets_met': compression_benchmark.performance_targets_met,
                    'recommendations': self._generate_quick_recommendations(compression_benchmark)
                }
                
                compression_summary_file = job.output_dir / "compression_summary.json"
                await self._save_json_metadata(compression_summary, compression_summary_file)
                
                self.logger.info(
                    f"✅ Compression completed: "
                    f"{compression_benchmark.overall_compression_ratio:.1%} compression, "
                    f"{compression_benchmark.throughput_groups_per_second:.1f} groups/s"
                )
        
        except Exception as e:
            self.logger.error(f"Compression optimization failed: {e}")
            self.logger.info("Falling back to standard JSON output")
    
    def _generate_quick_recommendations(self, benchmark: Any) -> List[str]:
        """Generate quick optimization recommendations from compression benchmark."""
        recommendations = []
        
        if benchmark.overall_compression_ratio < 0.5:
            recommendations.append("Enable more aggressive redundancy elimination")
        
        if benchmark.peak_memory_usage_mb > 200:
            recommendations.append("Implement streaming processing for memory optimization")
        
        if benchmark.avg_compression_time_per_group_ms > 3000:
            recommendations.append("Increase parallel processing workers")
        
        if not ORJSON_AVAILABLE:
            recommendations.append("Install orjson library for faster JSON serialization")
        
        return recommendations
    
    async def _save_json_metadata(self, data: Dict[str, Any], output_path: Path) -> None:
        """Save JSON metadata using optimal library and settings."""
        
        try:
            if self.json_library == 'orjson' and ORJSON_AVAILABLE:
                import orjson
                # orjson provides fastest serialization with proper formatting
                json_bytes = orjson.dumps(data, option=orjson.OPT_INDENT_2)
                with open(output_path, 'wb') as f:
                    f.write(json_bytes)
            
            elif self.json_library == 'ujson' and UJSON_AVAILABLE:
                import ujson
                # ujson faster than standard json but returns string
                json_content = ujson.dumps(data, indent=2)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_content)
            
            else:
                # Standard library with optimized parameters
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, separators=(',', ':'), ensure_ascii=False)
            
            self.logger.debug(f"JSON metadata saved using {self.json_library}: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON metadata: {e}")
            # Fallback to standard JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
    
    def get_compression_capabilities(self) -> Dict[str, Any]:
        """Get information about available compression capabilities."""
        return {
            'compression_enabled': self.enable_compression,
            'json_library': self.json_library,
            'available_libraries': {
                'orjson': ORJSON_AVAILABLE,
                'ujson': UJSON_AVAILABLE,
                'json': True
            },
            'performance_features': {
                'redundancy_analysis': self.enable_compression,
                'parallel_compression': self.enable_compression,
                'library_benchmarking': self.enable_compression,
                'memory_optimization': True,
                'async_io': True
            }
        }
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'compression_engine'):
            del self.compression_engine
        if hasattr(self, 'redundancy_analyzer'):
            del self.redundancy_analyzer
        if hasattr(self, 'json_compressor'):
            del self.json_compressor