"""High-performance JSON compression with library benchmarking."""

import asyncio
import json
import logging
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

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
class CompressionMetrics:
    """Performance metrics for JSON compression operations."""
    library_name: str
    serialization_time_ms: float
    file_size_bytes: int
    compression_ratio: float
    memory_usage_mb: float
    throughput_mb_per_second: float


class OptimizedJsonCompressor:
    """High-performance JSON compression with library benchmarking."""
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Available JSON libraries ranked by performance
        self.available_libraries = self._detect_available_libraries()
        
    def _detect_available_libraries(self) -> List[str]:
        """Detect and rank available JSON libraries by performance."""
        libraries = ['json']  # Standard library always available
        
        if ORJSON_AVAILABLE:
            libraries.insert(0, 'orjson')  # Fastest
        if UJSON_AVAILABLE:
            libraries.insert(-1, 'ujson')  # Second fastest
            
        self.logger.info(f"Available JSON libraries: {libraries}")
        return libraries
    
    async def benchmark_json_libraries(
        self, 
        data: Dict[str, Any], 
        output_dir: Path,
        iterations: int = 3
    ) -> Dict[str, CompressionMetrics]:
        """Benchmark all available JSON libraries for compression performance."""
        self.logger.info(f"Benchmarking {len(self.available_libraries)} JSON libraries...")
        
        benchmark_results = {}
        
        for library in self.available_libraries:
            metrics_list = []
            
            # Run multiple iterations for stable measurements
            for i in range(iterations):
                metrics = await self._benchmark_single_library(
                    library, data, output_dir / f"benchmark_{library}_{i}.json"
                )
                metrics_list.append(metrics)
            
            # Calculate average metrics
            avg_metrics = CompressionMetrics(
                library_name=library,
                serialization_time_ms=float(np.mean([m.serialization_time_ms for m in metrics_list])),
                file_size_bytes=int(np.mean([m.file_size_bytes for m in metrics_list])),
                compression_ratio=float(np.mean([m.compression_ratio for m in metrics_list])),
                memory_usage_mb=float(np.mean([m.memory_usage_mb for m in metrics_list])),
                throughput_mb_per_second=float(np.mean([m.throughput_mb_per_second for m in metrics_list]))
            )
            
            benchmark_results[library] = avg_metrics
            
            # Clean up benchmark files
            for i in range(iterations):
                test_file = output_dir / f"benchmark_{library}_{i}.json"
                if test_file.exists():
                    test_file.unlink()
        
        # Log benchmark summary
        best_library = min(benchmark_results.items(), key=lambda x: x[1].serialization_time_ms)
        self.logger.info(
            f"🏆 Best performer: {best_library[0]} "
            f"({best_library[1].serialization_time_ms:.1f}ms, "
            f"{best_library[1].throughput_mb_per_second:.1f} MB/s)"
        )
        
        return benchmark_results
    
    async def _benchmark_single_library(
        self, 
        library: str, 
        data: Dict[str, Any], 
        output_path: Path
    ) -> CompressionMetrics:
        """Benchmark a single JSON library."""
        tracemalloc.start()
        start_memory = self._get_memory_usage_mb()
        
        # Serialize data with timing
        start_time = time.perf_counter()
        
        serialized_data = await self._serialize_with_library(library, data)
        
        end_time = time.perf_counter()
        serialization_time_ms = (end_time - start_time) * 1000
        
        # Write to file and measure size
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._write_compressed_file,
            serialized_data,
            output_path,
            library
        )
        
        # Calculate metrics
        file_size = output_path.stat().st_size
        
        # Estimate compression ratio (compare with uncompressed JSON)
        if library != 'json':
            # Compare with standard JSON size
            std_json = json.dumps(data, separators=(',', ':'))
            baseline_size = len(std_json.encode('utf-8'))
            compression_ratio = 1.0 - (file_size / baseline_size)
        else:
            compression_ratio = 0.0  # Baseline
        
        # Memory metrics
        current, peak = tracemalloc.get_traced_memory()
        memory_usage_mb = peak / 1024 / 1024
        
        # Throughput calculation
        data_size_mb = file_size / 1024 / 1024
        throughput_mb_per_second = data_size_mb / (serialization_time_ms / 1000)
        
        tracemalloc.stop()
        
        return CompressionMetrics(
            library_name=library,
            serialization_time_ms=serialization_time_ms,
            file_size_bytes=file_size,
            compression_ratio=compression_ratio,
            memory_usage_mb=memory_usage_mb,
            throughput_mb_per_second=throughput_mb_per_second
        )
    
    async def _serialize_with_library(
        self, 
        library: str, 
        data: Dict[str, Any]
    ) -> Union[str, bytes]:
        """Serialize data using specified JSON library."""
        if library == 'orjson' and ORJSON_AVAILABLE:
            return orjson.dumps(data, option=orjson.OPT_INDENT_2)
        elif library == 'ujson' and UJSON_AVAILABLE:
            return ujson.dumps(data, indent=2)
        else:
            # Standard library with optimized parameters
            return json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    
    def _write_compressed_file(
        self, 
        serialized_data: Union[str, bytes], 
        output_path: Path,
        library: str
    ) -> None:
        """Write serialized data to file with optimal I/O."""
        if isinstance(serialized_data, bytes):
            # orjson returns bytes
            with open(output_path, 'wb') as f:
                f.write(serialized_data)
        else:
            # ujson and json return strings
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(serialized_data)
    
    async def compress_with_redundancy_elimination(
        self,
        group_df: pd.DataFrame,
        redundancy_analysis: Any,  # RedundancyAnalysis type
        output_path: Path,
        use_best_library: bool = True
    ) -> CompressionMetrics:
        """Compress DataFrame with redundancy elimination for maximum compression."""
        
        # Create compressed data structure (minimal, no metadata)
        compressed_data = {
            'parent_data': redundancy_analysis.parent_level_data,
            'data_rows': []
        }
        
        # Extract only non-redundant columns for each row
        retained_columns = [
            col for col in group_df.columns 
            if col not in redundancy_analysis.blank_columns 
            and col not in redundancy_analysis.redundant_columns
        ]
        
        if retained_columns:
            # Convert to records efficiently
            retained_df = group_df[retained_columns].copy()
            
            # Optimize data types for JSON serialization
            for col in retained_df.columns:
                if retained_df[col].dtype == 'object':
                    # Convert object columns to string, handling NaN
                    retained_df[col] = retained_df[col].fillna('').astype(str)
                elif pd.api.types.is_numeric_dtype(retained_df[col]):
                    # Handle NaN in numeric columns
                    retained_df[col] = retained_df[col].fillna(0)
            
            # Convert to records (list of dicts) - most compact JSON representation
            records = retained_df.to_dict('records')
            # Clean numpy types from records
            compressed_data['data_rows'] = self._clean_records_for_json(records)
        
        # Select best library for compression
        library = self.available_libraries[0] if use_best_library else 'json'
        
        # Perform compression with performance measurement
        start_time = time.perf_counter()
        tracemalloc.start()
        
        serialized_data = await self._serialize_with_library(library, compressed_data)
        
        # Write compressed file
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._write_compressed_file,
            serialized_data,
            output_path,
            library
        )
        
        # Calculate final metrics
        end_time = time.perf_counter()
        serialization_time_ms = (end_time - start_time) * 1000
        
        file_size = output_path.stat().st_size
        current, peak = tracemalloc.get_traced_memory()
        memory_usage_mb = peak / 1024 / 1024
        
        # Calculate compression ratio vs original CSV
        original_csv_size = len(group_df.to_csv(index=False).encode('utf-8'))
        compression_ratio = 1.0 - (file_size / original_csv_size)
        
        # Throughput calculation
        data_size_mb = file_size / 1024 / 1024
        throughput_mb_per_second = data_size_mb / (serialization_time_ms / 1000)
        
        tracemalloc.stop()
        
        return CompressionMetrics(
            library_name=library,
            serialization_time_ms=serialization_time_ms,
            file_size_bytes=file_size,
            compression_ratio=compression_ratio,
            memory_usage_mb=memory_usage_mb,
            throughput_mb_per_second=throughput_mb_per_second
        )
    
    def _clean_records_for_json(self, records: List[Dict]) -> List[Dict]:
        """Clean numpy types from records for JSON serialization."""
        import numpy as np
        
        def clean_value(value):
            if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                return float(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            elif hasattr(value, 'item'):  # numpy scalar
                return value.item()
            elif pd.isna(value):
                return None
            else:
                return value
        
        return [
            {k: clean_value(v) for k, v in record.items()}
            for record in records
        ]
    
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
        """Clean up thread pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)