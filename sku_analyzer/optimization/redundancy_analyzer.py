"""High-performance redundancy detection and analysis for CSV compression."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any, Optional
import time

import pandas as pd
import numpy as np


@dataclass
class RedundancyAnalysis:
    """Results of redundancy analysis for a parent group."""
    parent_sku: str
    total_columns: int
    blank_columns: Set[str]
    redundant_columns: Set[str]  # Columns with identical values across all rows
    low_variance_columns: Set[str]  # Columns with <5% unique values
    parent_level_data: Dict[str, Any]  # Data that's identical across all children
    compression_ratio: float
    analysis_duration_ms: float
    memory_usage_mb: float


class HighPerformanceRedundancyAnalyzer:
    """Vectorized redundancy detection with memory-efficient processing."""
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def analyze_group_redundancy(
        self, 
        df: pd.DataFrame, 
        parent_sku: str,
        chunk_size: int = 32
    ) -> RedundancyAnalysis:
        """Analyze redundancy for a single parent group with vectorized operations."""
        start_time = time.perf_counter()
        import tracemalloc
        tracemalloc.start()
        
        try:
            # Vectorized blank column detection (most critical optimization)
            blank_columns = self._detect_blank_columns_vectorized(df)
            
            # Vectorized redundancy detection for remaining columns
            non_blank_cols = [col for col in df.columns if col not in blank_columns]
            redundant_columns = self._detect_redundant_columns_vectorized(df[non_blank_cols])
            
            # Low variance analysis (for compression opportunities)
            low_variance_columns = self._detect_low_variance_columns(
                df[non_blank_cols], variance_threshold=0.05
            )
            
            # Extract parent-level data (data common across all children)
            parent_level_data = self._extract_parent_level_data(
                df, redundant_columns, parent_sku
            )
            
            # Calculate compression ratio estimation
            compression_ratio = self._estimate_compression_ratio(
                df, blank_columns, redundant_columns, low_variance_columns
            )
            
            # Performance metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            current, peak = tracemalloc.get_traced_memory()
            memory_mb = peak / 1024 / 1024
            
            return RedundancyAnalysis(
                parent_sku=parent_sku,
                total_columns=len(df.columns),
                blank_columns=blank_columns,
                redundant_columns=redundant_columns,
                low_variance_columns=low_variance_columns,
                parent_level_data=parent_level_data,
                compression_ratio=compression_ratio,
                analysis_duration_ms=duration_ms,
                memory_usage_mb=memory_mb
            )
            
        finally:
            tracemalloc.stop()
    
    def _detect_blank_columns_vectorized(self, df: pd.DataFrame) -> Set[str]:
        """Ultra-fast vectorized blank column detection."""
        # Use pandas vectorized operations for maximum speed
        # This is ~10x faster than iterating through columns
        
        # Method 1: isna() + all() - fastest for numeric data
        blank_mask = df.isna().all()
        
        # Method 2: Check empty strings for string columns
        string_columns = df.select_dtypes(include=['object', 'string']).columns
        for col in string_columns:
            if col not in blank_mask.index:
                continue
            if not blank_mask[col]:  # Not already detected as blank
                # Check if all values are empty strings or whitespace
                is_empty = (df[col].fillna('').astype(str).str.strip() == '').all()
                blank_mask[col] = blank_mask[col] or is_empty
        
        return set(blank_mask[blank_mask].index)
    
    def _detect_redundant_columns_vectorized(self, df: pd.DataFrame) -> Set[str]:
        """Vectorized detection of columns with identical values across all rows."""
        redundant_columns = set()
        
        # Vectorized approach: nunique() == 1 means all values are identical
        unique_counts = df.nunique()
        
        # Columns with exactly 1 unique value (ignoring NaN)
        redundant_mask = unique_counts == 1
        redundant_columns.update(unique_counts[redundant_mask].index)
        
        # Special case: columns with only NaN values (already handled in blank detection)
        # But we need to check columns with 1 unique non-NaN value
        for col in df.columns:
            if col in redundant_columns:
                continue
            non_null_unique = df[col].dropna().nunique()
            if non_null_unique <= 1:
                redundant_columns.add(col)
        
        return redundant_columns
    
    def _detect_low_variance_columns(
        self, 
        df: pd.DataFrame, 
        variance_threshold: float = 0.05
    ) -> Set[str]:
        """Detect columns with low variance for compression opportunities."""
        low_variance_columns = set()
        
        # Calculate unique value ratio for each column
        for col in df.columns:
            if len(df) == 0:
                continue
                
            # Skip columns already identified as redundant or blank
            unique_count = df[col].nunique()
            total_count = len(df)
            
            if unique_count > 1:  # Not redundant
                variance_ratio = unique_count / total_count
                if variance_ratio <= variance_threshold:
                    low_variance_columns.add(col)
        
        return low_variance_columns
    
    def _extract_parent_level_data(
        self, 
        df: pd.DataFrame, 
        redundant_columns: Set[str],
        parent_sku: str
    ) -> Dict[str, Any]:
        """Extract data that's common across all children in a parent group."""
        parent_data = {}
        
        # For redundant columns, extract the common value
        for col in redundant_columns:
            if col in df.columns and len(df) > 0:
                # Get the first non-null value (since all values are identical)
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    value = non_null_values.iloc[0]
                    # Convert numpy/pandas types to native Python types for JSON
                    parent_data[col] = self._convert_to_python_type(value)
                else:
                    parent_data[col] = None
        
        # Add parent metadata
        parent_data['_parent_sku'] = parent_sku
        parent_data['_child_count'] = len(df)
        parent_data['_analysis_timestamp'] = time.time()
        
        return parent_data
    
    def _convert_to_python_type(self, value):
        """Convert pandas/numpy types to native Python types for JSON serialization."""
        import numpy as np
        
        if pd.isna(value):
            return None
        elif isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif hasattr(value, 'item'):  # numpy scalar
            return value.item()
        elif isinstance(value, pd.Timestamp):
            return value.isoformat()
        else:
            return str(value) if pd.notna(value) else None
    
    def _estimate_compression_ratio(
        self,
        df: pd.DataFrame,
        blank_columns: Set[str],
        redundant_columns: Set[str],
        low_variance_columns: Set[str]
    ) -> float:
        """Estimate compression ratio based on redundancy analysis."""
        if len(df.columns) == 0:
            return 0.0
        
        total_columns = len(df.columns)
        
        # Calculate data reduction potential
        blank_reduction = len(blank_columns) / total_columns
        redundant_reduction = len(redundant_columns) / total_columns
        low_variance_reduction = len(low_variance_columns) * 0.3 / total_columns  # 30% compression for low variance
        
        # Estimate overall compression (conservative)
        total_reduction = blank_reduction + redundant_reduction + low_variance_reduction
        compression_ratio = min(total_reduction, 0.85)  # Cap at 85% to be conservative
        
        return compression_ratio
    
    async def batch_analyze_all_groups(
        self, 
        original_df: pd.DataFrame,
        relationships: Dict[str, Any]
    ) -> Dict[str, RedundancyAnalysis]:
        """Parallel redundancy analysis for all parent groups."""
        self.logger.info(f"Starting batch redundancy analysis for {len(relationships)} groups")
        
        # Prepare parallel analysis tasks
        analysis_tasks = []
        
        for parent_sku, relationship in relationships.items():
            # Filter DataFrame for this parent group
            child_skus = getattr(relationship, 'child_skus', set())
            all_group_skus = child_skus | {parent_sku}
            
            # Create task for parallel execution
            task = self._analyze_group_async(
                original_df, parent_sku, all_group_skus
            )
            analysis_tasks.append((parent_sku, task))
        
        # Execute all analyses in parallel
        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in analysis_tasks], 
            return_exceptions=True
        )
        
        # Process results
        for (parent_sku, _), result in zip(analysis_tasks, completed_tasks):
            if isinstance(result, Exception):
                self.logger.error(f"Redundancy analysis failed for {parent_sku}: {result}")
                continue
            results[parent_sku] = result
        
        # Log summary statistics
        if results:
            avg_compression = float(np.mean([r.compression_ratio for r in results.values()]))
            avg_duration = float(np.mean([r.analysis_duration_ms for r in results.values()]))
            
            self.logger.info(
                f"✅ Redundancy analysis complete: "
                f"avg compression {avg_compression:.1%}, "
                f"avg duration {avg_duration:.1f}ms"
            )
        
        return results
    
    async def _analyze_group_async(
        self, 
        original_df: pd.DataFrame, 
        parent_sku: str, 
        group_skus: Set[str]
    ) -> RedundancyAnalysis:
        """Async wrapper for group redundancy analysis."""
        # Filter to group data
        mask = original_df['SUPPLIER_PID'].isin(group_skus)
        group_df = original_df[mask].copy()
        
        # Run analysis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._run_sync_analysis,
            group_df,
            parent_sku
        )
        
        return result
    
    def _run_sync_analysis(
        self, 
        group_df: pd.DataFrame, 
        parent_sku: str
    ) -> RedundancyAnalysis:
        """Synchronous redundancy analysis for thread pool execution."""
        # This runs in thread pool, so we can use sync operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                self.analyze_group_redundancy(group_df, parent_sku)
            )
            return result
        finally:
            loop.close()
    
    def get_compression_summary(
        self, 
        analyses: Dict[str, RedundancyAnalysis]
    ) -> Dict[str, Any]:
        """Generate compression strategy summary from redundancy analyses."""
        if not analyses:
            return {'message': 'No redundancy analysis data available'}
        
        # Aggregate statistics
        total_groups = len(analyses)
        avg_compression = float(np.mean([a.compression_ratio for a in analyses.values()]))
        total_blank_cols = len(set().union(*[a.blank_columns for a in analyses.values()]))
        total_redundant_cols = len(set().union(*[a.redundant_columns for a in analyses.values()]))
        
        # Performance metrics
        avg_analysis_time = float(np.mean([a.analysis_duration_ms for a in analyses.values()]))
        max_memory_usage = max([a.memory_usage_mb for a in analyses.values()])
        
        # Best compression opportunities
        best_group = max(analyses.items(), key=lambda x: x[1].compression_ratio)
        
        return {
            'summary': {
                'total_groups_analyzed': total_groups,
                'avg_compression_ratio': round(avg_compression, 3),
                'total_blank_columns': total_blank_cols,
                'total_redundant_columns': total_redundant_cols,
                'best_compression_group': best_group[0],
                'best_compression_ratio': round(best_group[1].compression_ratio, 3)
            },
            'performance': {
                'avg_analysis_time_ms': round(avg_analysis_time, 1),
                'max_memory_usage_mb': round(max_memory_usage, 1),
                'total_analysis_time_ms': round(sum(a.analysis_duration_ms for a in analyses.values()), 1)
            },
            'compression_strategy': {
                'eliminate_blank_columns': total_blank_cols > 0,
                'extract_parent_data': total_redundant_cols > 0,
                'use_dictionary_compression': avg_compression > 0.3,
                'recommended_json_library': 'orjson' if avg_compression > 0.5 else 'ujson'
            }
        }
    
    def __del__(self):
        """Clean up thread pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)