"""Advanced CSV compression engine with modern pandas techniques and JSON optimization."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


@dataclass
class CompressionMetrics:
    """Metrics tracking for compression operations."""
    total_columns: int = 0
    blank_columns: int = 0
    redundant_columns: int = 0
    parent_level_columns: int = 0
    compression_ratio: float = 0.0
    processing_time_seconds: float = 0.0
    memory_saved_mb: float = 0.0
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0


@dataclass
class CompressionResult:
    """Complete compression result with metadata."""
    parent_data: Dict[str, Any] = field(default_factory=dict)
    child_variations: List[Dict[str, Any]] = field(default_factory=list)
    schema_info: Dict[str, str] = field(default_factory=dict)
    metrics: CompressionMetrics = field(default_factory=CompressionMetrics)
    compression_metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedBlankDetector:
    """Modern pandas-based blank column detection with vectorized operations."""
    
    @staticmethod
    def detect_blank_columns(df: pd.DataFrame, threshold: float = 0.95) -> Set[str]:
        """
        Detect columns that are effectively blank using modern pandas methods.
        
        Args:
            df: DataFrame to analyze
            threshold: Percentage threshold for considering a column blank (0.95 = 95%)
            
        Returns:
            Set of column names that are effectively blank
        """
        blank_columns = set()
        total_rows = len(df)
        
        if total_rows == 0:
            return blank_columns
        
        # Vectorized null detection using pandas 2.x methods
        null_counts = df.isna().sum()
        
        # Check for columns with high null percentage
        for col in df.columns:
            null_ratio = null_counts[col] / total_rows
            
            if null_ratio >= threshold:
                blank_columns.add(col)
                continue
            
            # Check for effectively empty string patterns
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                # Vectorized empty string detection
                empty_mask = (df[col].astype(str).str.strip() == '') | df[col].isna()
                empty_ratio = empty_mask.sum() / total_rows
                
                if empty_ratio >= threshold:
                    blank_columns.add(col)
        
        return blank_columns
    
    @staticmethod
    def analyze_column_patterns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze patterns in columns for advanced blank detection."""
        patterns = {}
        
        for col in df.columns:
            unique_count = df[col].nunique(dropna=False)
            total_count = len(df)
            
            patterns[col] = {
                'unique_ratio': unique_count / total_count if total_count > 0 else 0,
                'null_count': df[col].isna().sum(),
                'dtype': str(df[col].dtype),
                'sample_values': df[col].dropna().head(3).tolist() if not df[col].dropna().empty else []
            }
        
        return patterns


class RedundancyAnalyzer:
    """Analyze data redundancy to separate parent-level from child-level data."""
    
    @staticmethod
    def identify_parent_level_data(df: pd.DataFrame, redundancy_threshold: float = 0.8) -> Tuple[Set[str], Dict[str, Any]]:
        """
        Identify columns that represent parent-level data (constant across children).
        
        Args:
            df: DataFrame to analyze
            redundancy_threshold: Ratio threshold for considering data parent-level
            
        Returns:
            Tuple of (parent_columns, parent_values)
        """
        parent_columns = set()
        parent_values = {}
        total_rows = len(df)
        
        if total_rows <= 1:
            return parent_columns, parent_values
        
        for col in df.columns:
            # Skip if column is mostly null
            non_null_series = df[col].dropna()
            if len(non_null_series) == 0:
                continue
            
            # Check if values are highly redundant (same value across most rows)
            value_counts = non_null_series.value_counts()
            if len(value_counts) == 0:
                continue
                
            most_common_count = value_counts.iloc[0]
            redundancy_ratio = most_common_count / len(non_null_series)
            
            if redundancy_ratio >= redundancy_threshold:
                parent_columns.add(col)
                parent_values[col] = value_counts.index[0]
        
        return parent_columns, parent_values
    
    @staticmethod
    def extract_child_variations(df: pd.DataFrame, parent_columns: Set[str]) -> List[Dict[str, Any]]:
        """Extract child-specific variations by excluding parent-level data."""
        child_columns = [col for col in df.columns if col not in parent_columns]
        
        if not child_columns:
            return []
        
        # Convert to list of dictionaries for child variations
        child_data = df[child_columns].to_dict('records')
        
        # Remove entries that are entirely null/empty for efficiency
        filtered_child_data = []
        for record in child_data:
            # Keep record if it has at least one meaningful value
            if any(pd.notna(value) and str(value).strip() != '' for value in record.values()):
                # Clean up null values for JSON serialization
                cleaned_record = {
                    k: v for k, v in record.items() 
                    if pd.notna(v) and str(v).strip() != ''
                }
                if cleaned_record:  # Only add if there's actual data
                    filtered_child_data.append(cleaned_record)
        
        return filtered_child_data


class OptimizedJsonSerializer:
    """High-performance JSON serialization with type preservation."""
    
    @staticmethod
    def serialize_with_orjson(data: Any) -> bytes:
        """Attempt to use orjson for fastest serialization."""
        def convert_types(obj):
            """Convert numpy/pandas types to JSON serializable types."""
            import numpy as np
            
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            else:
                return obj
        
        # Convert all types first
        converted_data = convert_types(data)
        
        try:
            import orjson
            return orjson.dumps(converted_data, option=orjson.OPT_SORT_KEYS)
        except ImportError:
            # Fallback to standard JSON with deterministic ordering
            return json.dumps(converted_data, ensure_ascii=False, sort_keys=True).encode('utf-8')
    
    @staticmethod
    def serialize_with_ujson(data: Any) -> str:
        """Attempt to use ujson for fast serialization."""
        def convert_types(obj):
            """Convert numpy/pandas types to JSON serializable types."""
            import numpy as np
            
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            else:
                return obj
        
        # Convert all types first
        converted_data = convert_types(data)
        
        try:
            import ujson
            return ujson.dumps(converted_data, ensure_ascii=False, sort_keys=True)
        except ImportError:
            return json.dumps(converted_data, ensure_ascii=False, sort_keys=True)
    
    @staticmethod
    def prepare_for_json(obj: Any) -> Any:
        """Prepare pandas objects for JSON serialization with type preservation."""
        import numpy as np
        
        if pd.isna(obj):
            return None
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat() if pd.notna(obj) else None
        elif pd.api.types.is_datetime64_any_dtype(type(obj)):
            return str(obj) if pd.notna(obj) else None
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj) if not pd.isna(obj) else None
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj) if not pd.isna(obj) else None
        elif isinstance(obj, np.bool_):
            return bool(obj) if not pd.isna(obj) else None
        elif pd.api.types.is_numeric_dtype(type(obj)) and not pd.isna(obj):
            # Convert any remaining numpy numeric types to Python natives
            return obj.item() if hasattr(obj, 'item') else obj
        elif isinstance(obj, pd.Categorical):
            return str(obj) if pd.notna(obj) else None
        elif isinstance(obj, (list, dict)):
            return obj
        else:
            return str(obj) if pd.notna(obj) and str(obj).strip() != '' else None
    
    @staticmethod
    def clean_data_for_json(data: Any) -> Any:
        """Recursively clean data structures for JSON serialization."""
        import numpy as np
        
        if isinstance(data, dict):
            return {k: JsonDataSerializer.clean_data_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [JsonDataSerializer.clean_data_for_json(item) for item in data]
        elif isinstance(data, tuple):
            return [JsonDataSerializer.clean_data_for_json(item) for item in data]
        elif isinstance(data, set):
            return [JsonDataSerializer.clean_data_for_json(item) for item in data]
        elif isinstance(data, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32, np.float16)):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif hasattr(data, 'item'):  # numpy scalar
            return data.item()
        else:
            return JsonDataSerializer.prepare_for_json(data)


class CompressionEngine:
    """Main compression engine with async processing and modern pandas techniques."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logging()
        self.blank_detector = AdvancedBlankDetector()
        self.redundancy_analyzer = RedundancyAnalyzer()
        self.json_serializer = OptimizedJsonSerializer()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup compression-specific logging."""
        logger = logging.getLogger(f"{__name__}.CompressionEngine")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def compress_csv_to_json(
        self, 
        csv_path: Path, 
        output_path: Optional[Path] = None,
        blank_threshold: float = 0.95,
        redundancy_threshold: float = 0.8
    ) -> CompressionResult:
        """
        Main compression method with advanced analysis.
        
        Args:
            csv_path: Path to CSV file to compress
            output_path: Optional output path for JSON file
            blank_threshold: Threshold for blank column detection
            redundancy_threshold: Threshold for parent-level data detection
            
        Returns:
            CompressionResult with complete compression information
        """
        start_time = time.time()
        self.logger.info(f"Starting compression of {csv_path}")
        
        # Load CSV with optimal pandas settings
        df = await self._load_csv_optimized(csv_path)
        original_size = csv_path.stat().st_size
        
        # Step 1: Advanced blank column detection
        self.logger.info("Analyzing blank columns...")
        blank_columns = self.blank_detector.detect_blank_columns(df, blank_threshold)
        
        # Step 2: Column pattern analysis
        column_patterns = self.blank_detector.analyze_column_patterns(df)
        
        # Step 3: Redundancy analysis for parent-level data separation
        self.logger.info("Analyzing data redundancy...")
        parent_columns, parent_values = self.redundancy_analyzer.identify_parent_level_data(
            df, redundancy_threshold
        )
        
        # Step 4: Extract meaningful data (remove blank columns)
        meaningful_columns = [col for col in df.columns if col not in blank_columns]
        meaningful_df = df[meaningful_columns].copy()
        
        # Step 5: Separate parent and child data
        child_variations = self.redundancy_analyzer.extract_child_variations(
            meaningful_df, parent_columns
        )
        
        # Step 6: Prepare final compression result
        result = CompressionResult()
        
        # Parent data (constant across all children)
        result.parent_data = {
            col: self.json_serializer.prepare_for_json(value)
            for col, value in parent_values.items()
            if col in meaningful_columns
        }
        
        # Child variations (unique per child)
        result.child_variations = [
            {k: self.json_serializer.prepare_for_json(v) for k, v in child.items()}
            for child in child_variations
        ]
        
        # Schema information
        result.schema_info = {
            col: str(df[col].dtype) for col in meaningful_columns
        }
        
        # Compression metadata
        result.compression_metadata = {
            'source_file': str(csv_path),
            'compression_timestamp': pd.Timestamp.now().isoformat(),
            'pandas_version': pd.__version__,
            'compression_settings': {
                'blank_threshold': blank_threshold,
                'redundancy_threshold': redundancy_threshold
            },
            'column_analysis': column_patterns
        }
        
        # Calculate metrics
        processing_time = time.time() - start_time
        
        # Serialize to estimate compressed size
        if output_path:
            compressed_data = await self._write_compressed_json(result, output_path)
            compressed_size = output_path.stat().st_size if output_path.exists() else 0
        else:
            compressed_json = self.json_serializer.serialize_with_orjson(result.__dict__)
            compressed_size = len(compressed_json)
        
        result.metrics = CompressionMetrics(
            total_columns=len(df.columns),
            blank_columns=len(blank_columns),
            redundant_columns=len(parent_columns),
            parent_level_columns=len(parent_columns),
            compression_ratio=1 - (compressed_size / original_size) if original_size > 0 else 0,
            processing_time_seconds=processing_time,
            memory_saved_mb=(original_size - compressed_size) / (1024 * 1024),
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size
        )
        
        self.logger.info(
            f"Compression completed: "
            f"{result.metrics.compression_ratio:.2%} reduction, "
            f"{processing_time:.2f}s processing time"
        )
        
        return result
    
    async def _load_csv_optimized(self, csv_path: Path) -> pd.DataFrame:
        """Load CSV with memory-optimized pandas settings."""
        def _load_csv():
            # Use efficient data types and chunked reading for large files
            dtype_hints = {
                'SUPPLIER_PID': 'string',  # pandas 2.x string dtype
                'DESCRIPTION_SHORT': 'string',
                'PRODUCT_STATUS': 'category',
                'MANUFACTURER_NAME': 'category',
            }
            
            try:
                df = pd.read_csv(
                    csv_path,
                    dtype=dtype_hints,
                    na_values=['', 'NULL', 'null', 'N/A', 'n/a', 'NA', 'na'],
                    keep_default_na=True,
                    low_memory=False
                )
                
                # Optimize memory usage by converting appropriate columns to categories
                for col in df.select_dtypes(include=['object']).columns:
                    if col not in dtype_hints:
                        unique_ratio = df[col].nunique() / len(df)
                        if unique_ratio < 0.5:  # Less than 50% unique values
                            df[col] = df[col].astype('category')
                
                return df
                
            except Exception as e:
                logging.error(f"Optimized CSV loading failed: {e}")
                # Fallback to basic loading
                return pd.read_csv(csv_path)
        
        # Run in thread pool for non-blocking operation
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _load_csv)
    
    async def _write_compressed_json(self, result: CompressionResult, output_path: Path) -> Dict:
        """Write compression result to JSON with async I/O."""
        def _write_json():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            output_data = {
                'parent_data': result.parent_data,
                'child_variations': result.child_variations,
                'schema_info': result.schema_info,
                'metrics': {
                    'total_columns': result.metrics.total_columns,
                    'blank_columns': result.metrics.blank_columns,
                    'redundant_columns': result.metrics.redundant_columns,
                    'compression_ratio': result.metrics.compression_ratio,
                    'processing_time_seconds': result.metrics.processing_time_seconds,
                    'memory_saved_mb': result.metrics.memory_saved_mb
                },
                'metadata': result.compression_metadata
            }
            
            # Clean data recursively before serialization
            clean_data = self.json_serializer.clean_data_for_json(output_data)
            
            # Use high-performance JSON serialization
            try:
                json_bytes = self.json_serializer.serialize_with_orjson(clean_data)
                with open(output_path, 'wb') as f:
                    f.write(json_bytes)
            except Exception as e:
                self.logger.warning(f"orjson serialization failed: {e}, falling back to standard JSON")
                # Fallback to standard JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(clean_data, f, ensure_ascii=False, indent=2)
            
            return clean_data
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _write_json)
    
    async def bulk_compress_parent_folders(
        self,
        production_output_dir: Path,
        job_id: str,
        max_concurrent: int = 4
    ) -> Dict[str, CompressionResult]:
        """
        Compress all parent folders in parallel with controlled concurrency.
        
        Args:
            production_output_dir: Base production output directory
            job_id: Job identifier
            max_concurrent: Maximum concurrent compression operations
            
        Returns:
            Dictionary mapping parent folder names to compression results
        """
        job_dir = production_output_dir / job_id
        if not job_dir.exists():
            raise FileNotFoundError(f"Job directory not found: {job_dir}")
        
        # Find all parent folders
        parent_folders = [
            folder for folder in job_dir.iterdir() 
            if folder.is_dir() and folder.name.startswith('parent_')
        ]
        
        if not parent_folders:
            self.logger.warning(f"No parent folders found in {job_dir}")
            return {}
        
        self.logger.info(f"Starting bulk compression of {len(parent_folders)} parent folders")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def compress_single_parent(parent_folder: Path) -> Tuple[str, Optional[CompressionResult]]:
            """Compress a single parent folder with semaphore control."""
            async with semaphore:
                try:
                    csv_path = parent_folder / "data.csv"
                    if not csv_path.exists():
                        self.logger.warning(f"No data.csv found in {parent_folder}")
                        return parent_folder.name, None
                    
                    output_path = parent_folder / "step2_compressed.json"
                    result = await self.compress_csv_to_json(csv_path, output_path)
                    
                    self.logger.info(f"Completed compression for {parent_folder.name}")
                    return parent_folder.name, result
                    
                except Exception as e:
                    self.logger.error(f"Failed to compress {parent_folder.name}: {e}")
                    return parent_folder.name, None
        
        # Execute all compressions concurrently
        tasks = [compress_single_parent(folder) for folder in parent_folders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile results
        compression_results = {}
        successful_compressions = 0
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Compression task failed with exception: {result}")
                continue
            
            parent_name, compression_result = result
            if compression_result:
                compression_results[parent_name] = compression_result
                successful_compressions += 1
        
        self.logger.info(
            f"Bulk compression completed: "
            f"{successful_compressions}/{len(parent_folders)} successful"
        )
        
        return compression_results
    
    async def validate_compression_integrity(
        self, 
        original_csv: Path, 
        compressed_json: Path
    ) -> Dict[str, Any]:
        """
        Validate that compressed data maintains integrity with original CSV.
        
        Args:
            original_csv: Path to original CSV file
            compressed_json: Path to compressed JSON file
            
        Returns:
            Validation report with integrity checks
        """
        self.logger.info("Validating compression integrity...")
        
        try:
            # Load original data
            original_df = await self._load_csv_optimized(original_csv)
            
            # Load compressed data
            with open(compressed_json, 'r', encoding='utf-8') as f:
                compressed_data = json.load(f)
            
            # Reconstruct data from compressed format
            parent_data = compressed_data.get('parent_data', {})
            child_variations = compressed_data.get('child_variations', [])
            
            # Validation checks
            validation_report = {
                'original_row_count': len(original_df),
                'compressed_child_count': len(child_variations),
                'parent_data_keys': len(parent_data),
                'data_integrity_checks': {},
                'validation_passed': True,
                'issues': []
            }
            
            # Check if we can account for all meaningful data
            meaningful_columns = [col for col in original_df.columns 
                                if not original_df[col].isna().all()]
            
            total_compressed_keys = set(parent_data.keys())
            if child_variations:
                for child in child_variations:
                    total_compressed_keys.update(child.keys())
            
            missing_columns = set(meaningful_columns) - total_compressed_keys
            if missing_columns:
                validation_report['issues'].append(
                    f"Missing columns in compression: {missing_columns}"
                )
                validation_report['validation_passed'] = False
            
            validation_report['data_integrity_checks'] = {
                'meaningful_columns_preserved': len(missing_columns) == 0,
                'parent_data_extracted': len(parent_data) > 0,
                'child_variations_extracted': len(child_variations) > 0
            }
            
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {
                'validation_passed': False,
                'error': str(e),
                'issues': [f"Validation process failed: {e}"]
            }
    
    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Integration helper functions for existing pipeline
async def compress_job_data(job_id: str, production_dir: Path = None) -> Dict[str, CompressionResult]:
    """
    Convenience function to compress all data for a specific job.
    
    Args:
        job_id: Job identifier
        production_dir: Production output directory (defaults to 'production_output')
        
    Returns:
        Dictionary of compression results by parent folder
    """
    if production_dir is None:
        production_dir = Path("production_output")
    
    engine = CompressionEngine()
    return await engine.bulk_compress_parent_folders(production_dir, job_id)


def add_compression_to_pipeline():
    """
    Integration point to add compression step to existing analyzer pipeline.
    This function can be called from the main analyzer to add compression capabilities.
    """
    # This will be used for integration with existing pipeline
    pass