"""File splitter for creating parent group CSV files."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from ..models import ProcessingJob, ParentChildRelationship


class FileSplitter:
    """Modern async file splitter with deterministic CSV generation for parent groups."""
    
    def __init__(self):
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure structured logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def load_analysis_results(self, job_id: str) -> Optional[Dict]:
        """Load analysis results from JSON file."""
        analysis_file = Path("production_output") / job_id / f"analysis_{job_id}.json"
        
        if not analysis_file.exists():
            self.logger.error(f"Analysis file not found: {analysis_file}")
            return None
        
        try:
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, analysis_file.read_text)
            return json.loads(content)
        except Exception as e:
            self.logger.error(f"Failed to load analysis results: {e}")
            return None
    
    async def load_original_data(self, input_path: Path) -> Optional[pd.DataFrame]:
        """Asynchronously load original XLSX data."""
        self.logger.info(f"Loading original data from {input_path}")
        
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None, 
                self._load_xlsx_sync, 
                input_path
            )
            
            self.logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load original data: {e}")
            return None
    
    def _load_xlsx_sync(self, file_path: Path) -> pd.DataFrame:
        """Synchronous XLSX loading with pandas 2.x optimizations."""
        # Use pandas 2.x string dtype for better performance
        dtype_mapping = {'SUPPLIER_PID': 'string'}
        
        return pd.read_excel(
            file_path,
            dtype=dtype_mapping,
            engine='openpyxl'
        )
    
    def _create_parent_folder(self, job_output_dir: Path, parent_sku: str) -> Path:
        """Create deterministic folder structure for parent group."""
        parent_folder = job_output_dir / f"parent_{parent_sku}"
        parent_folder.mkdir(parents=True, exist_ok=True)
        return parent_folder
    
    async def _export_parent_group_csv(
        self, 
        parent_sku: str, 
        child_skus: Set[str], 
        original_df: pd.DataFrame, 
        output_folder: Path
    ) -> bool:
        """Export CSV for a single parent group with deterministic formatting."""
        try:
            # Filter data for this parent group
            group_data = original_df[original_df['SUPPLIER_PID'].isin(child_skus)].copy()
            
            if group_data.empty:
                self.logger.warning(f"No data found for parent {parent_sku}")
                return False
            
            # Deterministic sorting by SUPPLIER_PID for consistent output
            group_data = group_data.sort_values('SUPPLIER_PID')
            
            # Reset index to ensure clean CSV output
            group_data = group_data.reset_index(drop=True)
            
            # Create deterministic filename
            csv_file = output_folder / f"{parent_sku}_children.csv"
            
            # Export with consistent formatting
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: group_data.to_csv(
                    csv_file,
                    index=False,
                    encoding='utf-8',
                    lineterminator='\n'  # Consistent line endings
                )
            )
            
            self.logger.info(
                f"Exported {len(group_data)} rows for parent {parent_sku} -> {csv_file}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export parent group {parent_sku}: {e}")
            return False
    
    async def _create_split_metadata(
        self, 
        job_output_dir: Path, 
        split_results: Dict[str, Dict],
        job_id: str
    ) -> None:
        """Create metadata file documenting the split operation."""
        metadata = {
            'job_id': job_id,
            'operation': 'file_split',
            'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
            'parent_groups': {},
            'summary': {
                'total_parent_groups': len(split_results),
                'total_files_created': sum(1 for r in split_results.values() if r['success']),
                'total_rows_exported': sum(r['row_count'] for r in split_results.values())
            }
        }
        
        # Document each parent group
        for parent_sku, result in split_results.items():
            metadata['parent_groups'][parent_sku] = {
                'success': result['success'],
                'row_count': result['row_count'],
                'csv_file': f"parent_{parent_sku}/{parent_sku}_children.csv",
                'child_count': result['child_count']
            }
        
        metadata_file = job_output_dir / f"split_metadata_{job_id}.json"
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: metadata_file.write_text(json.dumps(metadata, indent=2))
        )
        
        self.logger.info(f"Split metadata saved to {metadata_file}")
    
    async def split_by_parent_groups(
        self, 
        job_id: str, 
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Main entry point for splitting files by parent groups.
        
        Args:
            job_id: The job ID to process
            progress_callback: Optional callback for progress tracking
        
        Returns:
            bool: True if splitting was successful
        """
        self.logger.info(f"Starting file split operation for job {job_id}")
        
        # Load analysis results
        analysis_data = await self.load_analysis_results(job_id)
        if not analysis_data:
            return False
        
        # Load original data
        input_path = Path(analysis_data['input_file'])
        original_df = await self.load_original_data(input_path)
        if original_df is None:
            return False
        
        # Prepare output directory
        job_output_dir = Path(analysis_data['output_dir'])
        
        # Process each parent group
        split_results = {}
        total_parents = len(analysis_data['analysis'])
        
        self.logger.info(f"Processing {total_parents} parent groups...")
        
        for i, parent_analysis in enumerate(analysis_data['analysis']):
            parent_sku = parent_analysis['parent_sku']
            child_skus = set(parent_analysis['child_skus'])
            
            # Create folder for this parent
            parent_folder = self._create_parent_folder(job_output_dir, parent_sku)
            
            # Export CSV for this parent group
            success = await self._export_parent_group_csv(
                parent_sku, child_skus, original_df, parent_folder
            )
            
            # Track results
            split_results[parent_sku] = {
                'success': success,
                'row_count': len(original_df[original_df['SUPPLIER_PID'].isin(child_skus)]),
                'child_count': len(child_skus)
            }
            
            # Progress callback
            if progress_callback:
                progress = ((i + 1) / total_parents) * 100
                progress_callback(parent_sku, progress)
        
        # Create metadata for the split operation
        await self._create_split_metadata(job_output_dir, split_results, job_id)
        
        # Summary
        successful_splits = sum(1 for r in split_results.values() if r['success'])
        total_rows_exported = sum(r['row_count'] for r in split_results.values())
        
        self.logger.info(f"Split operation completed:")
        self.logger.info(f"  • {successful_splits}/{total_parents} parent groups processed successfully")
        self.logger.info(f"  • {total_rows_exported} total rows exported")
        self.logger.info(f"  • Results saved to {job_output_dir}")
        
        return successful_splits == total_parents
    
    async def validate_split_integrity(self, job_id: str) -> bool:
        """
        Validate that split operation maintained data integrity.
        
        Args:
            job_id: The job ID to validate
            
        Returns:
            bool: True if validation passes
        """
        self.logger.info(f"Validating split integrity for job {job_id}")
        
        # Load analysis and split metadata
        analysis_data = await self.load_analysis_results(job_id)
        if not analysis_data:
            return False
        
        split_metadata_file = Path("production_output") / job_id / f"split_metadata_{job_id}.json"
        if not split_metadata_file.exists():
            self.logger.error("Split metadata file not found")
            return False
        
        try:
            split_metadata = json.loads(split_metadata_file.read_text())
        except Exception as e:
            self.logger.error(f"Failed to load split metadata: {e}")
            return False
        
        # Validate row counts
        expected_total_rows = analysis_data['summary']['total_child_skus']
        actual_total_rows = split_metadata['summary']['total_rows_exported']
        
        if expected_total_rows != actual_total_rows:
            self.logger.error(
                f"Row count mismatch: expected {expected_total_rows}, got {actual_total_rows}"
            )
            return False
        
        # Validate individual parent groups
        for parent_analysis in analysis_data['analysis']:
            parent_sku = parent_analysis['parent_sku']
            expected_child_count = parent_analysis['child_count']
            
            if parent_sku in split_metadata['parent_groups']:
                actual_row_count = split_metadata['parent_groups'][parent_sku]['row_count']
                if actual_row_count != expected_child_count:
                    self.logger.error(
                        f"Parent {parent_sku}: expected {expected_child_count} rows, got {actual_row_count}"
                    )
                    return False
            else:
                self.logger.error(f"Parent {parent_sku} missing from split metadata")
                return False
        
        self.logger.info("Split integrity validation passed")
        return True