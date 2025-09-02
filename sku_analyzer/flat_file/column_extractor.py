"""Column extraction module for XLSM template analysis.

This module provides focused functionality for extracting column mappings
from worksheets with performance optimization and validation.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from openpyxl.worksheet.worksheet import Worksheet

from .data_structures import ColumnMapping


class ColumnExtractor:
    """Optimized column mapping extraction with streaming processing."""
    
    # Extraction constants
    BATCH_SIZE_ROWS = 100
    MEMORY_LIMIT_MB = 100
    TECHNICAL_NAME_PATTERN = re.compile(r'^[a-zäöüß][a-zäöüß0-9_]*$')
    
    # Requirement status mapping
    REQUIREMENT_STATUS_MAP = {
        "Erforderlich": "mandatory",
        "erforderlich": "mandatory",
        "ERFORDERLICH": "mandatory",
        "Optional": "optional",
        "optional": "optional",
        "OPTIONAL": "optional",
        "Empfohlen": "recommended",
        "empfohlen": "recommended",
        "EMPFOHLEN": "recommended",
        "Bevorzugt": "recommended",
        "bevorzugt": "recommended",
        "BEVORZUGT": "recommended"
    }
    
    def __init__(self, enable_performance_monitoring: bool = True) -> None:
        """Initialize column extractor.
        
        Args:
            enable_performance_monitoring: Whether to monitor memory usage
        """
        self.logger = logging.getLogger(__name__)
        self.enable_monitoring = enable_performance_monitoring
    
    def extract_column_mappings(
        self, 
        worksheet: Worksheet, 
        header_row: int, 
        tech_col: str, 
        display_col: str,
        req_col: Optional[str] = None
    ) -> List[ColumnMapping]:
        """Extract column mappings with streaming processing.
        
        Args:
            worksheet: Worksheet to extract from
            header_row: Header row number
            tech_col: Technical column letter
            display_col: Display column letter
            req_col: Optional requirement column letter
            
        Returns:
            List of extracted column mappings
        """
        mappings = []
        start_row = header_row + 1
        max_row = worksheet.max_row or start_row
        
        self.logger.debug(f"Streaming extraction from rows {start_row} to {max_row}")
        
        # Process in batches for memory efficiency
        for batch_start in range(start_row, max_row + 1, self.BATCH_SIZE_ROWS):
            batch_end = min(batch_start + self.BATCH_SIZE_ROWS - 1, max_row)
            
            batch_mappings = self._process_row_batch(
                worksheet, batch_start, batch_end, tech_col, display_col, req_col
            )
            mappings.extend(batch_mappings)
            
            # Memory usage check during processing
            if self.enable_monitoring:
                current_memory = self._get_current_memory_mb()
                if current_memory > self.MEMORY_LIMIT_MB:
                    self.logger.warning(
                        f"Memory usage high: {current_memory:.1f}MB at row {batch_end}"
                    )
        
        self.logger.info(f"Extracted {len(mappings)} column mappings using streaming approach")
        return mappings
    
    def _process_row_batch(
        self,
        worksheet: Worksheet,
        start_row: int,
        end_row: int,
        tech_col: str,
        display_col: str,
        req_col: Optional[str] = None
    ) -> List[ColumnMapping]:
        """Process a batch of rows efficiently.
        
        Args:
            worksheet: Worksheet to process
            start_row: Starting row number
            end_row: Ending row number
            tech_col: Technical column letter
            display_col: Display column letter
            req_col: Optional requirement column letter
            
        Returns:
            List of column mappings for this batch
        """
        batch_mappings = []
        
        for row_num in range(start_row, end_row + 1):
            try:
                # Get cell values directly
                tech_cell = worksheet[f"{tech_col}{row_num}"]
                display_cell = worksheet[f"{display_col}{row_num}"]
                
                tech_value = str(tech_cell.value or '').strip()
                display_value = str(display_cell.value or '').strip()
                
                # Extract requirement status if available
                requirement_status = None
                if req_col:
                    req_cell = worksheet[f"{req_col}{row_num}"]
                    req_value = str(req_cell.value or '').strip()
                    requirement_status = self._parse_requirement_status(req_value)
                
                # Fast validation checks
                if not tech_value or not display_value:
                    continue
                
                if not self._is_valid_technical_name(tech_value):
                    continue
                
                # Create mapping for valid rows
                mapping = ColumnMapping(
                    technical_name=tech_value,
                    display_name=display_value,
                    row_index=row_num,
                    technical_col=tech_col,
                    display_col=display_col,
                    requirement_status=requirement_status,
                    requirement_col=req_col
                )
                
                batch_mappings.append(mapping)
                
            except Exception as e:
                self.logger.debug(f"Error processing row {row_num}: {e}")
                continue
        
        return batch_mappings
    
    def _is_valid_technical_name(self, name: str) -> bool:
        """Validate technical name with optimized checks.
        
        Args:
            name: Technical name to validate
            
        Returns:
            True if valid technical name
        """
        if not name or len(name) > 50:
            return False
        
        if name != name.lower():
            return False
        
        if name[0].isdigit():
            return False
        
        return bool(self.TECHNICAL_NAME_PATTERN.match(name))
    
    def _parse_requirement_status(self, raw_value: str) -> Optional[str]:
        """Parse requirement status from German terms.
        
        Args:
            raw_value: Raw requirement value from Excel
            
        Returns:
            Standardized requirement status or None
        """
        if not raw_value or not raw_value.strip():
            return None
            
        normalized_value = raw_value.strip()
        
        # Direct mapping lookup
        if normalized_value in self.REQUIREMENT_STATUS_MAP:
            return self.REQUIREMENT_STATUS_MAP[normalized_value]
        
        # Case-insensitive fallback
        for key, value in self.REQUIREMENT_STATUS_MAP.items():
            if key.lower() == normalized_value.lower():
                return value
        
        # Unknown status - return None for graceful handling
        self.logger.debug(f"Unknown requirement status: '{normalized_value}'")
        return None
    
    def generate_requirement_statistics(self, mappings: List[ColumnMapping]) -> Dict[str, int]:
        """Generate requirement status distribution statistics.
        
        Args:
            mappings: List of column mappings
            
        Returns:
            Dictionary containing requirement status counts
        """
        stats = {
            'mandatory': 0,
            'optional': 0,
            'recommended': 0,
            'unknown': 0
        }
        
        for mapping in mappings:
            status = mapping.requirement_status
            if status in stats:
                stats[status] += 1
            else:
                stats['unknown'] += 1
        
        return stats
    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB.
        
        Returns:
            Current memory usage in megabytes
        """
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            try:
                import tracemalloc
                if tracemalloc.is_tracing():
                    current, _ = tracemalloc.get_traced_memory()
                    return current / 1024 / 1024
            except Exception:
                pass
            return 0.0
