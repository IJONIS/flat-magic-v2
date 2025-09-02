"""Header detection module for XLSM template analysis.

This module provides focused functionality for detecting header rows
and column positions within worksheets.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from openpyxl.worksheet.worksheet import Worksheet

from .data_structures import TemplateDetectionError


class HeaderDetector:
    """Optimized header detection with early termination."""
    
    # Header detection constants
    MAX_HEADER_SEARCH_ROWS = 50
    TECHNICAL_HEADER = "Feldname"
    DISPLAY_HEADER = "Lokale Bezeichnung"
    REQUIREMENT_HEADER = "Pflichtfeld?"
    
    def __init__(self) -> None:
        """Initialize header detector."""
        self.logger = logging.getLogger(__name__)
    
    def detect_header_row(self, worksheet: Worksheet) -> Tuple[int, str, str, Optional[str]]:
        """Detect header row and column positions.
        
        Args:
            worksheet: Worksheet to analyze
            
        Returns:
            Tuple of (header_row, technical_col, display_col, requirement_col)
            
        Raises:
            TemplateDetectionError: When required headers not found
        """
        max_search_rows = min(self.MAX_HEADER_SEARCH_ROWS, worksheet.max_row or 50)
        
        self.logger.debug(f"Scanning for headers in first {max_search_rows} rows")
        
        for row_num in range(1, max_search_rows + 1):
            try:
                row_cells = list(worksheet[row_num])
                
                tech_col_idx = None
                display_col_idx = None
                req_col_idx = None
                
                for idx, cell in enumerate(row_cells):
                    value = str(cell.value or '').strip()
                    
                    if value == self.TECHNICAL_HEADER and tech_col_idx is None:
                        tech_col_idx = idx
                    elif value == self.DISPLAY_HEADER and display_col_idx is None:
                        display_col_idx = idx
                    elif value == self.REQUIREMENT_HEADER and req_col_idx is None:
                        req_col_idx = idx
                
                # Early termination when required headers found
                if tech_col_idx is not None and display_col_idx is not None:
                    tech_col = self._index_to_column_letter(tech_col_idx)
                    display_col = self._index_to_column_letter(display_col_idx)
                    req_col = self._index_to_column_letter(req_col_idx) if req_col_idx is not None else None
                    
                    log_msg = (
                        f"Headers found at row {row_num}: "
                        f"'{self.TECHNICAL_HEADER}' in {tech_col}, "
                        f"'{self.DISPLAY_HEADER}' in {display_col}"
                    )
                    if req_col:
                        log_msg += f", '{self.REQUIREMENT_HEADER}' in {req_col}"
                    else:
                        log_msg += " (no requirement column found)"
                    
                    self.logger.info(log_msg)
                    return row_num, tech_col, display_col, req_col
                    
            except Exception as e:
                self.logger.debug(f"Error scanning row {row_num}: {e}")
                continue
        
        raise TemplateDetectionError(
            f"Required headers not found in first {max_search_rows} rows. "
            f"Expected: '{self.TECHNICAL_HEADER}' and '{self.DISPLAY_HEADER}'"
        )
    
    def _index_to_column_letter(self, idx: int) -> str:
        """Convert 0-based column index to Excel column letter.
        
        Args:
            idx: Zero-based column index
            
        Returns:
            Excel column letter (A, B, C, etc.)
        """
        result = ""
        idx += 1  # Convert to 1-based
        while idx > 0:
            idx -= 1
            result = chr(65 + idx % 26) + result
            idx //= 26
        return result
