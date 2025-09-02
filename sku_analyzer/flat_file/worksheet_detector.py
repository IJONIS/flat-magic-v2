"""Worksheet detection module for XLSM template analysis.

This module provides focused functionality for detecting and selecting
the correct worksheet containing template data.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

from openpyxl.worksheet.worksheet import Worksheet

from .data_structures import TemplateDetectionError


class WorksheetDetector:
    """Optimized worksheet detection with pattern matching."""
    
    # Worksheet detection constants
    MAX_WORKSHEET_SCAN = 5
    TECHNICAL_HEADER = "Feldname"
    DISPLAY_HEADER = "Lokale Bezeichnung"
    
    # Worksheet priority order
    WORKSHEET_PRIORITIES = [
        "Datendefinitionen",
        "Vorlage",
        # Fuzzy match patterns
        r".*[Dd]aten.*",
        r".*[Vv]orlage.*",
        r".*[Tt]emplate.*",
        r".*[Dd]efinition.*"
    ]
    
    def __init__(self) -> None:
        """Initialize worksheet detector."""
        self.logger = logging.getLogger(__name__)
        self._compiled_patterns = self._precompile_patterns()
    
    def _precompile_patterns(self) -> List[re.Pattern[str]]:
        """Precompile regex patterns for faster matching."""
        patterns = []
        for pattern in self.WORKSHEET_PRIORITIES[2:]:  # Skip exact matches
            try:
                patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                self.logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        return patterns
    
    def detect_target_worksheet(self, workbook: Any) -> Worksheet:
        """Detect the target worksheet containing template data.
        
        Args:
            workbook: Loaded openpyxl workbook
            
        Returns:
            Target worksheet containing template headers
            
        Raises:
            TemplateDetectionError: When no suitable worksheet found
        """
        worksheet_names = workbook.sheetnames
        self.logger.debug(f"Available worksheets: {worksheet_names[:self.MAX_WORKSHEET_SCAN]}")
        
        # Try exact matches first (fastest path)
        for priority_name in self.WORKSHEET_PRIORITIES[:2]:
            if priority_name in worksheet_names:
                self.logger.info(f"Found exact match worksheet: '{priority_name}'")
                return workbook[priority_name]
        
        # Try precompiled fuzzy patterns (limited scan)
        scan_limit = min(self.MAX_WORKSHEET_SCAN, len(worksheet_names))
        for pattern in self._compiled_patterns:
            for sheet_name in worksheet_names[:scan_limit]:
                if pattern.match(sheet_name):
                    self.logger.info(f"Found fuzzy match: '{sheet_name}'")
                    return workbook[sheet_name]
        
        # Fallback: check first worksheet
        first_sheet = workbook[worksheet_names[0]]
        if self._contains_required_headers(first_sheet):
            self.logger.info(f"Using first worksheet with headers: '{first_sheet.title}'")
            return first_sheet
        
        raise TemplateDetectionError(
            f"No suitable worksheet found in first {scan_limit} sheets. "
            f"Available: {worksheet_names[:scan_limit]}. "
            f"Expected worksheets containing '{self.TECHNICAL_HEADER}' and '{self.DISPLAY_HEADER}' headers."
        )
    
    def _contains_required_headers(self, worksheet: Worksheet) -> bool:
        """Check if worksheet contains required headers.
        
        Args:
            worksheet: Worksheet to check
            
        Returns:
            True if both required headers found
        """
        max_check_rows = min(20, worksheet.max_row or 20)
        
        for row_num in range(1, max_check_rows + 1):
            try:
                row_cells = list(worksheet[row_num])
                
                tech_found = False
                display_found = False
                
                for cell in row_cells:
                    value = str(cell.value or '').strip()
                    if value == self.TECHNICAL_HEADER:
                        tech_found = True
                    elif value == self.DISPLAY_HEADER:
                        display_found = True
                    
                    if tech_found and display_found:
                        return True
                        
            except Exception:
                continue
        
        return False
