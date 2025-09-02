"""Hierarchical SKU pattern extraction."""

from typing import Dict, List
import pandas as pd


class HierarchyExtractor:
    """Extract hierarchical patterns from SKU data using delimiter splitting."""
    
    def __init__(self, min_pattern_length: int = 3):
        self.min_pattern_length = min_pattern_length
    
    def extract_hierarchical_patterns(self, df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        """Extract hierarchical SKU patterns using delimiter splitting."""
        sku_column = 'SUPPLIER_PID'
        if sku_column not in df.columns:
            raise ValueError(f"Required column '{sku_column}' not found")
        
        # Clean and sort SKUs for deterministic processing
        skus = df[sku_column].dropna().astype(str).sort_values().unique()
        
        hierarchy = {}
        
        for sku in skus:
            parts = self._split_sku_hierarchically(sku)
            if len(parts) >= 2:
                # Level 1: Main category (e.g., "4301")
                level1 = parts[0]
                if level1 not in hierarchy:
                    hierarchy[level1] = {}
                
                # Level 2: Sub-category (e.g., "4301_40") 
                if len(parts) >= 3:
                    level2 = f"{parts[0]}_{parts[1]}"
                    if level2 not in hierarchy[level1]:
                        hierarchy[level1][level2] = []
                    hierarchy[level1][level2].append(sku)
                else:
                    # Two-part SKU goes directly under level1
                    level2 = sku
                    if level2 not in hierarchy[level1]:
                        hierarchy[level1][level2] = []
        
        return hierarchy
    
    def _split_sku_hierarchically(self, sku: str) -> List[str]:
        """Split SKU by common delimiters in hierarchical order."""
        # Try delimiters in order of importance
        delimiters = ['_', '-', '.', ' ']
        
        for delimiter in delimiters:
            if delimiter in sku:
                return sku.split(delimiter)
        
        # No delimiter found, return as single part
        return [sku]