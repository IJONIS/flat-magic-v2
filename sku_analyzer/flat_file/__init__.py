"""Modern flat file template analysis with clean modular architecture.

Refactored to use focused, single-responsibility modules following
modern Python patterns with full type hints and clean separation of concerns.
"""

from __future__ import annotations

# Core data structures
from .data_structures import (
    ColumnMapping,
    PerformanceMetrics,
    TemplateAnalysisResult,
    ExtractionMetrics,
    FieldValidation,
    ExtractionResult,
    TemplateDetectionError
)

# Focused components
from .worksheet_detector import WorksheetDetector
from .header_detector import HeaderDetector
from .column_extractor import ColumnExtractor
from .value_extractor import ValueExtractor
from .mandatory_fields_processor import MandatoryFieldsProcessor
from .performance_monitor import PerformanceMonitor
from .validation_utils import ValidationUtils

# Main analyzers
from .template_analyzer import XlsmTemplateAnalyzer, OptimizedXlsmTemplateAnalyzer
# Use the refactored analyzer as the modern version
ModernXlsmTemplateAnalyzer = OptimizedXlsmTemplateAnalyzer
from .template_data_extractor import (
    HighPerformanceTemplateDataExtractor,
    TemplateDataExtractor,
    create_step2_extractor,
    extract_template_data
)

__all__ = [
    # Core data structures
    'ColumnMapping',
    'PerformanceMetrics',
    'TemplateAnalysisResult',
    'ExtractionMetrics',
    'FieldValidation',
    'ExtractionResult',
    'TemplateDetectionError',
    
    # Focused components
    'WorksheetDetector',
    'HeaderDetector',
    'ColumnExtractor',
    'ValueExtractor',
    'MandatoryFieldsProcessor',
    'PerformanceMonitor',
    'ValidationUtils',
    
    # Main analyzers
    'XlsmTemplateAnalyzer',                    # Backward compatibility
    'OptimizedXlsmTemplateAnalyzer',           # Modular optimized version
    'ModernXlsmTemplateAnalyzer',              # Alias for modern version
    'HighPerformanceTemplateDataExtractor',    # Main Step 2 extractor
    'TemplateDataExtractor',                   # Backward compatibility
    
    # Convenience functions
    'create_step2_extractor',
    'extract_template_data'
]


# Recommended modern usage
def create_modern_analyzer(enable_performance_monitoring: bool = True) -> ModernXlsmTemplateAnalyzer:
    """Create a modern template analyzer with clean modular architecture.
    
    Args:
        enable_performance_monitoring: Whether to enable performance tracking
        
    Returns:
        Configured modern template analyzer
    """
    return ModernXlsmTemplateAnalyzer(enable_performance_monitoring)


def create_modern_extractor(enable_performance_monitoring: bool = True) -> HighPerformanceTemplateDataExtractor:
    """Create a modern data extractor with clean modular architecture.
    
    Args:
        enable_performance_monitoring: Whether to enable performance tracking
        
    Returns:
        Configured modern data extractor
    """
    return HighPerformanceTemplateDataExtractor(enable_performance_monitoring)


def create_mandatory_fields_processor(enable_performance_monitoring: bool = True) -> MandatoryFieldsProcessor:
    """Create a mandatory fields processor for Step 3 processing.
    
    Args:
        enable_performance_monitoring: Whether to enable performance tracking
        
    Returns:
        Configured mandatory fields processor
    """
    return MandatoryFieldsProcessor(enable_performance_monitoring)


# Add convenience functions to __all__
__all__.extend(['create_modern_analyzer', 'create_modern_extractor', 'create_mandatory_fields_processor'])