"""SKU Pattern Analyzer Package.

A modern Python package for analyzing SKU parent-child relationships
using hierarchical delimiter-based pattern recognition.
"""

from .core import SkuPatternAnalyzer, FileSplitter, CompressionEngine, CompressionResult, compress_job_data
from .core.analyzer import PipelineValidationError
from .models import SkuPattern, ParentChildRelationship, ProcessingJob
from .utils import JobManager
from .flat_file import XlsmTemplateAnalyzer

__version__ = "1.0.0"
__all__ = ['SkuPatternAnalyzer', 'FileSplitter', 'CompressionEngine', 'CompressionResult', 'compress_job_data', 'PipelineValidationError', 'SkuPattern', 'ParentChildRelationship', 'ProcessingJob', 'JobManager', 'XlsmTemplateAnalyzer']