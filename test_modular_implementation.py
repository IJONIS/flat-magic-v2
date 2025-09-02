#!/usr/bin/env python3
"""Test script for the modular template analyzer implementation.

This script validates that the refactored modular design works correctly
and maintains compatibility with existing functionality.
"""

from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_modular_imports():
    """Test that all modular components can be imported correctly."""
    print("🧪 Testing modular imports...")
    
    try:
        # Test core data structures
        from sku_analyzer.flat_file.data_structures import (
            ColumnMapping, 
            PerformanceMetrics, 
            TemplateAnalysisResult,
            TemplateDetectionError
        )
        print("✅ Data structures imported successfully")
        
        # Test focused components
        from sku_analyzer.flat_file.worksheet_detector import WorksheetDetector
        from sku_analyzer.flat_file.header_detector import HeaderDetector
        from sku_analyzer.flat_file.column_extractor import ColumnExtractor
        from sku_analyzer.flat_file.value_extractor import ValueExtractor
        from sku_analyzer.flat_file.performance_monitor import PerformanceMonitor
        from sku_analyzer.flat_file.validation_utils import ValidationUtils
        print("✅ Focused components imported successfully")
        
        # Test main analyzers
        from sku_analyzer.flat_file.template_analyzer import OptimizedXlsmTemplateAnalyzer
        from sku_analyzer.flat_file.template_data_extractor import HighPerformanceTemplateDataExtractor
        print("✅ Main analyzers imported successfully")
        
        # Test convenience functions
        from sku_analyzer.flat_file import create_modern_analyzer, create_modern_extractor
        print("✅ Convenience functions imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_component_instantiation():
    """Test that modular components can be instantiated correctly."""
    print("🧪 Testing component instantiation...")
    
    try:
        from sku_analyzer.flat_file.worksheet_detector import WorksheetDetector
        from sku_analyzer.flat_file.header_detector import HeaderDetector
        from sku_analyzer.flat_file.column_extractor import ColumnExtractor
        from sku_analyzer.flat_file.value_extractor import ValueExtractor
        from sku_analyzer.flat_file.performance_monitor import PerformanceMonitor
        from sku_analyzer.flat_file.validation_utils import ValidationUtils
        
        # Instantiate components
        worksheet_detector = WorksheetDetector()
        header_detector = HeaderDetector()
        column_extractor = ColumnExtractor()
        value_extractor = ValueExtractor()
        performance_monitor = PerformanceMonitor()
        validation_utils = ValidationUtils()
        
        print("✅ All components instantiated successfully")
        
        # Test that components have expected methods
        assert hasattr(worksheet_detector, 'detect_target_worksheet')
        assert hasattr(header_detector, 'detect_header_row')
        assert hasattr(column_extractor, 'extract_column_mappings')
        assert hasattr(value_extractor, 'extract_bulk_values')
        assert hasattr(performance_monitor, 'measure_performance')
        assert hasattr(validation_utils, 'validate_analysis_result')
        
        print("✅ All components have expected methods")
        return True
        
    except Exception as e:
        print(f"❌ Component instantiation failed: {e}")
        return False

def test_analyzer_creation():
    """Test that analyzers can be created using factory functions."""
    print("🧪 Testing analyzer creation...")
    
    try:
        from sku_analyzer.flat_file import create_modern_analyzer, create_modern_extractor
        
        # Create analyzer and extractor
        analyzer = create_modern_analyzer(enable_performance_monitoring=True)
        extractor = create_modern_extractor(enable_performance_monitoring=True)
        
        print("✅ Modern analyzer and extractor created successfully")
        
        # Test that they have expected methods
        assert hasattr(analyzer, 'analyze_template')
        assert hasattr(extractor, 'extract_template_values')
        
        print("✅ Analyzers have expected methods")
        return True
        
    except Exception as e:
        print(f"❌ Analyzer creation failed: {e}")
        return False

def test_data_structure_creation():
    """Test that data structures can be created and used correctly."""
    print("🧪 Testing data structure creation...")
    
    try:
        from sku_analyzer.flat_file.data_structures import ColumnMapping, PerformanceMetrics
        
        # Create a column mapping
        mapping = ColumnMapping(
            technical_name="test_field",
            display_name="Test Field",
            row_index=1,
            technical_col="A",
            display_col="B",
            requirement_status="optional"
        )
        
        # Test serialization
        mapping_dict = mapping.to_dict()
        assert 'technical_name' in mapping_dict
        assert mapping_dict['technical_name'] == "test_field"
        
        print("✅ ColumnMapping created and serialized successfully")
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            duration_ms=100.0,
            peak_memory_mb=10.0,
            memory_delta_mb=2.0,
            rows_processed=50
        )
        
        # Test derived metrics calculation
        assert hasattr(metrics, 'throughput_rows_per_second')
        assert metrics.throughput_rows_per_second > 0
        
        print("✅ PerformanceMetrics created with derived metrics")
        return True
        
    except Exception as e:
        print(f"❌ Data structure creation failed: {e}")
        return False

def main():
    """Run all modular implementation tests."""
    print("🚀 Starting modular implementation tests...\n")
    
    tests = [
        test_modular_imports,
        test_component_instantiation,
        test_analyzer_creation,
        test_data_structure_creation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}\n")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The modular implementation is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)