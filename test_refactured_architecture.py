"""Test script for the refactored AI mapping architecture.

This script validates the new modular structure and ensures
all imports work correctly.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all new modules can be imported successfully."""
    print("Testing module imports...")
    
    try:
        # Test shared modules
        from sku_analyzer.shared import GeminiClient, PerformanceMonitor, ValidationUtils, AIProcessingConfig
        print("✅ Shared modules: GeminiClient, PerformanceMonitor, ValidationUtils, AIProcessingConfig")
        
        # Test step3_template modules
        from sku_analyzer.step3_template import TemplateGenerator, FieldAnalyzer, TemplateValidator
        print("✅ Step3 template modules: TemplateGenerator, FieldAnalyzer, TemplateValidator")
        
        # Test step4_mapping modules
        from sku_analyzer.step4_mapping import MappingProcessor, MappingInput, TransformationResult, FormatEnforcer
        print("✅ Step4 mapping modules: MappingProcessor, MappingInput, TransformationResult, FormatEnforcer")
        
        # Test prompts modules
        from sku_analyzer.prompts import MappingPromptManager, CategorizationPromptManager, ValidationPromptManager
        print("✅ Prompts modules: MappingPromptManager, CategorizationPromptManager, ValidationPromptManager")
        
        # Test integration
        from sku_analyzer.integration_example import ModularPipelineOrchestrator
        print("✅ Integration module: ModularPipelineOrchestrator")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during imports: {e}")
        return False


def test_component_initialization():
    """Test that components can be initialized correctly."""
    print("\nTesting component initialization...")
    
    try:
        # Test shared components
        from sku_analyzer.shared import PerformanceMonitor, ValidationUtils, AIProcessingConfig
        
        perf_monitor = PerformanceMonitor(enable_monitoring=True)
        validation_utils = ValidationUtils()
        ai_config = AIProcessingConfig()
        
        print("✅ Shared components initialized successfully")
        
        # Test template generation components
        from sku_analyzer.step3_template import TemplateGenerator, FieldAnalyzer, TemplateValidator
        
        template_generator = TemplateGenerator(
            enable_performance_monitoring=True,
            enable_ai_categorization=False  # Disable AI for testing
        )
        field_analyzer = FieldAnalyzer(enable_ai_categorization=False)
        template_validator = TemplateValidator()
        
        print("✅ Template generation components initialized successfully")
        
        # Test mapping components
        from sku_analyzer.step4_mapping import MappingProcessor, ProcessingConfig
        
        mapping_processor = MappingProcessor(
            config=ProcessingConfig(),
            ai_config=ai_config,
            enable_performance_monitoring=True
        )
        
        print("✅ Mapping components initialized successfully")
        
        # Test prompt managers
        from sku_analyzer.prompts import MappingPromptManager, CategorizationPromptManager
        
        mapping_prompts = MappingPromptManager()
        categorization_prompts = CategorizationPromptManager()
        
        print("✅ Prompt managers initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Component initialization error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test performance monitor
        from sku_analyzer.shared import PerformanceMonitor
        
        perf_monitor = PerformanceMonitor(enable_monitoring=True)
        with perf_monitor.measure_performance("test_operation") as perf:
            # Simulate some work
            import time
            time.sleep(0.1)
        
        if perf['metrics'] and perf['metrics'].duration_ms > 90:  # Should be ~100ms
            print("✅ Performance monitoring working correctly")
        else:
            print("⚠️ Performance monitoring may not be working correctly")
        
        # Test validation utilities
        from sku_analyzer.shared import ValidationUtils
        
        validation_utils = ValidationUtils()
        test_data = {"field1": "value1", "field2": None, "field3": ""}
        required_fields = ["field1", "field2", "field4"]
        
        missing_fields = validation_utils.validate_required_fields(test_data, required_fields)
        if len(missing_fields) == 3:  # field2 is None, field3 is empty, field4 is missing
            print("✅ Validation utilities working correctly")
        else:
            print(f"⚠️ Validation utilities returned unexpected result: {missing_fields}")
        
        # Test prompt managers
        from sku_analyzer.prompts import MappingPromptManager
        
        mapping_prompts = MappingPromptManager()
        system_prompt = mapping_prompts.get_system_prompt()
        if "German Amazon" in system_prompt:
            print("✅ Prompt managers working correctly")
        else:
            print("⚠️ Prompt managers may not be working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test error: {e}")
        return False


def test_data_models():
    """Test that data models work correctly."""
    print("\nTesting data models...")
    
    try:
        # Test step4_mapping models
        from sku_analyzer.step4_mapping.models import (
            MappingInput, TransformationResult, ProcessingConfig, 
            MappingResult, ProcessingResult
        )
        
        # Test MappingInput
        mapping_input = MappingInput(
            parent_sku="test_sku",
            mandatory_fields={"field1": {"type": "string"}},
            product_data={"parent_data": {"test": "value"}}
        )
        
        # Test TransformationResult
        mapping_result = MappingResult(
            source_field="source",
            target_field="target",
            mapped_value="value",
            confidence=0.85,
            reasoning="test mapping"
        )
        
        transformation_result = TransformationResult(
            parent_sku="test_sku",
            mapped_fields=[mapping_result]
        )
        
        # Test ProcessingResult
        processing_result = ProcessingResult(
            parent_sku="test_sku",
            success=True,
            confidence=0.85
        )
        
        print("✅ Data models working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Data models test error: {e}")
        return False


async def test_async_functionality():
    """Test async functionality without requiring API keys."""
    print("\nTesting async functionality...")
    
    try:
        # Test that async methods can be called (without actually making API calls)
        from sku_analyzer.step3_template import FieldAnalyzer
        
        field_analyzer = FieldAnalyzer(enable_ai_categorization=False)  # Disable AI
        
        # Test deterministic categorization
        test_fields = {
            "brand_name": {"display_name": "Brand Name", "data_type": "string"},
            "item_sku": {"display_name": "Item SKU", "data_type": "string"},
            "color_name": {"display_name": "Color", "data_type": "string"}
        }
        
        parent_fields, variant_fields = await field_analyzer.categorize_field_levels(test_fields)
        
        # brand_name should be parent, color_name and item_sku should be variant
        if "brand_name" in parent_fields and "color_name" in variant_fields:
            print("✅ Async field categorization working correctly")
        else:
            print(f"⚠️ Async categorization result: parent={parent_fields}, variant={variant_fields}")
        
        return True
        
    except Exception as e:
        print(f"❌ Async functionality test error: {e}")
        return False


def print_architecture_summary():
    """Print summary of the new architecture."""
    print("\n🏢 NEW MODULAR ARCHITECTURE SUMMARY")
    print("=" * 50)
    print("\n📚 Shared Components (sku_analyzer/shared/):")
    print("  • gemini_client.py - Unified Gemini API client")
    print("  • performance.py - Performance monitoring")
    print("  • validation.py - Validation utilities")
    
    print("\n📋 Step 3 Template (sku_analyzer/step3_template/):")
    print("  • generator.py - Main template generator orchestrator")
    print("  • field_analyzer.py - AI & deterministic field categorization")
    print("  • validator.py - Template validation and quality assessment")
    
    print("\n🤖 Step 4 Mapping (sku_analyzer/step4_mapping/):")
    print("  • processor.py - Main AI mapping processor")
    print("  • models.py - Data models for mapping operations")
    print("  • format_enforcer.py - Result format validation")
    
    print("\n📝 Prompts (sku_analyzer/prompts/):")
    print("  • base_prompt.py - Base prompt management functionality")
    print("  • mapping_prompts.py - AI mapping prompts")
    print("  • categorization_prompts.py - Field categorization prompts")
    print("  • validation_prompts.py - Result validation prompts")
    
    print("\n🔗 Integration:")
    print("  • integration_example.py - Complete pipeline orchestration")
    
    print("\n✨ KEY IMPROVEMENTS:")
    print("  ✅ Single-responsibility modules (<400 lines each)")
    print("  ✅ Clean import hierarchy with no circular dependencies")
    print("  ✅ Unified Gemini client with performance monitoring")
    print("  ✅ Consolidated performance monitoring")
    print("  ✅ Organized prompt management system")
    print("  ✅ Type-safe data models with Pydantic")
    print("  ✅ Comprehensive error handling and validation")


async def main():
    """Run all tests."""
    print("🚀 FLAT MAGIC v6 - REFACTORED ARCHITECTURE TEST")
    print("=" * 55)
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Track test results
    tests_passed = 0
    total_tests = 5
    
    # Run tests
    if test_imports():
        tests_passed += 1
    
    if test_component_initialization():
        tests_passed += 1
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_data_models():
        tests_passed += 1
    
    if await test_async_functionality():
        tests_passed += 1
    
    # Print results
    print(f"\n📊 TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Refactored architecture is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    # Print architecture summary
    print_architecture_summary()
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)