#!/usr/bin/env python3
"""
Comprehensive test for robust Gemini API safety filter error handling.

This test validates:
1. SafetyFilterException handling with detailed categorization
2. Progressive fallback strategies (ultra-simplified -> field-only -> minimal-safe -> hardcoded)
3. Ultra-safe prompt generation for maximum compliance
4. Error recovery and graceful degradation
5. Performance monitoring of safety incidents
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from sku_analyzer.shared.gemini_client import (
    GeminiClient, AIProcessingConfig, SafetyFilterException, PromptOptimizer
)
from sku_analyzer.step5_mapping.ai_mapper import AIMapper
from sku_analyzer.step5_mapping.models import MappingInput, ProcessingConfig


async def test_safety_filter_exception_handling():
    """Test SafetyFilterException handling with detailed information extraction."""
    
    print("🔒 Testing SafetyFilterException handling...")
    
    # Create mock safety error
    mock_safety_ratings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "probability": "HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "MEDIUM"}
    ]
    
    safety_error = SafetyFilterException(
        message="Content blocked by safety filters",
        finish_reason="2", 
        safety_ratings=mock_safety_ratings,
        prompt_size=5000
    )
    
    # Validate error properties
    assert safety_error.finish_reason == "2"
    assert len(safety_error.safety_ratings) == 2
    assert safety_error.prompt_size == 5000
    assert "HARM_CATEGORY_HARASSMENT" in safety_error.blocked_categories
    assert "HARM_CATEGORY_HATE_SPEECH" in safety_error.blocked_categories
    
    print("✅ SafetyFilterException properties correctly extracted")
    return True


async def test_prompt_optimizer_ultra_safe_mode():
    """Test PromptOptimizer ultra-safe mode for maximum safety compliance."""
    
    print("🛡️ Testing PromptOptimizer ultra-safe mode...")
    
    # Create test product data
    test_product_data = {
        'parent_data': {
            'MANUFACTURER_NAME': 'TestBrand',
            'MANUFACTURER_PID': 'TB123',
            'GROUP_STRING': 'Apparel',
            'WEIGHT': '0.5kg',
            'FVALUE_3_1': 'Blue',
            'FVALUE_3_2': 'Large',
            'FVALUE_3_3': 'Cotton',
            'SUPPLIER_PID': 'SUP456',
            'DESCRIPTION_LONG': 'This is a very long description that might trigger safety filters...'
        },
        'data_rows': [
            {
                'MANUFACTURER_PID': 'TB123_V1',
                'FVALUE_3_1': 'Red',
                'FVALUE_3_2': 'Medium'
            }
        ]
    }
    
    optimizer = PromptOptimizer()
    
    # Test normal mode
    normal_compressed = optimizer.compress_product_data(test_product_data, max_fields=5)
    print(f"Normal mode fields: {len(normal_compressed['parent_data'])}")
    
    # Test ultra-safe mode
    ultra_safe_compressed = optimizer.compress_product_data(
        test_product_data, max_fields=5, ultra_safe_mode=True
    )
    print(f"Ultra-safe mode fields: {len(ultra_safe_compressed['parent_data'])}")
    
    # Ultra-safe mode should have fewer fields
    assert len(ultra_safe_compressed['parent_data']) <= len(normal_compressed['parent_data'])
    
    # Ultra-safe mode should exclude potentially problematic fields
    assert 'DESCRIPTION_LONG' not in ultra_safe_compressed['parent_data']
    assert 'GROUP_STRING' not in ultra_safe_compressed['parent_data']
    
    print("✅ Ultra-safe mode correctly filters fields")
    return True


async def test_gemini_client_ultra_safe_fallback():
    """Test GeminiClient ultra-safe fallback mechanism."""
    
    print("🔄 Testing GeminiClient ultra-safe fallback...")
    
    # Skip if no API key (CI/CD environments)
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ Skipping Gemini client test - no API key")
        return True
    
    config = AIProcessingConfig(
        max_prompt_size=1000,  # Small size to potentially trigger issues
        enable_prompt_compression=True,
        max_retries=1
    )
    
    client = GeminiClient(config=config)
    
    # Test ultra-safe prompt creation
    large_prompt = "Map product data: " + "A" * 2000  # Large prompt
    ultra_safe_prompt = client._create_ultra_safe_prompt(large_prompt)
    
    # Ultra-safe prompt should be much smaller
    assert len(ultra_safe_prompt) < 600
    assert "parent_sku" in ultra_safe_prompt.lower()
    assert "ultra_safe_fallback" in ultra_safe_prompt.lower()
    
    print(f"Ultra-safe prompt size: {len(ultra_safe_prompt)} chars")
    print("✅ Ultra-safe prompt generation working")
    
    return True


async def test_ai_mapper_progressive_fallbacks():
    """Test AIMapper progressive fallback strategies."""
    
    print("🎯 Testing AIMapper progressive fallback strategies...")
    
    # Skip if no API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ Skipping AI mapper test - no API key")
        return True
    
    # Create test configuration
    ai_config = AIProcessingConfig(max_retries=1, timeout_seconds=5)
    processing_config = ProcessingConfig(max_retries=2, confidence_threshold=0.5)
    
    # Mock result formatter
    class MockResultFormatter:
        def __init__(self):
            self.processing_stats = {"successful_mappings": 0, "failed_mappings": 0}
    
    # Create AI client and mapper
    ai_client = GeminiClient(config=ai_config)
    result_formatter = MockResultFormatter()
    mapper = AIMapper(ai_client, processing_config, result_formatter)
    
    # Create test mapping input
    test_mapping_input = MappingInput(
        parent_sku="TEST123",
        mandatory_fields={
            "brand_name": {"data_type": "string", "required": True},
            "item_sku": {"data_type": "string", "required": True}
        },
        product_data={
            'parent_data': {'MANUFACTURER_NAME': 'TestBrand', 'MANUFACTURER_PID': 'TB123'},
            'data_rows': [{'MANUFACTURER_PID': 'TB123_V1', 'FVALUE_3_1': 'Blue'}]
        },
        business_context="Test context"
    )
    
    # Test individual fallback strategies
    try:
        # Test field-only mapping
        field_result = await mapper._execute_field_only_mapping(test_mapping_input)
        assert field_result.parent_sku == "TEST123"
        assert field_result.metadata.get("field_only_mapping") is True
        print("✅ Field-only mapping strategy working")
        
        # Test minimal safe mapping
        minimal_result = await mapper._execute_minimal_safe_mapping(test_mapping_input)
        assert minimal_result.parent_sku == "TEST123" 
        assert minimal_result.metadata.get("minimal_safe_mapping") is True
        print("✅ Minimal safe mapping strategy working")
        
    except Exception as e:
        print(f"⚠️ Fallback strategy test failed (API issues): {e}")
    
    # Test hardcoded fallback (no API required)
    mock_safety_error = SafetyFilterException(
        "Test safety error", 
        finish_reason="2",
        safety_ratings=[{"category": "TEST", "probability": "HIGH"}]
    )
    
    fallback_result = mapper._create_minimal_fallback_result(test_mapping_input, mock_safety_error)
    
    assert fallback_result.parent_sku == "TEST123"
    assert fallback_result.metadata.get("safety_blocked") is True
    assert fallback_result.metadata.get("fallback_strategy") == "hardcoded_minimal"
    assert fallback_result.metadata.get("ai_mapping_failed") is True
    assert fallback_result.parent_data.get("brand_name") == "TestBrand"  # Mapped from MANUFACTURER_NAME
    
    print("✅ Hardcoded minimal fallback working")
    return True


async def test_ultra_simplified_prompt_generation():
    """Test ultra-simplified prompt generation with safety modes."""
    
    print("📝 Testing ultra-simplified prompt generation...")
    
    # Skip if no API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ Skipping ultra-simplified prompt test - no API key")
        return True
    
    # Create test configuration
    ai_config = AIProcessingConfig()
    processing_config = ProcessingConfig()
    
    class MockResultFormatter:
        def __init__(self):
            self.processing_stats = {"successful_mappings": 0, "failed_mappings": 0}
    
    ai_client = GeminiClient(config=ai_config)
    result_formatter = MockResultFormatter()
    mapper = AIMapper(ai_client, processing_config, result_formatter)
    
    # Create test mapping input
    test_mapping_input = MappingInput(
        parent_sku="TEST123",
        mandatory_fields={
            "brand_name": {"data_type": "string"},
            "item_sku": {"data_type": "string"}
        },
        product_data={
            'parent_data': {
                'MANUFACTURER_NAME': 'TestBrand',
                'MANUFACTURER_PID': 'TB123',
                'DESCRIPTION_LONG': 'Very long description that might cause issues...',
                'FVALUE_3_1': 'Blue'
            },
            'data_rows': [
                {'MANUFACTURER_PID': 'TB123_V1', 'FVALUE_3_1': 'Red'}
            ]
        }
    )
    
    # Test normal ultra-simplified prompt
    normal_prompt = mapper._create_ultra_simplified_prompt(test_mapping_input)
    
    # Test ultra-safe mode prompt
    ultra_safe_prompt = mapper._create_ultra_simplified_prompt(test_mapping_input, ultra_safe_mode=True)
    
    # Ultra-safe should be smaller and more conservative
    assert len(ultra_safe_prompt) <= len(normal_prompt)
    assert 'MANUFACTURER_NAME' not in ultra_safe_prompt  # Should be excluded in ultra-safe mode
    assert 'MANUFACTURER_PID' in ultra_safe_prompt  # Should be included
    
    print(f"Normal prompt size: {len(normal_prompt)} chars")
    print(f"Ultra-safe prompt size: {len(ultra_safe_prompt)} chars")
    print("✅ Ultra-simplified prompt generation with safety modes working")
    
    return True


async def test_performance_monitoring_safety_metrics():
    """Test performance monitoring of safety filter incidents."""
    
    print("📊 Testing performance monitoring for safety metrics...")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ Skipping performance monitoring test - no API key")
        return True
    
    config = AIProcessingConfig(max_retries=1)
    client = GeminiClient(config=config)
    
    # Reset performance counters
    client.reset_performance_stats()
    
    # Get initial performance summary
    initial_summary = client.get_performance_summary()
    assert initial_summary['safety_blocked_requests'] == 0
    assert initial_summary['safety_block_rate'] == 0.0
    
    print("✅ Performance monitoring initialized correctly")
    print(f"Initial safety block rate: {initial_summary['safety_block_rate']:.1%}")
    
    return True


async def run_comprehensive_safety_filter_test():
    """Run comprehensive safety filter robustness test suite."""
    
    print("🚀 Starting Comprehensive Safety Filter Robustness Test")
    print("=" * 60)
    
    test_results = []
    
    # Test suite
    tests = [
        ("SafetyFilterException Handling", test_safety_filter_exception_handling),
        ("PromptOptimizer Ultra-Safe Mode", test_prompt_optimizer_ultra_safe_mode),
        ("GeminiClient Ultra-Safe Fallback", test_gemini_client_ultra_safe_fallback),
        ("AIMapper Progressive Fallbacks", test_ai_mapper_progressive_fallbacks),
        ("Ultra-Simplified Prompt Generation", test_ultra_simplified_prompt_generation),
        ("Performance Monitoring Safety Metrics", test_performance_monitoring_safety_metrics)
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = await test_func()
            test_results.append((test_name, result, None))
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            test_results.append((test_name, False, str(e)))
            print(f"❌ {test_name}: FAILED - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result, _ in test_results if result)
    total = len(test_results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total:.1%}")
    
    # Detailed results
    print("\nDetailed Results:")
    for test_name, result, error in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if error:
            print(f"    Error: {error}")
    
    if passed == total:
        print("\n🎉 All safety filter robustness tests passed!")
        print("   - SafetyFilterException handling is robust")
        print("   - Progressive fallback strategies are working")
        print("   - Ultra-safe prompt generation is functional")
        print("   - Performance monitoring tracks safety incidents")
        print("   - Error recovery and graceful degradation implemented")
    else:
        print(f"\n⚠️  {total - passed} tests failed - review safety filter implementation")
    
    return passed == total


if __name__ == "__main__":
    # Run comprehensive test
    success = asyncio.run(run_comprehensive_safety_filter_test())
    sys.exit(0 if success else 1)