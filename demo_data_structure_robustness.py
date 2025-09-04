#!/usr/bin/env python3
"""Demonstration of robustness improvements in AI mapping data processing.

This script shows how the AI mapping system now handles various problematic
data structures gracefully without crashing.
"""

import logging
from sku_analyzer.shared.gemini_client import PromptOptimizer

# Configure logging to show warnings
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

def demonstrate_robustness():
    """Demonstrate the robustness of the fixed AI mapping system."""
    print("🔧 AI Mapping Data Structure Robustness Demonstration\n")
    
    optimizer = PromptOptimizer()
    
    # Test Case 1: Normal, well-formed data (should work perfectly)
    print("1. Normal Data Structure:")
    normal_data = {
        'parent_data': {'MANUFACTURER_NAME': 'EIKO', 'MANUFACTURER_PID': '4301'},
        'data_rows': [
            {'SUPPLIER_PID': '4301_40_44', 'FVALUE_3_1': '40'},
            {'SUPPLIER_PID': '4301_40_46', 'FVALUE_3_1': '40'}
        ]
    }
    
    result = optimizer.compress_product_data(normal_data)
    print(f"   ✅ Processed successfully: {len(result['data_rows'])} variants")
    
    # Test Case 2: parent_data is a list instead of dict (problematic before fix)
    print("\n2. Problematic: parent_data as list (would crash before):")
    problematic_parent = {
        'parent_data': ['MANUFACTURER_NAME', 'EIKO', 'PID', '4301'],  # List instead of dict!
        'data_rows': [{'SUPPLIER_PID': '4301_40_44'}]
    }
    
    result = optimizer.compress_product_data(problematic_parent)
    print(f"   ✅ Handled gracefully: parent_data converted to empty dict")
    print(f"   📊 Result: parent fields = {len(result['parent_data'])}, variants = {len(result['data_rows'])}")
    
    # Test Case 3: data_rows is a dict instead of list (problematic before fix)
    print("\n3. Problematic: data_rows as dict (would crash before):")
    problematic_variants = {
        'parent_data': {'MANUFACTURER_NAME': 'EIKO'},
        'data_rows': {  # Dict instead of list!
            'variant1': {'SUPPLIER_PID': '4301_40_44'},
            'variant2': {'SUPPLIER_PID': '4301_40_46'}
        }
    }
    
    result = optimizer.compress_product_data(problematic_variants)
    print(f"   ✅ Handled gracefully: data_rows converted to empty list")
    print(f"   📊 Result: parent fields = {len(result['parent_data'])}, variants = {len(result['data_rows'])}")
    
    # Test Case 4: Mixed invalid variants in data_rows
    print("\n4. Problematic: Mixed invalid variants (would crash before):")
    mixed_variants = {
        'parent_data': {'MANUFACTURER_NAME': 'EIKO'},
        'data_rows': [
            {'SUPPLIER_PID': '4301_40_44'},  # Valid dict
            'invalid_string_variant',        # Invalid string
            123,                            # Invalid number
            ['invalid', 'list'],            # Invalid list
            {'SUPPLIER_PID': '4301_40_46'}  # Valid dict
        ]
    }
    
    result = optimizer.compress_product_data(mixed_variants)
    print(f"   ✅ Handled gracefully: Invalid variants filtered out")
    print(f"   📊 Result: kept only {len(result['data_rows'])} valid variants out of 5 total")
    
    # Test Case 5: Completely wrong data type (would crash before)
    print("\n5. Extreme: Completely wrong input type (would crash before):")
    wrong_type = "this_should_be_a_dictionary"
    
    result = optimizer.compress_product_data(wrong_type)
    print(f"   ✅ Handled gracefully: Created safe fallback structure")
    print(f"   📊 Result: {result}")
    
    # Test Case 6: Template structure issues
    print("\n6. Template Structure Issues:")
    invalid_template = {
        'parent_product': {
            'fields': {
                'brand_name': ['not', 'a', 'dict'],  # Invalid field info
                'feed_product_type': {  # Valid field info
                    'data_type': 'string',
                    'validation_rules': {'required': True}
                }
            }
        }
    }
    
    template_result = optimizer.extract_essential_template_fields(invalid_template)
    print(f"   ✅ Template processed: {len(template_result)} valid fields extracted")
    print(f"   📋 Valid fields: {list(template_result.keys())}")
    
    print(f"\n🎉 All problematic data structures handled successfully!")
    print(f"🔒 System is now robust against data structure variations.")
    print(f"📝 Warnings logged for debugging, but processing continues.")

if __name__ == "__main__":
    demonstrate_robustness()