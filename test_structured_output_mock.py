#!/usr/bin/env python3
"""
Mock test demonstrating the structured output format and validation.

This script shows exactly what the API call would look like and validates
a mock response against our schema requirements.

Usage:
    python test_structured_output_mock.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_structured_output_validation import DataValidator
from test_real_gemini_structured_output import StructuredOutputValidator


def create_mock_structured_response() -> Dict[str, Any]:
    """Create a mock structured response matching our schema.
    
    Returns:
        Mock response with all required fields
    """
    # Create mock parent data with all 14 required fields
    mock_parent_data = {
        "age_range_description": "Adult",
        "bottoms_size_class": "Bundweite & Schrittlänge",
        "bottoms_size_system": "DE / NL / SE / PL",
        "brand_name": "FHB",
        "country_of_origin": "Germany",
        "department_name": "Workwear",
        "external_product_id_type": "EAN",
        "fabric_type": "Cotton Corduroy",
        "feed_product_type": "pants",
        "item_name": "Professional Work Pants",
        "main_image_url": "https://example.com/image.jpg",
        "outer_material_type": "Cotton",
        "recommended_browse_nodes": "16310091",
        "target_gender": "Männlich"
    }
    
    # Create mock variants (28 variants as expected from the real data)
    mock_variants = []
    sizes = ["46", "48", "50", "52", "54", "56", "58"]
    colors = ["black", "navy", "brown", "grey"]
    
    variant_counter = 1
    for color in colors:
        for size in sizes:
            variant = {
                "color_map": color.title(),
                "color_name": color,
                "external_product_id": f"403{variant_counter:04d}",
                "item_sku": f"FHB-41282-{color.upper()}-{size}",
                "list_price_with_tax": "89.95",
                "quantity": "10",
                "size_map": size,
                "size_name": size,
                "standard_price": "75.42"
            }
            mock_variants.append(variant)
            variant_counter += 1
    
    return {
        "parent_data": mock_parent_data,
        "variants": mock_variants
    }


def show_api_call_example():
    """Show what the actual API call would look like."""
    print("🔧 GEMINI API CALL EXAMPLE")
    print("=" * 60)
    
    # Load real data for the prompt
    data_validator = DataValidator()
    if data_validator.validate_data_files()["success"]:
        data_validator.validate_data_structure()
        
        # Show prompt creation
        from sku_analyzer.prompts.mapping_prompts import MappingPromptManager
        prompt_manager = MappingPromptManager()
        
        context = {
            "parent_sku": "41282",
            "mandatory_fields": data_validator.loaded_data["step3_mandatory"],
            "product_data": data_validator.loaded_data["step2_compressed"],
            "template_structure": data_validator.loaded_data.get("step4_template", {})
        }
        
        prompt = prompt_manager.render_mapping_prompt(context)
        
        print(f"📤 PROMPT LENGTH: {len(prompt)} characters")
        print(f"📤 PROMPT PREVIEW (first 300 chars):")
        print("-" * 40)
        print(prompt[:300] + "...")
        print("-" * 40)
        
        print("\n🔗 API CONFIGURATION:")
        print("   Model: gemini-2.5-flash")
        print("   Temperature: 0.3")
        print("   Response Format: application/json")
        print("   Structured Output: ENABLED")
        print("   Schema Fields: 14 parent + 9 variant fields")
        
    else:
        print("❌ Could not load production data for example")


def validate_mock_response():
    """Validate the mock response against our requirements."""
    print("\n🧪 MOCK RESPONSE VALIDATION")
    print("=" * 60)
    
    # Create mock response
    mock_response = create_mock_structured_response()
    
    # Validate using our validator
    validator = StructuredOutputValidator()
    
    # Validate structure
    structure_valid = validator.validate_response_structure(mock_response)
    parent_valid = validator.validate_parent_data(mock_response["parent_data"]) if structure_valid else False
    variants_valid = validator.validate_variants_data(mock_response["variants"], 28) if structure_valid else False
    
    # Print validation results
    print(f"✅ Structure Valid: {structure_valid}")
    print(f"✅ Parent Data Valid: {parent_valid} ({len(mock_response['parent_data'])} fields)")
    print(f"✅ Variants Valid: {variants_valid} ({len(mock_response['variants'])} variants)")
    
    # Show validation summary
    validation_summary = validator.get_validation_summary()
    if validation_summary["error_count"] > 0:
        print(f"\n❌ ERRORS ({validation_summary['error_count']}):")
        for error in validation_summary["errors"]:
            print(f"   - {error}")
    
    if validation_summary["warning_count"] > 0:
        print(f"\n⚠️  WARNINGS ({validation_summary['warning_count']}):")
        for warning in validation_summary["warnings"]:
            print(f"   - {warning}")
    
    return validation_summary["is_valid"]


def show_expected_response_structure():
    """Show the exact structure we expect from the API."""
    print("\n📋 EXPECTED RESPONSE STRUCTURE")
    print("=" * 60)
    
    mock_response = create_mock_structured_response()
    
    print("JSON Structure:")
    print("{")
    print('  "parent_data": {')
    parent_fields = list(mock_response["parent_data"].keys())
    for i, field in enumerate(parent_fields):
        comma = "," if i < len(parent_fields) - 1 else ""
        example_value = mock_response["parent_data"][field]
        print(f'    "{field}": "{example_value}"{comma}')
    print("  },")
    print('  "variants": [')
    print("    {")
    variant_fields = list(mock_response["variants"][0].keys())
    for i, field in enumerate(variant_fields):
        comma = "," if i < len(variant_fields) - 1 else ""
        example_value = mock_response["variants"][0][field]
        print(f'      "{field}": "{example_value}"{comma}')
    print("    },")
    print("    ... (28 total variants)")
    print("  ]")
    print("}")
    
    # Show field counts
    print(f"\n📊 FIELD REQUIREMENTS:")
    print(f"   Parent Data: {len(mock_response['parent_data'])} fields (all required)")
    print(f"   Each Variant: {len(mock_response['variants'][0])} fields (all required)")
    print(f"   Total Variants: {len(mock_response['variants'])} (from real data)")
    print(f"   Total Fields: {len(mock_response['parent_data']) + len(mock_response['variants']) * len(mock_response['variants'][0])}")


def main():
    """Main test execution."""
    print("🔬 Structured Output Mock Test & API Example")
    print("=" * 80)
    
    # Show what the API call looks like
    show_api_call_example()
    
    # Validate mock response
    mock_valid = validate_mock_response()
    
    # Show expected structure
    show_expected_response_structure()
    
    # Final summary
    print("\n" + "=" * 80)
    print("MOCK TEST SUMMARY")
    print("=" * 80)
    
    if mock_valid:
        print("✅ Mock response validation PASSED")
        print("🚀 Schema and validation logic working correctly")
        print("\n📝 TO RUN REAL API TEST:")
        print("   1. Set GOOGLE_API_KEY environment variable")
        print("   2. Run: python test_real_gemini_structured_output.py")
        print("\n🎯 EXPECTED RESULTS:")
        print("   - Response time: <10 seconds")
        print("   - Parent data: 14 fields populated")
        print("   - Variants: 28 variants with 9 fields each")
        print("   - JSON format: Valid structured output")
        print("   - All required fields: Present and non-empty")
    else:
        print("❌ Mock response validation FAILED")
        print("🔧 Fix validation logic before API testing")
    
    print("=" * 80)
    
    return 0 if mock_valid else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)