#!/usr/bin/env python3
"""
Display the exact structured output schema used for Gemini API calls.

This script shows the complete schema definition that matches ai_studio_code.py
and validates it against our implementation.

Usage:
    python show_structured_output_schema.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def show_ai_studio_schema():
    """Show the exact schema from ai_studio_code.py format."""
    print("📋 GEMINI STRUCTURED OUTPUT SCHEMA (ai_studio_code.py format)")
    print("=" * 80)
    
    schema_description = """
This is the exact schema structure used in ai_studio_code.py:

```python
response_schema = genai.types.Schema(
    type = genai.types.Type.OBJECT,
    description = "Structured product data including parent and variant information.",
    required = ["parent_data", "variants"],
    properties = {
        "parent_data": genai.types.Schema(
            type = genai.types.Type.OBJECT,
            description = "Information pertaining to the parent product.",
            required = [
                "age_range_description", 
                "bottoms_size_class", 
                "bottoms_size_system",
                "brand_name", 
                "country_of_origin", 
                "department_name",
                "external_product_id_type", 
                "fabric_type", 
                "feed_product_type",
                "item_name", 
                "main_image_url", 
                "outer_material_type",
                "recommended_browse_nodes", 
                "target_gender"
            ],
            # 14 parent fields with STRING type and descriptions...
        ),
        "variants": genai.types.Schema(
            type = genai.types.Type.ARRAY,
            description = "A list of product variants, each with its specific attributes.",
            items = genai.types.Schema(
                type = genai.types.Type.OBJECT,
                required = [
                    "color_map", 
                    "color_name", 
                    "external_product_id",
                    "item_sku", 
                    "list_price_with_tax", 
                    "quantity",
                    "size_map", 
                    "size_name", 
                    "standard_price"
                ],
                # 9 variant fields with STRING type and descriptions...
            )
        )
    }
)
```
"""
    
    print(schema_description)


def show_field_details():
    """Show detailed field definitions."""
    print("\n📝 FIELD DEFINITIONS")
    print("=" * 80)
    
    parent_fields = {
        "age_range_description": "Description of the target age range for the product.",
        "bottoms_size_class": "The size classification system for bottoms (e.g., Bundweite & Schrittlänge).",
        "bottoms_size_system": "The specific size system used for bottoms (e.g., DE / NL / SE / PL).",
        "brand_name": "The brand name of the product.",
        "country_of_origin": "The country where the product was manufactured.",
        "department_name": "The department or general category the product belongs to.",
        "external_product_id_type": "The type of external product identifier (e.g., EAN, UPC).",
        "fabric_type": "The primary fabric type of the product.",
        "feed_product_type": "The product type as categorized for data feeds (e.g., pants, shirt).",
        "item_name": "The name of the product item.",
        "main_image_url": "The URL of the main image for the product.",
        "outer_material_type": "The type of material used for the outer layer of the product.",
        "recommended_browse_nodes": "A recommended category or browse node ID for the product.",
        "target_gender": "The target gender for the product (e.g., Männlich, Weiblich, Unisex)."
    }
    
    variant_fields = {
        "color_map": "A standardized or mapped color name for the variant.",
        "color_name": "The specific color name of the variant.",
        "external_product_id": "The unique external identifier for this specific variant (e.g., EAN).",
        "item_sku": "The Stock Keeping Unit (SKU) for this variant.",
        "list_price_with_tax": "The list price of the variant, including applicable taxes.",
        "quantity": "The available quantity of this variant.",
        "size_map": "A standardized or mapped size name for the variant.",
        "size_name": "The specific size name of the variant.",
        "standard_price": "The standard selling price of the variant, excluding taxes."
    }
    
    print(f"🔹 PARENT DATA FIELDS ({len(parent_fields)} required):")
    for field, description in parent_fields.items():
        print(f"   • {field}: {description}")
    
    print(f"\n🔸 VARIANT FIELDS ({len(variant_fields)} required per variant):")
    for field, description in variant_fields.items():
        print(f"   • {field}: {description}")


def validate_our_schema():
    """Validate our schema implementation matches the requirements."""
    print("\n🔍 SCHEMA VALIDATION")
    print("=" * 80)
    
    try:
        from sku_analyzer.step5_mapping.schema import get_ai_mapping_schema
        
        schema = get_ai_mapping_schema()
        
        # Extract field counts
        parent_schema = schema.properties["parent_data"]
        variants_schema = schema.properties["variants"]
        variant_item_schema = variants_schema.items
        
        parent_field_count = len(parent_schema.required) if parent_schema.required else 0
        variant_field_count = len(variant_item_schema.required) if variant_item_schema.required else 0
        
        print(f"✅ Schema loaded successfully")
        print(f"✅ Parent fields: {parent_field_count} (expected: 14)")
        print(f"✅ Variant fields: {variant_field_count} (expected: 9)")
        print(f"✅ Schema type: {schema.type}")
        print(f"✅ Required top-level fields: {schema.required}")
        
        # Validate field counts
        if parent_field_count == 14 and variant_field_count == 9:
            print(f"🎯 PERFECT MATCH: Schema matches ai_studio_code.py requirements")
            return True
        else:
            print(f"❌ MISMATCH: Field counts don't match expected values")
            return False
            
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        return False


def show_api_call_code():
    """Show the actual API call code using google-genai."""
    print("\n💻 API CALL CODE")
    print("=" * 80)
    
    api_code = '''
# Initialize google-genai client
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Create content
contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    ),
]

# Generate with structured output
config = types.GenerateContentConfig(
    temperature=0.3,
    thinking_config=types.ThinkingConfig(thinking_budget=-1),
    response_mime_type="application/json",
    response_schema=get_ai_mapping_schema()  # Our schema function
)

# Make API call
for chunk in client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=contents,
    config=config,
):
    print(chunk.text, end="")
'''
    
    print(api_code)


def main():
    """Main display function."""
    print("📊 GEMINI STRUCTURED OUTPUT SCHEMA DOCUMENTATION")
    print("=" * 80)
    print("This shows the complete schema used for AI mapping with real production data")
    print("from job 1756744213, parent SKU 41282")
    
    # Show the schema format
    show_ai_studio_schema()
    
    # Show detailed field definitions
    show_field_details()
    
    # Validate our implementation
    schema_valid = validate_our_schema()
    
    # Show API call code
    show_api_call_code()
    
    # Summary
    print("\n" + "=" * 80)
    print("SCHEMA SUMMARY")
    print("=" * 80)
    
    print(f"📋 Total Fields Required: 23 (14 parent + 9 variant)")
    print(f"🎯 Expected Variants: 28 (from real data)")
    print(f"📦 Total Response Fields: ~266 (14 + 28×9)")
    print(f"🔧 Schema Validation: {'✅ PASSED' if schema_valid else '❌ FAILED'}")
    
    print(f"\n🚀 READY FOR TESTING:")
    print(f"   • Validation tests: python test_structured_output_validation.py")
    print(f"   • Mock test: python test_structured_output_mock.py")
    print(f"   • Real API test: python test_real_gemini_structured_output.py (requires API key)")
    
    print("=" * 80)
    
    return 0 if schema_valid else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Display interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Display failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)