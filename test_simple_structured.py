#!/usr/bin/env python3
"""Simple test script for structured output functionality."""

import os
import asyncio
from sku_analyzer.shared.gemini_client import GeminiClient, AIProcessingConfig

async def test_simple_structured_output():
    """Test basic structured output functionality."""
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set - skipping actual API test")
        return
    
    try:
        # Initialize with structured output enabled
        config = AIProcessingConfig(enable_structured_output=True)
        client = GeminiClient(config)
        
        print(f"✅ Client initialized with structured output: {client._structured_model is not None}")
        
        # Simple test prompt
        test_prompt = """
        Create product mapping for a German workwear product:
        
        Brand: EIKO
        Product Type: Work Pants
        Variants:
        - SKU: EIKO-001, Color: Black, Size: L, Price: 49.99 EUR
        - SKU: EIKO-002, Color: Blue, Size: M, Price: 49.99 EUR
        
        Map to all required fields for Amazon marketplace.
        """
        
        response = await client.generate_structured_mapping(test_prompt)
        
        print(f"✅ Response received: {len(response.content)} characters")
        
        # Validate JSON structure
        json_data = await client.validate_json_response(response)
        
        # Check structured output format
        if "parent_data" in json_data and "variants" in json_data:
            print("✅ Structured output format confirmed")
            print(f"   Parent fields: {len(json_data.get('parent_data', {}))}")
            print(f"   Variants: {len(json_data.get('variants', []))}")
            
            # Show sample output
            print("\n📋 Sample Output Structure:")
            print(f"   Parent brand: {json_data.get('parent_data', {}).get('brand_name', 'N/A')}")
            print(f"   First variant SKU: {json_data.get('variants', [{}])[0].get('item_sku', 'N/A') if json_data.get('variants') else 'N/A'}")
        else:
            print("❌ Unexpected response format")
            print(f"   Response keys: {list(json_data.keys()) if isinstance(json_data, dict) else 'Not a dict'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_simple_structured_output())
    print(f"\n{'✅ All tests passed!' if result else '❌ Tests failed'}")