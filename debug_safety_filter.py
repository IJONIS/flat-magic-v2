#!/usr/bin/env python3
"""Debug script to identify Gemini safety filter triggers."""

import asyncio
import json
import os
from pathlib import Path
from sku_analyzer.shared.gemini_client import GeminiClient, AIProcessingConfig


async def test_safety_filters():
    """Test different prompt variations to identify safety filter triggers."""
    
    # Initialize client
    config = AIProcessingConfig(
        model_name="gemini-2.0-flash-exp",  # Try different model
        temperature=0.1,
        max_tokens=4096
    )
    
    client = GeminiClient(config=config)
    
    # Test prompts of increasing complexity
    test_prompts = [
        # Basic test
        """You are a product data mapper. Return this JSON:
{"test": "success", "status": "working"}""",
        
        # Simple German content
        """Map this German product data:
Product: "Bundhose"
Brand: "EIKO" 
Size: "44"
Color: "Schwarz"

Return JSON:
{"brand_name": "EIKO", "size_name": "44", "color_name": "Schwarz"}""",
        
        # With clothing descriptions
        """Map product data to Amazon fields:

PRODUCT DATA:
- Name: "ALLER - Bundhose - Genuacord - Schwarz"
- Description: "Diese Hose ist praktisch für Arbeit und Freizeit"
- Brand: "EIKO"
- Type: "Bundhose"

Return JSON:
{"brand_name": "EIKO", "item_name": "ALLER Bundhose", "feed_product_type": "pants"}""",
        
        # With URLs (potential trigger)
        """Map this product:
Image URL: "https://blob.redpim.de/company-53e006db-2b74-4ce1-5a4d-08dca19c0e21/mimes/1441585_normal.jpg"
Product: "Bundhose"

Return JSON: {"main_image_url": "https://blob.redpim.de/..."}""",
        
        # With full description (potential trigger)
        """Map this German product:
Description: "Diese Hose ist ein Dauerbrenner, weil sie einfach praktisch ist. Egal ob man es Manchester oder Cord nennt. Mit dieser Hose ist man zur Arbeit, in der Freizeit oder zum Kirchgang immer gut gekleidet. Zumeist ältere Herren und Skater lieben die weiten Beine."

Return JSON: {"item_name": "Practical work pants"}"""
    ]
    
    print("🔍 Testing Gemini safety filters...\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}: ", end="", flush=True)
        try:
            response = await client.generate_mapping(
                prompt=prompt,
                operation_name=f"safety_test_{i}"
            )
            
            print("✅ SUCCESS")
            print(f"   Response: {response.content[:100]}...")
            if response.safety_ratings:
                print(f"   Safety ratings: {response.safety_ratings}")
            print(f"   Finish reason: {response.finish_reason}\n")
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            print(f"   Error type: {type(e).__name__}\n")
    
    return client.get_performance_summary()


async def test_actual_prompt():
    """Test the actual prompt being used in production."""
    
    # Load real data
    try:
        parent_path = Path("production_output/1756744184/parent_4301/step2_compressed.json")
        template_path = Path("production_output/1756744184/flat_file_analysis/step4_template.json")
        
        if not parent_path.exists() or not template_path.exists():
            print("❌ Test data files not found")
            return
        
        with open(parent_path) as f:
            product_data = json.load(f)
        
        with open(template_path) as f:
            template_data = json.load(f)
        
        # Extract mandatory fields
        template_structure = template_data.get("template_structure", {})
        parent_fields = template_structure.get("parent_product", {}).get("fields", {})
        variant_fields = template_structure.get("child_variants", {}).get("fields", {})
        
        # Combine all mandatory fields
        mandatory_fields = {**parent_fields, **variant_fields}
        
        # Limit data for testing
        limited_mandatory = dict(list(mandatory_fields.items())[:5])
        limited_parent = dict(list(product_data.get('parent_data', {}).items())[:5])
        limited_variants = product_data.get('data_rows', [])[:2]
        
        # Create minimal prompt
        prompt = f"""You are a German Amazon product mapper.

TASK: Map this product data to Amazon fields.

MANDATORY FIELDS (5 shown):
{json.dumps(limited_mandatory, indent=2, ensure_ascii=False)}

PARENT DATA (5 shown):
{json.dumps(limited_parent, indent=2, ensure_ascii=False)}

VARIANTS (2 shown):
{json.dumps(limited_variants, indent=2, ensure_ascii=False)}

Return valid JSON:
{{
  "parent_sku": "4301",
  "parent_data": {{"brand_name": "EIKO"}},
  "variance_data": {{"variant_1": {{"item_sku": "4301_40_44"}}}}
}}"""
        
        print("🧪 Testing actual production prompt (simplified)...\n")
        
        client = GeminiClient(config=AIProcessingConfig(
            model_name="gemini-2.0-flash-exp",
            temperature=0.0,  # More deterministic
            max_tokens=2048
        ))
        
        response = await client.generate_mapping(prompt, operation_name="actual_prompt_test")
        
        print("✅ Production prompt test SUCCESS!")
        print(f"Response: {response.content[:200]}...")
        print(f"Finish reason: {response.finish_reason}")
        if response.safety_ratings:
            print(f"Safety ratings: {response.safety_ratings}")
        
    except Exception as e:
        print(f"❌ Production prompt test FAILED: {e}")
        print(f"Error type: {type(e).__name__}")


async def main():
    """Run all safety filter tests."""
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set")
        return
    
    print("🔑 API Key found\n")
    
    # Run basic safety tests
    await test_safety_filters()
    
    print("=" * 60)
    
    # Test actual production prompt
    await test_actual_prompt()


if __name__ == "__main__":
    asyncio.run(main())