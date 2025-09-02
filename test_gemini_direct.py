"""Direct Gemini API test for AI mapping."""

import asyncio
import json
import os
from pathlib import Path


async def test_gemini_direct():
    """Test direct Gemini API integration."""
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable required")
        print("   Add to .env file: GOOGLE_API_KEY=your_api_key_here")
        return
    
    print("🧪 Testing direct Gemini API integration")
    
    try:
        import google.generativeai as genai
        from pydantic import BaseModel
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Create model with optimal settings
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=4096,
                response_mime_type="application/json"
            )
        )
        
        print("✅ Gemini model configured")
        
        # Load test data
        output_dir = Path("production_output/1756744145")
        step3_file = output_dir / "flat_file_analysis" / "step3_mandatory_fields.json"
        step2_file = output_dir / "parent_4301" / "step2_compressed.json"
        
        with step3_file.open('r') as f:
            mandatory_fields = json.load(f)
        
        with step2_file.open('r') as f:
            product_data = json.load(f)
        
        # Create mapping prompt
        prompt = f"""
Map this German Amazon product data to mandatory fields with high accuracy.

MANDATORY FIELDS TO MAP:
{json.dumps(dict(list(mandatory_fields.items())[:3]), indent=2)}

PRODUCT DATA:
{json.dumps(dict(list(product_data['parent_data'].items())[:10]), indent=2)}

Return JSON with this structure:
{{
  "parent_sku": "4301",
  "mapped_fields": [
    {{
      "source_field": "MANUFACTURER_NAME",
      "target_field": "brand_name", 
      "mapped_value": "EIKO",
      "confidence": 0.95,
      "reasoning": "Direct brand name mapping"
    }}
  ],
  "unmapped_mandatory": ["list", "of", "unmapped"],
  "overall_confidence": 0.87,
  "processing_notes": "Mapping summary"
}}
"""
        
        print("🔄 Sending request to Gemini...")
        
        # Make request
        response = model.generate_content(prompt)
        
        if response.text:
            print("✅ Gemini response received!")
            print("📄 Raw response:")
            print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
            
            # Try to parse JSON
            try:
                # Handle markdown code blocks
                content = response.text.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                
                result = json.loads(content)
                
                print("\n✅ JSON parsing successful!")
                print(f"   Parent SKU: {result.get('parent_sku')}")
                print(f"   Mapped fields: {len(result.get('mapped_fields', []))}")
                print(f"   Confidence: {result.get('overall_confidence', 0.0)}")
                
                # Save test result
                test_output = output_dir / "parent_4301" / "step3_ai_mapping_test.json"
                test_output.parent.mkdir(parents=True, exist_ok=True)
                
                with test_output.open('w') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Test result saved: {test_output}")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print("Raw content for debugging:")
                print(repr(response.text))
        else:
            print("❌ No response text received")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_gemini_direct())