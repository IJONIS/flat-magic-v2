"""Simple Gemini integration using available packages."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class SimpleGeminiMapper:
    """Simple Gemini-based AI mapper without complex dependencies."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize simple mapper.
        
        Args:
            api_key: Google API key (from env if not provided)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required")
        
        self.logger = logging.getLogger(__name__)
        
        # Simple stats tracking
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0.0
        }
    
    async def map_product_data(
        self,
        parent_sku: str,
        mandatory_fields: Dict[str, Any],
        product_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map product data using Gemini API.
        
        Args:
            parent_sku: Parent SKU identifier
            mandatory_fields: Required fields schema
            product_data: Source product data
            
        Returns:
            Mapping result dictionary
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create mapping prompt
            prompt = self._create_mapping_prompt(
                parent_sku, mandatory_fields, product_data
            )
            
            # Make API request
            result = await self._call_gemini_api(prompt)
            
            # Update stats
            self.stats["total_processed"] += 1
            self.stats["successful"] += 1
            self.stats["total_time"] += asyncio.get_event_loop().time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Mapping failed for {parent_sku}: {e}")
            self.stats["total_processed"] += 1
            self.stats["failed"] += 1
            
            return {
                "parent_sku": parent_sku,
                "mapped_fields": [],
                "unmapped_mandatory": list(mandatory_fields.keys()),
                "overall_confidence": 0.0,
                "processing_notes": f"Processing failed: {e}"
            }
    
    def _create_mapping_prompt(
        self,
        parent_sku: str,
        mandatory_fields: Dict[str, Any],
        product_data: Dict[str, Any]
    ) -> str:
        """Create mapping prompt for Gemini.
        
        Args:
            parent_sku: Parent SKU
            mandatory_fields: Required fields
            product_data: Source data
            
        Returns:
            Formatted prompt string
        """
        # Limit data size for prompt
        limited_fields = dict(list(mandatory_fields.items())[:10])
        limited_product = dict(list(product_data.get('parent_data', {}).items())[:15])
        
        return f"""You are an expert at mapping German Amazon product data to mandatory fields.

TASK: Map product data for SKU {parent_sku} to mandatory Amazon fields.

MANDATORY FIELDS (showing first 10):
{json.dumps(limited_fields, indent=2, ensure_ascii=False)}

PRODUCT DATA (showing first 15):
{json.dumps(limited_product, indent=2, ensure_ascii=False)}

INSTRUCTIONS:
1. Map source fields to mandatory fields based on semantic meaning
2. Consider German language context and Amazon marketplace requirements
3. Only map with confidence >70%
4. Provide clear reasoning for each mapping

REQUIRED JSON OUTPUT:
{{
  "parent_sku": "{parent_sku}",
  "mapped_fields": [
    {{
      "source_field": "MANUFACTURER_NAME",
      "target_field": "brand_name",
      "mapped_value": "EIKO",
      "confidence": 0.95,
      "reasoning": "Direct manufacturer to brand mapping"
    }}
  ],
  "unmapped_mandatory": ["field1", "field2"],
  "overall_confidence": 0.87,
  "processing_notes": "Mapped X of Y mandatory fields"
}}

Map the product data now:"""
    
    async def _call_gemini_api(self, prompt: str) -> Dict[str, Any]:
        """Call Gemini API using subprocess for compatibility.
        
        Args:
            prompt: Request prompt
            
        Returns:
            Parsed JSON response
        """
        # Create temporary script for API call
        script = f'''
import os
import json
import google.generativeai as genai

genai.configure(api_key="{self.api_key}")

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=genai.types.GenerationConfig(
        temperature=0.1,
        max_output_tokens=4096,
        response_mime_type="application/json"
    )
)

prompt = """{prompt}"""

try:
    response = model.generate_content(prompt)
    if response.text:
        print(response.text)
    else:
        print('{{"error": "No response text"}}')
except Exception as e:
    print(f'{{"error": "{str(e)}"}}')
'''
        
        # Write script to temporary file
        script_file = Path("temp_gemini_call.py")
        script_file.write_text(script)
        
        try:
            # Execute script
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                response_text = stdout.decode().strip()
                
                # Parse response
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                return json.loads(response_text)
            else:
                raise RuntimeError(f"Gemini call failed: {stderr.decode()}")
                
        finally:
            # Clean up
            script_file.unlink(missing_ok=True)
    
    async def process_parent_4301(self, output_dir: Path) -> Dict[str, Any]:
        """Process parent 4301 specifically.
        
        Args:
            output_dir: Output directory with test data
            
        Returns:
            Processing result
        """
        print(f"🎯 Processing parent 4301 in {output_dir}")
        
        # Load input files
        step3_file = output_dir / "flat_file_analysis" / "step3_mandatory_fields.json"
        step2_file = output_dir / "parent_4301" / "step2_compressed.json"
        
        if not step3_file.exists():
            return {"success": False, "error": f"Missing {step3_file}"}
        if not step2_file.exists():
            return {"success": False, "error": f"Missing {step2_file}"}
        
        with step3_file.open('r') as f:
            mandatory_fields = json.load(f)
        
        with step2_file.open('r') as f:
            product_data = json.load(f)
        
        # Perform AI mapping
        result = await self.map_product_data("4301", mandatory_fields, product_data)
        
        # Save result
        output_file = output_dir / "parent_4301" / "step3_ai_mapping.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with output_file.open('w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "parent_sku": "4301",
            "mapped_fields": len(result.get("mapped_fields", [])),
            "confidence": result.get("overall_confidence", 0.0),
            "output_file": str(output_file),
            "stats": self.stats.copy()
        }


async def main():
    """Main test function."""
    print("🚀 Simple AI Mapping Test")
    
    try:
        mapper = SimpleGeminiMapper()
        
        # Test with existing data
        output_dir = Path("production_output/1756744145")
        result = await mapper.process_parent_4301(output_dir)
        
        if result["success"]:
            print("✅ AI mapping test successful!")
            print(f"   Mapped fields: {result['mapped_fields']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Output: {result['output_file']}")
        else:
            print(f"❌ Test failed: {result['error']}")
    
    except Exception as e:
        print(f"❌ Main test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())