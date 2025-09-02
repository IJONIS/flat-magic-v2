"""Simple AI mapping test without complex imports."""

import asyncio
import json
import os
from pathlib import Path


async def test_basic_ai_mapping():
    """Test basic AI mapping functionality."""
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable required")
        print("   Add to .env file: GOOGLE_API_KEY=your_api_key_here")
        return
    
    print("🧪 Testing basic AI mapping functionality")
    
    # Test data paths
    output_dir = Path("production_output/1756744145")
    step3_file = output_dir / "flat_file_analysis" / "step3_mandatory_fields.json"
    step2_file = output_dir / "parent_4301" / "step2_compressed.json"
    
    if not step3_file.exists() or not step2_file.exists():
        print("❌ Required test files not found")
        return
    
    # Load test data
    with step3_file.open('r') as f:
        mandatory_fields = json.load(f)
    
    with step2_file.open('r') as f:
        product_data = json.load(f)
    
    print(f"✅ Loaded test data:")
    print(f"   Mandatory fields: {len(mandatory_fields)} fields")
    print(f"   Product data keys: {len(product_data)} keys")
    
    # Test Pydantic AI import
    try:
        from pydantic_ai_slim import Agent
        from pydantic_ai_slim.models.google import GoogleModel
        from pydantic import BaseModel
        
        print("✅ Pydantic AI imports successful")
        
        # Simple mapping result model
        class SimpleMappingResult(BaseModel):
            parent_sku: str
            mapped_count: int
            confidence: float
            notes: str
        
        # Create agent
        model = GoogleModel("gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
        agent = Agent(
            model=model,
            result_type=SimpleMappingResult,
            system_prompt="You are a product data mapping expert. Map product data to Amazon fields."
        )
        
        # Simple test prompt
        prompt = f"""
        Map this product data (parent SKU 4301) to mandatory Amazon fields.
        
        Available mandatory fields (first 3):
        {list(mandatory_fields.keys())[:3]}
        
        Product data sample (first 5 keys):
        {dict(list(product_data.items())[:5])}
        
        Return a simple mapping assessment.
        """
        
        print("🔄 Running AI mapping test...")
        
        result = await agent.run(prompt)
        print(f"✅ AI mapping completed!")
        print(f"   SKU: {result.data.parent_sku}")
        print(f"   Mapped: {result.data.mapped_count}")
        print(f"   Confidence: {result.data.confidence}")
        print(f"   Notes: {result.data.notes}")
        
    except Exception as e:
        print(f"❌ AI mapping test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_basic_ai_mapping())