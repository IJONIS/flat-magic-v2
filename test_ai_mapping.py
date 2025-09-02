"""Test script for AI mapping functionality."""

import asyncio
import os
from pathlib import Path

# Direct import of AI components to avoid pandas dependency
import sys
sys.path.append('sku_analyzer')


async def test_ai_mapping_4301():
    """Test AI mapping with parent 4301."""
    
    # Set up test environment
    output_dir = Path("production_output/1756744145")  # Use existing test data
    
    if not output_dir.exists():
        print("❌ Test data directory not found")
        return
    
    # Verify required files exist
    step3_file = output_dir / "flat_file_analysis" / "step3_mandatory_fields.json"
    step2_file = output_dir / "parent_4301" / "step2_compressed.json"
    
    if not step3_file.exists():
        print(f"❌ Missing: {step3_file}")
        return
    
    if not step2_file.exists():
        print(f"❌ Missing: {step2_file}")
        return
    
    print("🧪 Testing AI mapping with parent 4301")
    print(f"   Input dir: {output_dir}")
    print(f"   Step3: {step3_file}")
    print(f"   Step2: {step2_file}")
    
    # Direct import to avoid module loading issues
    from ai_mapping.models import AIProcessingConfig
    from ai_mapping.processor import AIMappingProcessor
    
    # Create config
    config = AIProcessingConfig(
        model_name="gemini-2.5-flash",
        temperature=0.1,
        timeout_seconds=45,
        batch_size=1,
        max_concurrent=1
    )
    
    # Test processor
    processor = AIMappingProcessor(config)
    
    try:
        result = await processor.process_parent_directory(
            parent_sku="4301",
            step3_path=step3_file,
            step2_path=step2_file,
            output_dir=output_dir / "parent_4301"
        )
        
        if result["success"]:
            print("✅ AI mapping test successful!")
            print(f"   Mapped fields: {result['mapped_fields_count']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
            print(f"   Output file: {result['output_file']}")
        else:
            print(f"❌ AI mapping test failed: {result.get('error')}")
    
    except Exception as e:
        print(f"❌ Test error: {e}")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable required")
        print("   Add to .env file: GOOGLE_API_KEY=your_api_key_here")
        exit(1)
    
    print("🚀 Starting AI Mapping Test")
    asyncio.run(test_ai_mapping_4301())