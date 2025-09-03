#!/usr/bin/env python3
"""Test script to verify safety filter fixes."""

import asyncio
import json
import logging
from pathlib import Path

from sku_analyzer.step5_mapping.processor import MappingProcessor
from sku_analyzer.shared.gemini_client import AIProcessingConfig


async def test_safety_fix():
    """Test the safety filter fix with real data."""
    
    # Setup logging to see safety filter messages
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor with safety-permissive configuration
    ai_config = AIProcessingConfig(
        model_name="gemini-2.0-flash-exp",
        temperature=0.0,
        max_tokens=2048,
        max_retries=3
    )
    
    processor = MappingProcessor(ai_config=ai_config)
    
    # Use existing production output
    latest_output = Path("production_output/1756744184")
    
    if not latest_output.exists():
        print("❌ Test data directory not found")
        return
    
    print("🧪 Testing safety filter fixes...\n")
    
    # Test single parent that was previously failing
    test_parent = "4301"
    step4_template = latest_output / "flat_file_analysis" / "step4_template.json"
    step2_data = latest_output / f"parent_{test_parent}" / "step2_compressed.json"
    output_dir = latest_output / f"parent_{test_parent}"
    
    if not all(p.exists() for p in [step4_template, step2_data]):
        print(f"❌ Required files missing for parent {test_parent}")
        return
    
    try:
        print(f"🔄 Processing parent {test_parent}...")
        
        result = await processor.process_parent_directory(
            parent_sku=test_parent,
            step4_template_path=step4_template,
            step2_path=step2_data,
            output_dir=output_dir
        )
        
        if result.success:
            print(f"✅ SUCCESS! Parent {test_parent} processed successfully")
            print(f"   📊 Mapped fields: {result.mapped_fields_count}")
            print(f"   🎯 Confidence: {result.confidence:.2f}")
            print(f"   ⏱️ Processing time: {result.processing_time_ms:.1f}ms")
            
            # Check output file
            output_file = Path(result.output_file)
            if output_file.exists():
                with open(output_file) as f:
                    output_data = json.load(f)
                
                print(f"   📄 Output file created: {output_file.name}")
                print(f"   🔍 Parent data keys: {list(output_data.get('parent_data', {}).keys())}")
                print(f"   🔍 Variant count: {len(output_data.get('variants', []))}")
                
                # Check for safety filter indicators
                metadata = output_data.get('metadata', {})
                if metadata.get('safety_blocked'):
                    print(f"   ⚠️ Safety blocked but handled gracefully")
                    print(f"   🚨 Safety categories: {metadata.get('safety_categories', [])}")
                elif metadata.get('simplified_mapping'):
                    print(f"   🔄 Used simplified mapping approach")
        else:
            print(f"❌ FAILED: {result.error}")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    
    # Get performance stats
    try:
        stats = processor.get_performance_stats()
        print(f"\n📈 Performance Summary:")
        print(f"   Total requests: {stats.get('ai_client', {}).get('total_requests', 0)}")
        print(f"   Success rate: {stats.get('ai_client', {}).get('success_rate', 0):.1%}")
        print(f"   Safety blocked: {stats.get('ai_client', {}).get('safety_blocked_requests', 0)}")
        print(f"   Safety block rate: {stats.get('ai_client', {}).get('safety_block_rate', 0):.1%}")
    except Exception as e:
        print(f"⚠️ Could not get performance stats: {e}")


if __name__ == "__main__":
    asyncio.run(test_safety_fix())