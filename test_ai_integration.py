"""Test AI mapping integration without complex dependencies."""

import asyncio
import json
import os
from pathlib import Path


def test_integration_point():
    """Test the AI mapping integration point."""
    
    print("🧪 Testing AI Mapping Integration Point")
    
    # Import integration module
    from sku_analyzer.ai_mapping.integration_point import AIMapingIntegration
    
    # Test with existing data
    output_dir = Path("production_output/1756744145")
    
    if not output_dir.exists():
        print(f"❌ Test directory not found: {output_dir}")
        return
    
    # Initialize integration
    ai_integration = AIMapingIntegration(enable_ai=True)
    
    # Process AI mapping
    result = ai_integration.process_ai_mapping_step(
        output_dir=output_dir,
        starting_parent="4301"
    )
    
    print("✅ AI Integration Test Results:")
    if result.get("ai_mapping_completed"):
        summary = result["summary"]
        performance = result["performance"]
        
        print(f"   Total parents: {summary['total_parents']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Success rate: {summary['success_rate']:.1%}")
        print(f"   Avg confidence: {summary['average_confidence']:.2f}")
        print(f"   Total time: {performance['total_processing_time_ms']:.1f}ms")
        print(f"   Avg time/parent: {performance['average_time_per_parent']:.1f}ms")
        
        # Show first few results
        for i, detail in enumerate(result["details"][:3]):
            if detail.get("success"):
                print(f"   Parent {detail['parent_sku']}: "
                      f"{detail['mapped_fields_count']} mapped, "
                      f"confidence {detail['confidence']:.2f}")
    else:
        print(f"   Status: {result}")


def create_workflow_demo():
    """Create demo script for AI-enhanced workflow."""
    
    demo_script = '''"""Demo script for AI-enhanced workflow."""

import asyncio
from pathlib import Path
from sku_analyzer.ai_mapping.integration_point import AIMapingIntegration


async def demo_ai_workflow():
    """Demonstrate AI-enhanced workflow."""
    
    print("🚀 AI-Enhanced SKU Analysis Workflow Demo")
    
    # Find latest output directory
    output_base = Path("production_output")
    if not output_base.exists():
        print("❌ No production output found")
        return
    
    # Get most recent output directory
    output_dirs = sorted([d for d in output_base.iterdir() if d.is_dir()], 
                        key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not output_dirs:
        print("❌ No output directories found")
        return
    
    latest_output = output_dirs[0]
    print(f"📁 Using output directory: {latest_output.name}")
    
    # Initialize AI integration
    ai_integration = AIMapingIntegration(enable_ai=True)
    
    # Process with AI mapping
    print("🤖 Running AI mapping step...")
    result = ai_integration.process_ai_mapping_step(
        output_dir=latest_output,
        starting_parent="4301"
    )
    
    # Display results
    if result.get("ai_mapping_completed"):
        print("✅ AI Mapping Completed Successfully!")
        
        summary = result["summary"]
        print(f"   📊 Parents processed: {summary['total_parents']}")
        print(f"   ✅ Successful: {summary['successful']}")
        print(f"   📈 Success rate: {summary['success_rate']:.1%}")
        print(f"   🎯 Average confidence: {summary['average_confidence']:.2f}")
        
        performance = result["performance"]
        print(f"   ⏱️  Total time: {performance['total_processing_time_ms']:.1f}ms")
        print(f"   📋 Processing stats: {performance['processing_stats']}")
        
        # List output files
        successful_results = [r for r in result["details"] if r.get("success")]
        print(f"\\n📁 Generated AI mapping files:")
        for res in successful_results[:5]:  # Show first 5
            print(f"   {res['output_file']}")
        
        if len(successful_results) > 5:
            print(f"   ... and {len(successful_results) - 5} more files")
    
    else:
        print(f"❌ AI mapping failed: {result}")


if __name__ == "__main__":
    asyncio.run(demo_ai_workflow())
'''
    
    with open("demo_ai_workflow.py", "w") as f:
        f.write(demo_script)
    
    print("✅ Created demo_ai_workflow.py")


if __name__ == "__main__":
    test_integration_point()
    create_workflow_demo()