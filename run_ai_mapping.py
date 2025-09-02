#!/usr/bin/env python3
"""Production script for AI mapping operations."""

import asyncio
import logging
import sys
from pathlib import Path

from sku_analyzer.ai_mapping.integration_point import AIMapingIntegration


def setup_logging():
    """Setup production logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


async def main():
    """Main execution function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if len(sys.argv) < 2:
        print("Usage: uv run python run_ai_mapping.py <output_directory> [starting_parent]")
        print("Example: uv run python run_ai_mapping.py production_output/1756744145 4301")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    starting_parent = sys.argv[2] if len(sys.argv) > 2 else "4301"
    
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        sys.exit(1)
    
    print(f"🚀 AI Mapping Pipeline")
    print(f"   📁 Output dir: {output_dir}")
    print(f"   🎯 Starting parent: {starting_parent}")
    
    try:
        # Initialize AI integration
        ai_integration = AIMapingIntegration(enable_ai=True)
        
        # Process AI mapping
        logger.info("Starting AI mapping process")
        result = ai_integration.process_ai_mapping_step(
            output_dir=output_dir,
            starting_parent=starting_parent
        )
        
        # Display results
        if result.get("ai_mapping_completed"):
            print("\n✅ AI Mapping Completed Successfully!")
            
            summary = result["summary"]
            performance = result["performance"]
            
            print(f"📊 SUMMARY:")
            print(f"   Parents processed: {summary['total_parents']}")
            print(f"   Successful: {summary['successful']}")
            print(f"   Failed: {summary['failed']}")
            print(f"   Success rate: {summary['success_rate']:.1%}")
            print(f"   Average confidence: {summary['average_confidence']:.2f}")
            
            print(f"\n⏱️  PERFORMANCE:")
            print(f"   Total time: {performance['total_processing_time_ms']:.1f}ms")
            print(f"   Avg time/parent: {performance['average_time_per_parent']:.1f}ms")
            
            # List generated files
            successful_results = [r for r in result["details"] if r.get("success")]
            print(f"\n📁 GENERATED FILES ({len(successful_results)} total):")
            for res in successful_results:
                parent = res['parent_sku']
                mapped = res['mapped_fields_count']
                confidence = res['confidence']
                file_path = res['output_file']
                print(f"   {parent}: {mapped} mapped (conf: {confidence:.2f}) -> {file_path}")
            
            print(f"\n🎯 AI mapping pipeline completed successfully!")
            print(f"   All step3_ai_mapping.json files created in parent directories")
            
        else:
            print(f"❌ AI mapping failed: {result}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"AI mapping pipeline failed: {e}")
        print(f"❌ Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())