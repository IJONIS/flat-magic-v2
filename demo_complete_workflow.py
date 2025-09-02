#!/usr/bin/env python3
"""
Demo: Complete End-to-End Flat Magic Workflow

Demonstrates the full pipeline:
1. SKU Analysis: Extract parent-child relationships from data
2. Template Analysis (Step 1): Analyze template structure  
3. Value Extraction (Step 2): Extract valid values from template
4. Compression: Compress results for storage efficiency
"""

import asyncio
import json
from pathlib import Path

from sku_analyzer import SkuPatternAnalyzer


async def demo_complete_workflow():
    """Demonstrate the complete end-to-end workflow."""
    print("🚀 Flat Magic Complete Workflow Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SkuPatternAnalyzer()
    
    # Data file for SKU analysis
    data_file = "test-files/EIKO Stammdaten.xlsx"
    
    print(f"📊 Processing data file: {data_file}")
    print("📋 Features: SKU analysis + CSV export + compression")
    
    try:
        # Run complete SKU analysis workflow
        job_id = await analyzer.process_file(
            input_path=data_file,
            export_csv=True,
            enable_compression_benchmark=False
        )
        
        print(f"\n✅ Job {job_id} completed successfully!")
        
        # Show results
        output_dir = Path(f"production_output/{job_id}")
        
        print(f"\n📁 Results Structure:")
        print(f"   📊 Analysis: {output_dir}/analysis_results.json")
        print(f"   📄 CSV Files: {output_dir}/csv_splits/")
        print(f"   🗜️ Compressed: {output_dir}/parent_*/step2_compressed.json") 
        
        # Load and display summary
        analysis_file = output_dir / "analysis_results.json"
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                data = json.load(f)
                
            print(f"\n📈 Analysis Summary:")
            print(f"   Parents found: {len(data.get('relationships', {}))}")
            print(f"   Total SKUs: {data.get('total_skus', 0)}")
            
            # Show parent examples
            relationships = data.get('relationships', {})
            if relationships:
                print(f"\n👥 Parent-Child Examples:")
                for parent, rel in list(relationships.items())[:3]:
                    children_count = len(rel.get('children_skus', []))
                    print(f"   {parent}: {children_count} children")
        
        # Check for CSV outputs
        csv_dir = output_dir / "csv_splits"
        if csv_dir.exists():
            csv_files = list(csv_dir.glob("*.csv"))
            print(f"\n📄 CSV Exports: {len(csv_files)} files")
            
        # Check for compressed outputs
        compressed_files = list(output_dir.glob("parent_*/step2_compressed.json"))
        if compressed_files:
            print(f"🗜️ Compressed Results: {len(compressed_files)} files")
            
        return job_id
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        raise


async def demo_template_workflow():
    """Demonstrate template analysis workflow when template definitions are available."""
    print("\n" + "=" * 50)
    print("📋 Template Analysis Workflow (requires proper template file)")
    print("=" * 50)
    
    # This would work with a proper template definition file
    print("⚠️  Template analysis requires Excel files with:")
    print("   - 'Feldname' column (technical names)")
    print("   - 'Lokale Bezeichnung' column (display names)")
    print("   - Template structure (not product data)")
    
    print("\n💡 Current demo file is product data, not template definitions")
    print("   For template analysis, provide .xlsm files with field definitions")


async def main():
    """Main demo function."""
    try:
        # Run complete SKU analysis workflow  
        job_id = await demo_complete_workflow()
        
        # Show template analysis requirements
        await demo_template_workflow()
        
        print(f"\n🎉 Complete workflow demo finished!")
        print(f"📁 Check results in: production_output/{job_id}/")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())