#!/usr/bin/env python3
"""Main entry point for SKU analyzer with performance optimization."""

import asyncio
import json
import sys

from sku_analyzer import SkuPatternAnalyzer, PipelineValidationError
from sku_analyzer.utils import JobManager


async def main():
    """Main execution function for CLI usage."""
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python main.py <input_xlsx_file>                          # Process file with CSV export and compression")
        print("  python main.py <input_xlsx_file> --template <xlsm_file>   # Process with integrated template analysis")
        print("  python main.py <input_xlsx_file> --no-csv                 # Process file without CSV export")
        print("  python main.py <input_xlsx_file> --benchmark              # Run performance benchmark")
        print("  python main.py <input_xlsx_file> --full-bench             # Run comprehensive compression benchmark")
        print("  python main.py --latest                                   # Show latest job")
        print("  python main.py --add-template <job_id> <xlsm_file>        # Add template analysis to existing job")
        sys.exit(1)
    
    if sys.argv[1] == "--latest":
        JobManager.show_latest_job()
        return
    
    if sys.argv[1] == "--add-template":
        if len(sys.argv) < 4:
            print("❌ Usage: python main.py --add-template <job_id> <xlsm_file>")
            sys.exit(1)
        
        job_id = sys.argv[2]
        template_file = sys.argv[3]
        
        try:
            analyzer = SkuPatternAnalyzer()
            print(f"🔗 Adding template analysis to job {job_id}...")
            
            template_result = await analyzer.add_template_analysis_to_job(template_file, job_id)
            
            print(f"✅ Template analysis added successfully!")
            print(f"📁 Job ID: {job_id}")
            print(f"📄 Template results: production_output/{job_id}/flat_file_analysis/")
            print(f"🔍 Column mappings: {template_result.get('total_mappings', 0)}")
            
            # Show requirement statistics if available
            req_stats = template_result.get('analysis_metadata', {}).get('requirement_statistics', {})
            if req_stats:
                print(f"📊 Requirements: {req_stats.get('mandatory', 0)} mandatory, {req_stats.get('optional', 0)} optional, {req_stats.get('recommended', 0)} recommended")
            
        except Exception as e:
            print(f"❌ Failed to add template analysis: {e}")
            sys.exit(1)
        
        return
    
    input_file = sys.argv[1]
    analyzer = SkuPatternAnalyzer()
    
    # Parse command line options
    export_csv = "--no-csv" not in sys.argv
    run_benchmark = "--benchmark" in sys.argv
    run_full_benchmark = "--full-bench" in sys.argv
    
    # Check for template file parameter
    template_file = None
    if "--template" in sys.argv:
        template_idx = sys.argv.index("--template")
        if template_idx + 1 < len(sys.argv):
            template_file = sys.argv[template_idx + 1]
        else:
            print("❌ Error: --template flag requires a template file path")
            sys.exit(1)
    
    try:
        if run_benchmark:
            print(f"🚀 Running performance benchmark on {input_file}...")
            benchmark_results = await analyzer.benchmark_performance(input_file)
            
            print(f"📊 Benchmark Results:")
            print(f"   Duration: {benchmark_results['metrics']['duration_seconds']:.2f}s")
            print(f"   Peak Memory: {benchmark_results['metrics']['peak_memory_mb']:.1f}MB")
            print(f"   Throughput: {benchmark_results['metrics']['throughput_rows_per_second']:.0f} rows/s")
            print(f"   Files Created: {benchmark_results['metrics']['files_created']}")
            
            # Performance validation
            validation = benchmark_results['validation']
            if validation['overall_pass']:
                print(f"✅ Performance targets met!")
            else:
                print(f"⚠️ Performance targets not met:")
                if not validation['duration_target_met']:
                    print(f"   Duration exceeded 5.0s target")
                if not validation['memory_target_met']:
                    print(f"   Memory exceeded 100MB target")
            
            # Save benchmark report
            with open(f"benchmark_report_{input_file.replace('/', '_').replace('.xlsx', '')}.json", 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            print(f"📄 Benchmark report saved")
            
        elif run_full_benchmark:
            print(f"🎯 Running comprehensive compression benchmark on {input_file}...")
            
            # Load data for comprehensive benchmarking
            df = await analyzer.load_xlsx_data(input_file)
            hierarchy = analyzer.hierarchy_extractor.extract_hierarchical_patterns(df)
            relationships = analyzer.build_hierarchical_relationships(hierarchy)
            
            # Run comprehensive benchmark
            from sku_analyzer.optimization import CompressionPerformanceBenchmark
            from pathlib import Path
            
            benchmark_tool = CompressionPerformanceBenchmark(Path("compression_benchmarks"))
            benchmark_results = await benchmark_tool.run_full_benchmark_suite(
                df, relationships, f"cmdline_{input_file.replace('/', '_').replace('.xlsx', '')}"
            )
            
            # Display results summary
            print(f"\n📊 Comprehensive Benchmark Results:")
            print(f"   Dataset: {benchmark_results.dataset_info['total_rows']} rows, {benchmark_results.dataset_info['parent_groups']} groups")
            
            # Performance summary
            compression_perf = benchmark_results.compression_performance.get('pipeline_metrics', {})
            print(f"   Compression Ratio: {compression_perf.get('overall_compression_ratio', 0):.1%}")
            print(f"   Processing Speed: {compression_perf.get('total_processing_time_seconds', 0):.2f}s total")
            print(f"   Memory Usage: {compression_perf.get('memory_efficiency_mb_per_group', 0):.1f}MB per group")
            print(f"   Throughput: {compression_perf.get('throughput_groups_per_second', 0):.1f} groups/s")
            
            # Library recommendation
            library_rec = benchmark_results.library_comparison.get('recommendation', {})
            optimal_lib = library_rec.get('optimal_library', 'json')
            speedup = benchmark_results.library_comparison.get('performance_comparison', {}).get('speed_improvement_factor', 1)
            print(f"   Optimal JSON Library: {optimal_lib} ({speedup:.1f}x speedup)")
            
            # Performance validation
            targets = benchmark_results.performance_targets_met
            passed = sum(1 for v in targets.values() if v)
            total = len(targets)
            print(f"   Performance Targets: {passed}/{total} met")
            
            if targets.get('overall_performance_pass', False):
                print(f"✅ All performance targets achieved!")
            else:
                print(f"⚠️ Optimization opportunities identified")
            
            # Top recommendations
            if benchmark_results.recommendations:
                print(f"\n💡 Top Optimization Recommendations:")
                for i, rec in enumerate(benchmark_results.recommendations[:3], 1):
                    print(f"   {i}. {rec}")
            
            print(f"\n📄 Full benchmark report: compression_benchmarks/benchmark_report_cmdline_{input_file.replace('/', '_').replace('.xlsx', '')}.json")
            
        else:
            # Determine processing mode
            if template_file:
                print(f"🏗️ Running integrated analysis with template...")
                print(f"📊 Data file: {input_file}")
                print(f"📋 Template file: {template_file}")
                
                job_id = await analyzer.process_file_with_template(
                    input_file, 
                    template_file, 
                    export_csv=export_csv,
                    enable_compression_benchmark=False
                )
                
                print(f"✅ Integrated analysis completed successfully!")
                print(f"📁 Job ID: {job_id}")
                print(f"📄 Results saved to: production_output/{job_id}/")
                print(f"📊 SKU analysis: production_output/{job_id}/analysis_{job_id}.json")
                print(f"📋 Template analysis: production_output/{job_id}/flat_file_analysis/step1_template_columns.json")
                
                if export_csv:
                    print(f"📈 CSV files: production_output/{job_id}/parent_*/data.csv")
                
                compression_summary = f"production_output/{job_id}/compression_summary.json"
                print(f"🗜️ Compression analysis: {compression_summary}")
                
            else:
                print(f"📊 Using hierarchical delimiter analysis...")
                if export_csv:
                    print(f"⚡ CSV export enabled - optimized parallel processing")
                print(f"🗜️ Compression optimization enabled by default")
                
                job_id = await analyzer.process_file(input_file, export_csv=export_csv)
                print(f"✅ Analysis completed successfully!")
                print(f"📁 Job ID: {job_id}")
                print(f"📄 Results saved to: production_output/{job_id}/")
                
                if export_csv:
                    print(f"📊 CSV files available in: production_output/{job_id}/csv_splits/")
                
                compression_summary = f"production_output/{job_id}/compression_summary.json"
                print(f"🗜️ Compression analysis: {compression_summary}")
        
    except PipelineValidationError as e:
        print(f"❌ Pipeline validation failed: {e}")
        print("📋 Check that all processing steps completed successfully.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())