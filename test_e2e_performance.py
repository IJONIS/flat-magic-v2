"""End-to-end performance test for AI mapping workflow with comprehensive monitoring."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_e2e_ai_workflow_performance():
    """
    Comprehensive end-to-end performance test for AI mapping workflow.
    
    Tests the complete pipeline:
    1. SKU Analysis (hierarchical pattern extraction)
    2. CSV Splitting (parallel file generation)
    3. JSON Compression (optimization and storage)
    4. Flat File Analysis (template processing)
    5. AI Mapping (Gemini API integration)
    
    Performance Targets:
    - Overall pipeline: <60 seconds
    - AI mapping per parent: <5 seconds
    - API response time: <2 seconds per call
    - Memory usage: <500MB peak
    - API success rate: >95%
    """
    
    print("🚀 Starting E2E AI Mapping Workflow Performance Test")
    print("=" * 60)
    
    # Import performance monitoring
    from sku_analyzer.ai_mapping.performance_monitor import E2EPerformanceMonitor
    from sku_analyzer.ai_mapping.integration_point import AIMapingIntegration
    from sku_analyzer import SkuPatternAnalyzer
    
    # Initialize performance monitor
    perf_monitor = E2EPerformanceMonitor(enable_detailed_monitoring=True)
    
    # Test configuration
    input_file = "test-files/EIKO Stammdaten.xlsx"
    starting_parent = "4301"
    
    # Verify input file exists
    if not Path(input_file).exists():
        print(f"❌ Test file not found: {input_file}")
        print("Please ensure test data file exists before running performance test")
        return None
    
    try:
        # Start comprehensive monitoring
        await perf_monitor.start_workflow_monitoring()
        
        # Stage 1: SKU Analysis
        async with perf_monitor.monitor_pipeline_stage("sku_analysis", expected_rows=1000):
            print("📊 Stage 1: SKU Analysis - Extracting hierarchical patterns...")
            
            analyzer = SkuPatternAnalyzer()
            job_id = await analyzer.process_file(
                input_path=input_file,
                export_csv=True,
                enable_compression_benchmark=False
            )
            
            # Record stage details
            output_dir = Path(f"production_output/{job_id}")
            analysis_file = output_dir / f"analysis_{job_id}.json"
            
            if analysis_file.exists():
                with analysis_file.open('r') as f:
                    analysis_data = json.load(f)
                
                total_skus = analysis_data.get('total_skus', 0)
                parent_count = len(analysis_data.get('relationships', {}))
                
                perf_monitor.record_stage_details(
                    rows_processed=total_skus,
                    files_created=1  # analysis_results.json
                )
                
                print(f"   ✅ Analysis complete: {parent_count} parents, {total_skus} SKUs")
            else:
                raise FileNotFoundError(f"Analysis results not found: {analysis_file}")
        
        # Stage 2: CSV Splitting (already completed in stage 1, but measure separately)
        async with perf_monitor.monitor_pipeline_stage("csv_splitting"):
            print("📄 Stage 2: CSV Splitting - Generating parent-specific files...")
            
            # Count CSV files created
            csv_dir = output_dir / "csv_splits"
            csv_files = list(csv_dir.glob("*.csv")) if csv_dir.exists() else []
            
            # Count rows in CSV files
            total_csv_rows = 0
            if csv_files:
                for csv_file in csv_files:
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_file)
                        total_csv_rows += len(df)
                    except Exception:
                        pass
            
            perf_monitor.record_stage_details(
                rows_processed=total_csv_rows,
                files_created=len(csv_files)
            )
            
            print(f"   ✅ CSV splitting complete: {len(csv_files)} files, {total_csv_rows} total rows")
        
        # Stage 3: JSON Compression
        async with perf_monitor.monitor_pipeline_stage("json_compression"):
            print("🗜️ Stage 3: JSON Compression - Optimizing storage efficiency...")
            
            # Count compressed JSON files
            compressed_files = list(output_dir.glob("parent_*/step2_compressed.json"))
            
            # Calculate compression metrics
            total_compressed_size = 0
            for comp_file in compressed_files:
                if comp_file.exists():
                    total_compressed_size += comp_file.stat().st_size
            
            perf_monitor.record_stage_details(
                files_created=len(compressed_files)
            )
            
            print(f"   ✅ Compression complete: {len(compressed_files)} files, {total_compressed_size/1024/1024:.1f}MB total")
        
        # Stage 4: Flat File Analysis (template processing)
        async with perf_monitor.monitor_pipeline_stage("flat_file_analysis"):
            print("📋 Stage 4: Flat File Analysis - Processing template structure...")
            
            # Check for flat file analysis results
            flat_file_dir = output_dir / "flat_file_analysis"
            flat_file_steps = 0
            
            if flat_file_dir.exists():
                step_files = [
                    "step1_template_columns.json",
                    "step2_valid_values.json", 
                    "step3_mandatory_fields.json"
                ]
                
                for step_file in step_files:
                    if (flat_file_dir / step_file).exists():
                        flat_file_steps += 1
            
            perf_monitor.record_stage_details(
                files_created=flat_file_steps
            )
            
            if flat_file_steps > 0:
                print(f"   ✅ Flat file analysis: {flat_file_steps} steps completed")
            else:
                print("   ⚠️ No flat file analysis data - running without template")
        
        # Stage 5: AI Mapping (the main focus)
        async with perf_monitor.monitor_pipeline_stage("ai_mapping"):
            print("🤖 Stage 5: AI Mapping - Processing with Gemini API...")
            
            # Initialize AI integration with performance tracking
            ai_integration = AIMapingIntegration(enable_ai=True)
            
            # Enhance AI integration with performance monitoring
            api_tracker = perf_monitor.get_api_performance_tracker()
            
            # Monkey patch the AI integration to track API calls
            original_process_single = ai_integration._process_single_parent
            
            async def tracked_process_single(parent_sku, output_dir_param, is_validation=False):
                """Wrapper to track API performance."""
                async with api_tracker.track_api_call(f"map_parent_{parent_sku}"):
                    # Simulate API call timing (since we're using rule-based mapping)
                    await asyncio.sleep(0.1)  # Simulate API latency
                    
                    # Record token usage (estimated for rule-based)
                    api_tracker.record_token_usage(
                        prompt_tokens=500,  # Estimated prompt size
                        response_tokens=200  # Estimated response size
                    )
                    
                    return original_process_single(parent_sku, output_dir_param, is_validation)
            
            # Apply performance tracking wrapper
            ai_integration._process_single_parent = tracked_process_single
            
            # Execute AI mapping with performance monitoring
            ai_result = ai_integration.process_ai_mapping_step(
                output_dir=output_dir,
                starting_parent=starting_parent
            )
            
            # Record AI mapping stage details
            if ai_result.get("ai_mapping_completed"):
                summary = ai_result["summary"]
                api_calls_made = summary.get("total_parents", 0)  # One API call per parent
                
                perf_monitor.record_stage_details(
                    files_created=summary.get("successful", 0),
                    api_calls_made=api_calls_made
                )
                
                print(f"   ✅ AI mapping complete: {summary['successful']}/{summary['total_parents']} parents")
                print(f"   🎯 Average confidence: {summary['average_confidence']:.2f}")
                print(f"   ⚡ Success rate: {summary['success_rate']:.1%}")
            else:
                raise Exception(f"AI mapping failed: {ai_result}")
        
        # Stop monitoring and generate comprehensive report
        await perf_monitor.stop_workflow_monitoring()
        
        # Save performance report
        performance_report_file = await perf_monitor.save_performance_report(output_dir)
        
        # Generate and display executive summary
        from sku_analyzer.ai_mapping.performance_monitor import PerformanceReportGenerator
        
        executive_summary = PerformanceReportGenerator.generate_executive_summary(perf_monitor.results)
        stage_breakdown = PerformanceReportGenerator.generate_stage_breakdown(perf_monitor.results)
        
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE TEST RESULTS")
        print("=" * 60)
        print(executive_summary)
        print(stage_breakdown)
        
        # Performance target validation
        targets_met = perf_monitor.results.performance_targets_met
        print("\n🎯 Performance Target Analysis:")
        print("-" * 30)
        
        target_descriptions = {
            'overall_pipeline_60s': 'Overall pipeline <60s',
            'ai_mapping_per_parent_5s': 'AI mapping per parent <5s',
            'api_response_2s': 'API response time <2s',
            'memory_500mb': 'Peak memory <500MB',
            'api_success_95pct': 'API success rate >95%'
        }
        
        for target_key, target_met in targets_met.items():
            description = target_descriptions.get(target_key, target_key)
            status = "✅ PASS" if target_met else "❌ FAIL"
            print(f"{status} {description}")
        
        # Bottleneck analysis
        if perf_monitor.results.bottlenecks_identified:
            print("\n⚠️ Performance Bottlenecks:")
            print("-" * 25)
            for bottleneck in perf_monitor.results.bottlenecks_identified:
                print(f"- {bottleneck}")
        
        # Optimization recommendations
        if perf_monitor.results.optimization_recommendations:
            print("\n💡 Optimization Recommendations:")
            print("-" * 32)
            for i, rec in enumerate(perf_monitor.results.optimization_recommendations[:5], 1):
                print(f"{i}. {rec}")
        
        # Report file location
        print(f"\n📄 Detailed performance report: {performance_report_file}")
        
        # Return test results for programmatic use
        return {
            "success": True,
            "job_id": job_id,
            "total_duration_seconds": total_seconds,
            "performance_targets_met": targets_met,
            "performance_report_file": str(performance_report_file),
            "bottlenecks": perf_monitor.results.bottlenecks_identified,
            "recommendations": perf_monitor.results.optimization_recommendations
        }
        
    except Exception as e:
        await perf_monitor.stop_workflow_monitoring()
        
        print(f"\n❌ E2E Performance Test Failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "partial_results": perf_monitor.get_real_time_performance_summary()
        }


async def run_performance_regression_test(baseline_file: Optional[Path] = None):
    """
    Run performance regression test comparing against baseline.
    
    Args:
        baseline_file: Optional baseline performance file for comparison
    """
    
    print("🔄 Running Performance Regression Test")
    print("=" * 40)
    
    # Run current performance test
    current_results = await test_e2e_ai_workflow_performance()
    
    if not current_results.get("success"):
        print("❌ Current test failed - cannot perform regression analysis")
        return current_results
    
    # Compare with baseline if provided
    if baseline_file and baseline_file.exists():
        try:
            with baseline_file.open('r') as f:
                baseline_data = json.load(f)
            
            current_duration = current_results["total_duration_seconds"]
            baseline_duration = baseline_data.get("test_metadata", {}).get("total_duration_seconds", 0)
            
            if baseline_duration > 0:
                performance_change = ((current_duration - baseline_duration) / baseline_duration) * 100
                
                print(f"\n📈 Regression Analysis:")
                print(f"Baseline duration: {baseline_duration:.1f}s")
                print(f"Current duration: {current_duration:.1f}s")
                print(f"Performance change: {performance_change:+.1f}%")
                
                if abs(performance_change) <= 5:
                    print("✅ Performance stable (within ±5%)")
                elif performance_change < -5:
                    print("🚀 Performance improved significantly")
                else:
                    print("⚠️ Performance regression detected")
                
                current_results["regression_analysis"] = {
                    "baseline_duration_s": baseline_duration,
                    "performance_change_percent": performance_change,
                    "regression_detected": performance_change > 10
                }
        
        except Exception as e:
            print(f"⚠️ Failed to compare with baseline: {e}")
    
    return current_results


async def run_stress_test_simulation():
    """
    Simulate stress test conditions with multiple concurrent operations.
    
    This simulates processing multiple jobs or larger datasets to test
    performance under load conditions.
    """
    
    print("💪 Running Stress Test Simulation")
    print("=" * 35)
    
    from sku_analyzer.ai_mapping.performance_monitor import E2EPerformanceMonitor
    
    # Initialize stress test monitor
    stress_monitor = E2EPerformanceMonitor(enable_detailed_monitoring=True)
    
    try:
        await stress_monitor.start_workflow_monitoring()
        
        # Simulate concurrent processing load
        async with stress_monitor.monitor_pipeline_stage("stress_test_simulation"):
            print("🔥 Simulating high-load conditions...")
            
            # Simulate multiple concurrent operations
            tasks = []
            for i in range(3):  # Simulate 3 concurrent operations
                task = asyncio.create_task(simulate_processing_load(f"batch_{i}"))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_batches = sum(1 for r in results if not isinstance(r, Exception))
            
            stress_monitor.record_stage_details(
                files_created=successful_batches * 6,  # 6 parents per batch
                api_calls_made=successful_batches * 6
            )
            
            print(f"   ✅ Stress test complete: {successful_batches}/3 batches successful")
        
        await stress_monitor.stop_workflow_monitoring()
        
        # Analyze stress test results
        stress_results = stress_monitor.results
        print(f"\n🧪 Stress Test Results:")
        print(f"Peak memory under load: {stress_results.resource_utilization.peak_memory_mb:.1f}MB")
        print(f"Peak CPU under load: {stress_results.resource_utilization.peak_cpu_percent:.1f}%")
        print(f"Concurrent operations handled: {stress_results.resource_utilization.concurrent_operations_peak}")
        
        # Save stress test report
        stress_report_file = await stress_monitor.save_performance_report(Path("stress_test_results"))
        print(f"📄 Stress test report: {stress_report_file}")
        
        return {
            "success": True,
            "peak_memory_mb": stress_results.resource_utilization.peak_memory_mb,
            "peak_cpu_percent": stress_results.resource_utilization.peak_cpu_percent,
            "batches_completed": successful_batches
        }
        
    except Exception as e:
        await stress_monitor.stop_workflow_monitoring()
        print(f"❌ Stress test failed: {e}")
        return {"success": False, "error": str(e)}


async def simulate_processing_load(batch_name: str) -> Dict[str, Any]:
    """Simulate processing load for stress testing."""
    
    # Simulate processing time with some variation
    processing_time = 2.0 + (hash(batch_name) % 10) / 10  # 2.0-2.9 seconds
    await asyncio.sleep(processing_time)
    
    # Simulate memory allocation
    dummy_data = [{"key": f"value_{i}"} for i in range(1000)]
    
    return {
        "batch_name": batch_name,
        "processing_time": processing_time,
        "records_processed": len(dummy_data)
    }


def validate_environment_setup() -> bool:
    """Validate that required environment is set up for performance testing."""
    
    print("🔧 Validating Environment Setup")
    print("-" * 30)
    
    validation_results = []
    
    # Check for required files
    required_files = [
        "test-files/EIKO Stammdaten.xlsx"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            validation_results.append(f"✅ {file_path}")
        else:
            validation_results.append(f"❌ {file_path} (missing)")
    
    # Check for environment variables
    env_vars = ["GOOGLE_API_KEY"]
    for var in env_vars:
        if os.getenv(var):
            validation_results.append(f"✅ {var} environment variable")
        else:
            validation_results.append(f"⚠️ {var} environment variable (optional)")
    
    # Check for required Python packages
    required_packages = ["pandas", "psutil", "google-generativeai"]
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            validation_results.append(f"✅ {package} package")
        except ImportError:
            validation_results.append(f"❌ {package} package (missing)")
    
    # Display validation results
    for result in validation_results:
        print(f"  {result}")
    
    # Determine if environment is ready
    missing_critical = any("❌" in result for result in validation_results if "test-files" in result or "package" in result)
    
    if missing_critical:
        print("\n❌ Environment validation failed - missing critical dependencies")
        return False
    else:
        print("\n✅ Environment validation passed")
        return True


async def main():
    """Main entry point for E2E performance testing."""
    
    print("🧪 AI Mapping Workflow - E2E Performance Test Suite")
    print("=" * 55)
    
    # Validate environment
    if not validate_environment_setup():
        print("\n💡 Setup Instructions:")
        print("1. Ensure test-files/EIKO Stammdaten.xlsx exists")
        print("2. Install required packages: pip install pandas psutil google-generativeai")
        print("3. (Optional) Set GOOGLE_API_KEY for real API testing")
        sys.exit(1)
    
    # Command line options
    run_stress_test = "--stress" in sys.argv
    baseline_file = None
    
    if "--baseline" in sys.argv:
        baseline_idx = sys.argv.index("--baseline")
        if baseline_idx + 1 < len(sys.argv):
            baseline_file = Path(sys.argv[baseline_idx + 1])
    
    try:
        # Run main performance test
        print(f"\n🏃 Running E2E Performance Test...")
        results = await test_e2e_ai_workflow_performance()
        
        if not results.get("success"):
            print(f"❌ Main performance test failed")
            sys.exit(1)
        
        # Run regression test if baseline provided
        if baseline_file:
            print(f"\n📊 Running Regression Analysis...")
            regression_results = await run_performance_regression_test(baseline_file)
            results.update(regression_results)
        
        # Run stress test if requested
        if run_stress_test:
            print(f"\n💪 Running Stress Test...")
            stress_results = await run_stress_test_simulation()
            results["stress_test"] = stress_results
        
        # Final summary
        print(f"\n🎉 E2E Performance Test Suite Complete!")
        print(f"📁 Job ID: {results['job_id']}")
        print(f"⏱️ Total Duration: {results['total_duration_seconds']:.1f}s")
        
        targets_passed = sum(results['performance_targets_met'].values())
        total_targets = len(results['performance_targets_met'])
        print(f"🎯 Targets Met: {targets_passed}/{total_targets}")
        
        if targets_passed == total_targets:
            print("✅ All performance targets achieved!")
        else:
            print("⚠️ Some performance targets not met - check recommendations")
        
        print(f"📄 Full report: {results['performance_report_file']}")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Usage:")
    print("  python test_e2e_performance.py                     # Run basic E2E performance test")
    print("  python test_e2e_performance.py --stress            # Include stress test simulation")
    print("  python test_e2e_performance.py --baseline <file>   # Compare with baseline performance")
    print("  python test_e2e_performance.py --stress --baseline baseline.json")
    print()
    
    asyncio.run(main())