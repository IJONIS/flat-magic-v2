#!/usr/bin/env python3
"""
Production Test Execution Wrapper

Provides convenient entry points for running the production workflow test system
with different configurations and validation options.

Usage:
    python run_production_test.py --validate    # System readiness check only
    python run_production_test.py --full        # Complete production workflow test
    python run_production_test.py --quick       # Quick validation with minimal AI testing
"""

import asyncio
import argparse
import sys
from pathlib import Path


async def run_validation_only():
    """Run system readiness validation only."""
    print("🔍 Running system readiness validation...")
    
    from test_production_system import validate_system_readiness
    ready = await validate_system_readiness()
    
    if ready:
        print("\n✅ System validation passed! Ready for production testing.")
        return 0
    else:
        print("\n❌ System validation failed! Please fix issues before testing.")
        return 1


async def run_full_production_test():
    """Run the complete production workflow test."""
    print("🚀 Running complete production workflow test...")
    print("This will execute the full pipeline including real AI API calls.")
    print("Estimated duration: 2-5 minutes depending on data size.\n")
    
    from production_workflow_test import ProductionWorkflowTester
    
    try:
        tester = ProductionWorkflowTester()
        report = await tester.execute_complete_workflow()
        
        # Success/failure based on errors
        errors = report["test_execution_summary"]["errors_encountered"]
        if errors == 0:
            print("\n🎉 Production workflow test PASSED!")
            return 0
        else:
            print(f"\n⚠️ Production workflow test completed with {errors} errors.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Production workflow test FAILED: {e}")
        return 1


async def run_quick_test():
    """Run quick validation with minimal AI testing."""
    print("⚡ Running quick production test (validation + minimal AI)...")
    
    # First validate system
    validation_result = await run_validation_only()
    if validation_result != 0:
        return validation_result
    
    print("\n🧪 Running minimal AI connectivity test...")
    
    # Import here to avoid issues if modules aren't ready
    from production_workflow_test import ProductionWorkflowTester
    
    try:
        tester = ProductionWorkflowTester()
        
        # Just run API connectivity test
        api_test = await tester.step_5_test_api_connectivity()
        
        if api_test["connectivity_test_passed"]:
            print("✅ API connectivity test passed!")
            print(f"   Response time: {api_test['response_time_ms']}ms")
            return 0
        else:
            print("❌ API connectivity test failed!")
            print(f"   Error: {api_test.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return 1


def main():
    """Main execution function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Production Test Execution Wrapper for AI Mapping Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_production_test.py --validate    # Check system readiness
    python run_production_test.py --full        # Complete workflow test
    python run_production_test.py --quick       # Quick validation + API test
    
    Default behavior (no flags): runs validation only
        """
    )
    
    parser.add_argument(
        '--validate', 
        action='store_true',
        help='Run system readiness validation only'
    )
    
    parser.add_argument(
        '--full', 
        action='store_true',
        help='Run complete production workflow test with all steps'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick validation with minimal AI connectivity test'
    )
    
    args = parser.parse_args()
    
    # Determine execution mode
    if args.full:
        result = asyncio.run(run_full_production_test())
    elif args.quick:
        result = asyncio.run(run_quick_test())
    else:
        # Default to validation (args.validate or no flags)
        result = asyncio.run(run_validation_only())
    
    sys.exit(result)


if __name__ == "__main__":
    main()