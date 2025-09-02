#!/usr/bin/env python3
"""
Quick validation test for the production workflow system.

This script validates that all components are properly configured
and the system is ready for production workflow testing.
"""

import asyncio
import os
import sys
from pathlib import Path


def load_environment_variables():
    """Load environment variables from .env file if it exists."""
    env_file = Path('.env')
    if env_file.exists():
        with env_file.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value


async def validate_system_readiness():
    """Validate that the system is ready for production workflow testing."""
    
    # Load environment variables first
    load_environment_variables()
    
    print("🔍 Production System Readiness Check")
    print("=" * 50)
    
    validation_results = []
    
    # 1. Check environment configuration
    print("1. Environment Configuration...")
    env_file = Path(".env")
    if env_file.exists():
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            print(f"   ✅ API key configured: {api_key[:8]}...")
            validation_results.append(True)
        else:
            print("   ❌ GOOGLE_API_KEY not found in environment")
            validation_results.append(False)
    else:
        print("   ❌ .env file not found")
        validation_results.append(False)
    
    # 2. Check input files
    print("2. Input Files...")
    input_files = {
        "EIKO Stammdaten.xlsx": "test-files/EIKO Stammdaten.xlsx",
        "PANTS template": "test-files/PANTS (3).xlsm"
    }
    
    for file_name, file_path in input_files.items():
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {file_name}: {file_path} ({size_mb:.1f}MB)")
            validation_results.append(True)
        else:
            print(f"   ❌ Missing: {file_path}")
            validation_results.append(False)
    
    # 3. Check core modules can be imported
    print("3. Core Module Imports...")
    try:
        from sku_analyzer import SkuPatternAnalyzer, PipelineValidationError
        from sku_analyzer.utils import JobManager
        from sku_analyzer.ai_mapping.integration_point import AIMapingIntegration
        from production_workflow_test import ProductionWorkflowTester
        
        print("   ✅ All core modules imported successfully")
        validation_results.append(True)
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        validation_results.append(False)
    
    # 4. Check output directory structure
    print("4. Output Directory Structure...")
    production_output = Path("production_output")
    if production_output.exists():
        existing_jobs = list(production_output.glob("*"))
        print(f"   ✅ Production output directory exists ({len(existing_jobs)} existing jobs)")
        validation_results.append(True)
    else:
        print("   ℹ️ Production output directory will be created")
        validation_results.append(True)  # This is OK, it will be created
    
    # 5. Test basic analyzer initialization
    print("5. Basic Analyzer Initialization...")
    try:
        analyzer = SkuPatternAnalyzer()
        print("   ✅ SkuPatternAnalyzer initialized successfully")
        validation_results.append(True)
    except Exception as e:
        print(f"   ❌ Analyzer initialization failed: {e}")
        validation_results.append(False)
    
    # 6. Test AI integration initialization
    print("6. AI Integration Initialization...")
    try:
        ai_integration = AIMapingIntegration(enable_ai=True)
        print("   ✅ AI integration initialized successfully")
        validation_results.append(True)
    except Exception as e:
        print(f"   ❌ AI integration initialization failed: {e}")
        validation_results.append(False)
    
    # Summary
    print("\n📊 Validation Summary:")
    passed = sum(validation_results)
    total = len(validation_results)
    
    print(f"   Tests passed: {passed}/{total}")
    
    if passed == total:
        print("   ✅ System is ready for production workflow testing!")
        print("\n🚀 To run the complete production test:")
        print("   python production_workflow_test.py")
        return True
    else:
        print("   ❌ System is not ready - please fix the issues above")
        return False


async def main():
    """Main validation function."""
    try:
        ready = await validate_system_readiness()
        sys.exit(0 if ready else 1)
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())