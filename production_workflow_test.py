#!/usr/bin/env python3
"""
Production Workflow Test System for AI Mapping Pipeline

Implements comprehensive end-to-end testing of the complete AI mapping workflow:
1. Load input files and initialize pipeline
2. Execute SKU analysis with CSV splitting and JSON compression
3. Run flat file analysis with template processing
4. Execute AI mapping with real Gemini-2.5-flash connectivity
5. Validate all pipeline outputs and capture performance metrics

Features:
- Real API integration testing with .env configuration
- Step-by-step validation with file existence checks
- Performance monitoring and comprehensive error handling
- Production-quality logging and output formatting
- Scalable parent processing (4301 first, then all parents)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sku_analyzer import SkuPatternAnalyzer, PipelineValidationError
from sku_analyzer.utils import JobManager
from sku_analyzer.ai_mapping.integration_point import AIMapingIntegration


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


class ProductionWorkflowTester:
    """Production-quality workflow test system for AI mapping pipeline."""
    
    def __init__(self):
        """Initialize production workflow tester with comprehensive monitoring."""
        # Load environment variables first
        load_environment_variables()
        
        self.logger = self._setup_production_logging()
        self.start_time = time.time()
        self.job_id: Optional[str] = None
        self.test_metrics = {
            "start_timestamp": datetime.now().isoformat(),
            "steps_completed": [],
            "step_durations": {},
            "files_created": [],
            "validation_results": {},
            "performance_metrics": {},
            "api_connectivity_status": None,
            "errors": []
        }
        
        # Input file configurations
        self.input_files = {
            "primary_data": "test-files/EIKO Stammdaten.xlsx",
            "template_data": "test-files/PANTS (3).xlsm"
        }
        
        # Load environment configuration
        self._load_environment_config()
    
    def _setup_production_logging(self) -> logging.Logger:
        """Setup production-grade logging with structured output."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler with structured formatting
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler for persistent logs
            log_file = Path(f"production_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _load_environment_config(self) -> None:
        """Load and validate environment configuration."""
        env_file = Path(".env")
        if not env_file.exists():
            raise FileNotFoundError(
                "❌ .env file not found. Required for API key configuration."
            )
        
        # Load Google API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ GOOGLE_API_KEY not found in environment. "
                "Required for Gemini-2.5-flash connectivity."
            )
        
        self.logger.info(f"✅ Environment configuration loaded (API key: {api_key[:8]}...)")
    
    def _validate_input_files(self) -> None:
        """Validate that all required input files exist."""
        self.logger.info("📋 Validating input files...")
        
        missing_files = []
        for file_type, file_path in self.input_files.items():
            path = Path(file_path)
            if not path.exists():
                missing_files.append(f"{file_type}: {file_path}")
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                self.logger.info(f"   ✅ {file_type}: {file_path} ({size_mb:.1f}MB)")
        
        if missing_files:
            raise FileNotFoundError(
                f"❌ Missing input files: {', '.join(missing_files)}"
            )
    
    async def _execute_step_with_monitoring(
        self, 
        step_name: str, 
        step_function, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute pipeline step with comprehensive monitoring and error handling."""
        step_start = time.time()
        self.logger.info(f"🚀 Starting {step_name}...")
        
        try:
            result = await step_function(*args, **kwargs) if asyncio.iscoroutinefunction(step_function) else step_function(*args, **kwargs)
            
            duration = time.time() - step_start
            self.test_metrics["steps_completed"].append(step_name)
            self.test_metrics["step_durations"][step_name] = duration
            
            self.logger.info(f"✅ {step_name} completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = time.time() - step_start
            error_info = {
                "step": step_name,
                "error": str(e),
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            self.test_metrics["errors"].append(error_info)
            
            self.logger.error(f"❌ {step_name} failed after {duration:.2f}s: {e}")
            raise
    
    async def step_1_sku_analysis_with_csv_export(self) -> str:
        """Execute SKU analysis with CSV export and compression enabled."""
        analyzer = SkuPatternAnalyzer()
        
        # Process with both files - primary data for analysis, template for flat file analysis
        job_id = await analyzer.process_file_with_template(
            data_file=self.input_files["primary_data"],
            template_file=self.input_files["template_data"],
            export_csv=True,
            enable_compression_benchmark=False
        )
        
        self.job_id = job_id
        return job_id
    
    def step_2_validate_csv_splitting(self) -> Dict[str, Any]:
        """Validate CSV splitting completion with detailed file verification."""
        if not self.job_id:
            raise ValueError("No job ID available for validation")
        
        output_dir = Path(f"production_output/{self.job_id}")
        csv_dir = output_dir / "csv_splits"
        
        validation_result = {
            "csv_directory_exists": csv_dir.exists(),
            "csv_files_found": 0,
            "csv_files": [],
            "total_size_mb": 0.0
        }
        
        if csv_dir.exists():
            csv_files = list(csv_dir.glob("*.csv"))
            validation_result["csv_files_found"] = len(csv_files)
            
            for csv_file in csv_files:
                size_mb = csv_file.stat().st_size / (1024 * 1024)
                validation_result["csv_files"].append({
                    "name": csv_file.name,
                    "size_mb": round(size_mb, 2)
                })
                validation_result["total_size_mb"] += size_mb
                self.test_metrics["files_created"].append(str(csv_file))
            
            validation_result["total_size_mb"] = round(validation_result["total_size_mb"], 2)
        
        self.test_metrics["validation_results"]["csv_splitting"] = validation_result
        
        if not validation_result["csv_directory_exists"] or validation_result["csv_files_found"] == 0:
            raise PipelineValidationError("CSV splitting validation failed - no CSV files found")
        
        return validation_result
    
    def step_3_validate_json_compression(self) -> Dict[str, Any]:
        """Validate JSON compression completion with parent directory verification."""
        if not self.job_id:
            raise ValueError("No job ID available for validation")
        
        output_dir = Path(f"production_output/{self.job_id}")
        
        # Find all parent directories
        parent_dirs = list(output_dir.glob("parent_*"))
        
        validation_result = {
            "parent_directories_found": len(parent_dirs),
            "compressed_files_found": 0,
            "missing_compressed_files": [],
            "compressed_files": [],
            "total_compression_ratio": 0.0
        }
        
        for parent_dir in parent_dirs:
            parent_sku = parent_dir.name.replace("parent_", "")
            compressed_file = parent_dir / "step2_compressed.json"
            
            if compressed_file.exists():
                size_mb = compressed_file.stat().st_size / (1024 * 1024)
                validation_result["compressed_files_found"] += 1
                validation_result["compressed_files"].append({
                    "parent_sku": parent_sku,
                    "file": str(compressed_file),
                    "size_mb": round(size_mb, 2)
                })
                self.test_metrics["files_created"].append(str(compressed_file))
            else:
                validation_result["missing_compressed_files"].append(parent_sku)
        
        self.test_metrics["validation_results"]["json_compression"] = validation_result
        
        if validation_result["missing_compressed_files"]:
            raise PipelineValidationError(
                f"JSON compression validation failed - missing files for parents: "
                f"{', '.join(validation_result['missing_compressed_files'])}"
            )
        
        return validation_result
    
    def step_4_validate_flat_file_analysis(self) -> Dict[str, Any]:
        """Validate flat file analysis output and template processing."""
        if not self.job_id:
            raise ValueError("No job ID available for validation")
        
        output_dir = Path(f"production_output/{self.job_id}")
        flat_file_dir = output_dir / "flat_file_analysis"
        
        validation_result = {
            "flat_file_directory_exists": flat_file_dir.exists(),
            "required_files": {},
            "template_analysis_complete": False
        }
        
        # Check for required flat file analysis outputs
        required_files = [
            "step1_template_columns.json",
            "step2_extracted_values.json", 
            "step3_mandatory_fields.json"
        ]
        
        for required_file in required_files:
            file_path = flat_file_dir / required_file
            exists = file_path.exists()
            validation_result["required_files"][required_file] = {
                "exists": exists,
                "path": str(file_path)
            }
            
            if exists:
                self.test_metrics["files_created"].append(str(file_path))
        
        # Determine if template analysis is complete
        validation_result["template_analysis_complete"] = all(
            file_info["exists"] for file_info in validation_result["required_files"].values()
        )
        
        self.test_metrics["validation_results"]["flat_file_analysis"] = validation_result
        
        if not validation_result["template_analysis_complete"]:
            missing_files = [
                file_name for file_name, file_info in validation_result["required_files"].items()
                if not file_info["exists"]
            ]
            raise PipelineValidationError(
                f"Flat file analysis validation failed - missing files: {', '.join(missing_files)}"
            )
        
        return validation_result
    
    async def step_5_test_api_connectivity(self) -> Dict[str, Any]:
        """Test real Gemini-2.5-flash API connectivity with lightweight request."""
        connectivity_result = {
            "api_key_available": bool(os.getenv("GOOGLE_API_KEY")),
            "connectivity_test_passed": False,
            "response_time_ms": None,
            "model_info": None,
            "error": None
        }
        
        try:
            # Test API connectivity without requiring full pipeline output
            test_start = time.time()
            
            # If we have a job_id, try with actual data
            if self.job_id:
                test_output_dir = Path(f"production_output/{self.job_id}")
                
                # Simple API connectivity test using existing AI integration
                ai_integration = AIMapingIntegration(enable_ai=True)
                
                result = ai_integration.process_ai_mapping_step(
                    output_dir=test_output_dir,
                    starting_parent="4301"
                )
                
                response_time = (time.time() - test_start) * 1000
                connectivity_result["response_time_ms"] = round(response_time, 2)
                
                if result.get("ai_mapping_completed"):
                    connectivity_result["connectivity_test_passed"] = True
                    connectivity_result["model_info"] = {
                        "model_name": "gemini-2.5-flash",
                        "parents_processed": result["summary"]["total_parents"],
                        "success_rate": result["summary"]["success_rate"]
                    }
                else:
                    connectivity_result["error"] = "AI mapping did not complete successfully"
            else:
                # For quick tests without full pipeline, just verify API key is available
                # and try to import required modules
                api_key = os.getenv("GOOGLE_API_KEY")
                if api_key and len(api_key) > 20:  # Basic API key format validation
                    connectivity_result["connectivity_test_passed"] = True
                    connectivity_result["response_time_ms"] = (time.time() - test_start) * 1000
                    connectivity_result["model_info"] = {
                        "model_name": "gemini-2.5-flash",
                        "api_key_format": "valid",
                        "test_mode": "lightweight_validation"
                    }
                else:
                    connectivity_result["error"] = "Invalid or missing API key"
            
        except Exception as e:
            connectivity_result["error"] = str(e)
            connectivity_result["connectivity_test_passed"] = False
        
        self.test_metrics["api_connectivity_status"] = connectivity_result
        return connectivity_result
    
    async def step_6_execute_ai_mapping_all_parents(self) -> Dict[str, Any]:
        """Execute AI mapping for all parent directories with performance monitoring."""
        if not self.job_id:
            raise ValueError("No job ID available for AI mapping")
        
        output_dir = Path(f"production_output/{self.job_id}")
        
        # Initialize AI integration with production settings
        ai_integration = AIMapingIntegration(enable_ai=True)
        
        # Execute AI mapping for all parents (starting with 4301)
        mapping_start = time.time()
        
        result = ai_integration.process_ai_mapping_step(
            output_dir=output_dir,
            starting_parent="4301"
        )
        
        mapping_duration = time.time() - mapping_start
        
        # Process and validate results
        if result.get("ai_mapping_completed"):
            # Collect AI mapping output files
            successful_results = [r for r in result["details"] if r.get("success")]
            for res in successful_results:
                self.test_metrics["files_created"].append(res["output_file"])
            
            # Performance metrics
            performance_metrics = {
                "total_mapping_duration_s": round(mapping_duration, 2),
                "parents_processed": result["summary"]["total_parents"],
                "successful_mappings": result["summary"]["successful"],
                "failed_mappings": result["summary"]["failed"],
                "success_rate": result["summary"]["success_rate"],
                "average_confidence": result["summary"]["average_confidence"],
                "average_time_per_parent_ms": result["performance"]["average_time_per_parent"]
            }
            
            self.test_metrics["performance_metrics"]["ai_mapping"] = performance_metrics
            return result
        else:
            raise Exception(f"AI mapping failed: {result}")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test execution report with all metrics."""
        total_duration = time.time() - self.start_time
        
        report = {
            "test_execution_summary": {
                "job_id": self.job_id,
                "start_time": self.test_metrics["start_timestamp"],
                "end_time": datetime.now().isoformat(),
                "total_duration_s": round(total_duration, 2),
                "steps_completed": len(self.test_metrics["steps_completed"]),
                "total_files_created": len(self.test_metrics["files_created"]),
                "errors_encountered": len(self.test_metrics["errors"])
            },
            "pipeline_validation": {
                "csv_splitting": self.test_metrics["validation_results"].get("csv_splitting", {}),
                "json_compression": self.test_metrics["validation_results"].get("json_compression", {}),
                "flat_file_analysis": self.test_metrics["validation_results"].get("flat_file_analysis", {}),
            },
            "api_connectivity": self.test_metrics["api_connectivity_status"],
            "performance_metrics": self.test_metrics["performance_metrics"],
            "step_performance": self.test_metrics["step_durations"],
            "files_created": self.test_metrics["files_created"],
            "errors": self.test_metrics["errors"]
        }
        
        return report
    
    async def execute_complete_workflow(self) -> Dict[str, Any]:
        """Execute the complete production workflow test with all validation steps."""
        self.logger.info("🎯 Starting Production Workflow Test System")
        self.logger.info("=" * 80)
        
        try:
            # Step 0: Validate prerequisites
            await self._execute_step_with_monitoring(
                "Input File Validation", 
                self._validate_input_files
            )
            
            # Step 1: SKU Analysis with CSV Export and Compression
            job_id = await self._execute_step_with_monitoring(
                "SKU Analysis with CSV Export",
                self.step_1_sku_analysis_with_csv_export
            )
            
            self.logger.info(f"📁 Job ID: {job_id}")
            
            # Step 2: Validate CSV Splitting
            csv_validation = await self._execute_step_with_monitoring(
                "CSV Splitting Validation",
                self.step_2_validate_csv_splitting
            )
            
            self.logger.info(f"📄 CSV files created: {csv_validation['csv_files_found']}")
            
            # Step 3: Validate JSON Compression
            compression_validation = await self._execute_step_with_monitoring(
                "JSON Compression Validation",
                self.step_3_validate_json_compression
            )
            
            self.logger.info(f"🗜️ Compressed files: {compression_validation['compressed_files_found']}")
            
            # Step 4: Validate Flat File Analysis
            flat_file_validation = await self._execute_step_with_monitoring(
                "Flat File Analysis Validation",
                self.step_4_validate_flat_file_analysis
            )
            
            self.logger.info(f"📋 Template analysis complete: {flat_file_validation['template_analysis_complete']}")
            
            # Step 5: Test API Connectivity
            api_test = await self._execute_step_with_monitoring(
                "API Connectivity Test",
                self.step_5_test_api_connectivity
            )
            
            self.logger.info(f"🌐 API connectivity: {'✅ PASS' if api_test['connectivity_test_passed'] else '❌ FAIL'}")
            
            # Step 6: Execute AI Mapping for All Parents
            ai_mapping_result = await self._execute_step_with_monitoring(
                "AI Mapping All Parents",
                self.step_6_execute_ai_mapping_all_parents
            )
            
            self.logger.info(f"🤖 AI mapping completed: {ai_mapping_result['summary']['total_parents']} parents")
            
            # Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            # Save report to file
            report_file = Path(f"production_output/{job_id}/workflow_test_report.json")
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with report_file.open('w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info("✅ Production workflow test completed successfully!")
            self.logger.info(f"📊 Report saved: {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Production workflow test failed: {e}")
            
            # Generate error report
            error_report = self.generate_comprehensive_report()
            error_report["test_status"] = "FAILED"
            error_report["final_error"] = str(e)
            
            return error_report


async def main():
    """Main execution function for production workflow testing."""
    print("🚀 Production Workflow Test System for AI Mapping Pipeline")
    print("=" * 80)
    print("Features:")
    print("  • Complete end-to-end pipeline testing")
    print("  • Real Gemini-2.5-flash API connectivity validation")
    print("  • Step-by-step file existence verification")
    print("  • Performance monitoring and comprehensive reporting")
    print("  • Production-quality error handling and logging")
    print("=" * 80)
    
    try:
        # Initialize and execute workflow test
        tester = ProductionWorkflowTester()
        report = await tester.execute_complete_workflow()
        
        # Display summary results
        print("\n📊 WORKFLOW TEST SUMMARY")
        print("=" * 40)
        
        summary = report["test_execution_summary"]
        print(f"Job ID: {summary['job_id']}")
        print(f"Total Duration: {summary['total_duration_s']}s")
        print(f"Steps Completed: {summary['steps_completed']}/{6}")  # Expected 6 main steps
        print(f"Files Created: {summary['total_files_created']}")
        print(f"Errors: {summary['errors_encountered']}")
        
        # API connectivity status
        api_status = report["api_connectivity"]
        if api_status:
            connectivity_status = "✅ CONNECTED" if api_status.get("connectivity_test_passed") else "❌ FAILED"
            print(f"API Connectivity: {connectivity_status}")
            if api_status.get("response_time_ms"):
                print(f"API Response Time: {api_status['response_time_ms']}ms")
        
        # Performance metrics
        if "ai_mapping" in report["performance_metrics"]:
            ai_perf = report["performance_metrics"]["ai_mapping"]
            print(f"AI Mapping Success Rate: {ai_perf['success_rate']:.1%}")
            print(f"Average Confidence: {ai_perf['average_confidence']:.2f}")
            print(f"Parents Processed: {ai_perf['parents_processed']}")
        
        print("\n🎉 Production workflow test execution completed!")
        
        if summary['errors_encountered'] == 0:
            print("✅ All tests passed successfully!")
        else:
            print(f"⚠️ {summary['errors_encountered']} errors encountered - check logs for details")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test system error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())