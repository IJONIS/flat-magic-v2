#!/usr/bin/env python3
"""
Comprehensive Test Execution Runner

Combines the E2E test execution framework with the validation framework
to provide complete workflow testing with detailed validation and reporting.

Usage:
    python run_comprehensive_test.py
    python run_comprehensive_test.py --detailed-logs
    python run_comprehensive_test.py --skip-ai-mapping
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Import our test frameworks
from test_e2e_ai_workflow import E2ETestExecutor
from test_validation_framework import PipelineValidationOrchestrator, PERFORMANCE_BASELINES, QUALITY_CRITERIA


class ComprehensiveTestRunner:
    """Orchestrates complete test execution with integrated validation."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize test runner with configuration."""
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # Initialize frameworks
        self.test_executor = E2ETestExecutor(enable_detailed_logging=args.detailed_logs)
        self.validation_orchestrator = PipelineValidationOrchestrator()
        
        # Test session metadata
        self.session_id = f"test_session_{int(datetime.now().timestamp())}"
        self.test_start_time = datetime.now()
        
        # Results aggregation
        self.comprehensive_results = {
            "session_metadata": {
                "session_id": self.session_id,
                "start_time": self.test_start_time.isoformat(),
                "configuration": vars(args)
            },
            "execution_results": {},
            "validation_results": {},
            "performance_analysis": {},
            "quality_assessment": {},
            "recommendations": []
        }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Execute comprehensive test suite with integrated validation."""
        
        self.logger.info("🚀 COMPREHENSIVE AI MAPPING WORKFLOW TEST")
        self.logger.info("=" * 80)
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(f"Configuration: {vars(self.args)}")
        self.logger.info("=" * 80)
        
        try:
            # Phase 1: Execute E2E Test Suite
            execution_results = await self._execute_e2e_tests()
            self.comprehensive_results["execution_results"] = execution_results
            
            # Phase 2: Pipeline Validation
            validation_results = await self._execute_pipeline_validation()
            self.comprehensive_results["validation_results"] = validation_results
            
            # Phase 3: Performance Analysis
            performance_analysis = await self._analyze_performance()
            self.comprehensive_results["performance_analysis"] = performance_analysis
            
            # Phase 4: Quality Assessment
            quality_assessment = await self._assess_quality()
            self.comprehensive_results["quality_assessment"] = quality_assessment
            
            # Phase 5: Generate Recommendations
            recommendations = await self._generate_recommendations()
            self.comprehensive_results["recommendations"] = recommendations
            
            # Phase 6: Final Report Generation
            await self._generate_final_report()
            
            return self.comprehensive_results
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive test failed: {e}")
            self.comprehensive_results["session_metadata"]["error"] = str(e)
            raise
    
    async def _execute_e2e_tests(self) -> Dict[str, Any]:
        """Execute end-to-end test suite."""
        self.logger.info("📊 Executing E2E Test Suite...")
        
        try:
            execution_results = await self.test_executor.execute_complete_test_suite()
            
            self.logger.info("✅ E2E test suite completed successfully")
            return execution_results
            
        except Exception as e:
            self.logger.error(f"❌ E2E test execution failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
                "partial_results": getattr(self.test_executor, "test_results", {})
            }
    
    async def _execute_pipeline_validation(self) -> Dict[str, Any]:
        """Execute pipeline validation on test results."""
        self.logger.info("🔍 Executing Pipeline Validation...")
        
        validation_results = {}
        
        try:
            # Find job output directories from execution results
            execution_results = self.comprehensive_results.get("execution_results", {})
            pipeline_steps = execution_results.get("pipeline_steps", {})
            
            # Validate each completed job
            for phase_name, phase_data in pipeline_steps.items():
                if "job_id" in phase_data or "pants_job_id" in phase_data:
                    # Extract job ID
                    job_id = (
                        phase_data.get("job_id") or 
                        phase_data.get("pants_job_id") or
                        phase_data.get("pipeline_steps", {}).get("sku_analysis", {}).get("job_id")
                    )
                    
                    if job_id:
                        job_dir = Path(f"production_output/{job_id}")
                        if job_dir.exists():
                            self.logger.info(f"Validating job: {job_id}")
                            
                            job_validation = await self.validation_orchestrator.validate_complete_pipeline(job_dir)
                            validation_results[f"job_{job_id}"] = job_validation
            
            # Generate validation reports
            for job_name, job_validation in validation_results.items():
                report = self.validation_orchestrator.generate_validation_report(
                    job_validation,
                    Path(f"test_logs/validation_report_{job_name}_{self.session_id}.json")
                )
                validation_results[f"{job_name}_report"] = report
            
            self.logger.info("✅ Pipeline validation completed")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline validation failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance against baselines."""
        self.logger.info("⚡ Analyzing Performance Metrics...")
        
        performance_analysis = {
            "baseline_comparison": {},
            "bottlenecks_identified": [],
            "performance_score": 0.0,
            "optimization_opportunities": []
        }
        
        try:
            execution_results = self.comprehensive_results.get("execution_results", {})
            performance_metrics = execution_results.get("performance_metrics", {})
            
            # Compare against baselines
            for stage, baseline in PERFORMANCE_BASELINES.items():
                stage_performance = performance_metrics.get("performance_benchmarks", {})
                
                comparison = {
                    "stage": stage,
                    "baseline_met": True,
                    "metrics": {}
                }
                
                for metric, expected_value in baseline.items():
                    actual_value = stage_performance.get(metric, 0)
                    
                    if "max_" in metric:
                        meets_baseline = actual_value <= expected_value
                    elif "min_" in metric:
                        meets_baseline = actual_value >= expected_value
                    else:
                        meets_baseline = True
                    
                    comparison["metrics"][metric] = {
                        "expected": expected_value,
                        "actual": actual_value,
                        "meets_baseline": meets_baseline
                    }
                    
                    if not meets_baseline:
                        comparison["baseline_met"] = False
                        performance_analysis["bottlenecks_identified"].append({
                            "stage": stage,
                            "metric": metric,
                            "expected": expected_value,
                            "actual": actual_value
                        })
                
                performance_analysis["baseline_comparison"][stage] = comparison
            
            # Calculate overall performance score
            total_checks = sum(
                len(stage_data["metrics"]) 
                for stage_data in performance_analysis["baseline_comparison"].values()
            )
            
            passed_checks = sum(
                len([m for m in stage_data["metrics"].values() if m["meets_baseline"]])
                for stage_data in performance_analysis["baseline_comparison"].values()
            )
            
            performance_analysis["performance_score"] = passed_checks / total_checks if total_checks > 0 else 0.0
            
            self.logger.info(f"✅ Performance analysis completed: {performance_analysis['performance_score']:.2f} score")
            return performance_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Performance analysis failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def _assess_quality(self) -> Dict[str, Any]:
        """Assess output quality against criteria."""
        self.logger.info("🎯 Assessing Output Quality...")
        
        quality_assessment = {
            "criteria_evaluation": {},
            "quality_score": 0.0,
            "quality_issues": [],
            "data_integrity_status": "UNKNOWN"
        }
        
        try:
            validation_results = self.comprehensive_results.get("validation_results", {})
            
            # Evaluate quality criteria
            for criteria_category, criteria_items in QUALITY_CRITERIA.items():
                category_results = {
                    "category": criteria_category,
                    "criteria_met": {},
                    "overall_score": 0.0
                }
                
                criteria_passed = 0
                total_criteria = len(criteria_items)
                
                for criterion, expected_value in criteria_items.items():
                    # Extract relevant data from validation results
                    criterion_met = self._evaluate_quality_criterion(
                        criterion, expected_value, validation_results
                    )
                    
                    category_results["criteria_met"][criterion] = criterion_met
                    
                    if criterion_met:
                        criteria_passed += 1
                    else:
                        quality_assessment["quality_issues"].append({
                            "category": criteria_category,
                            "criterion": criterion,
                            "expected": expected_value
                        })
                
                category_results["overall_score"] = criteria_passed / total_criteria if total_criteria > 0 else 0.0
                quality_assessment["criteria_evaluation"][criteria_category] = category_results
            
            # Calculate overall quality score
            category_scores = [
                cat_data["overall_score"]
                for cat_data in quality_assessment["criteria_evaluation"].values()
            ]
            
            quality_assessment["quality_score"] = sum(category_scores) / len(category_scores) if category_scores else 0.0
            
            # Determine data integrity status
            if quality_assessment["quality_score"] >= 0.9:
                quality_assessment["data_integrity_status"] = "EXCELLENT"
            elif quality_assessment["quality_score"] >= 0.7:
                quality_assessment["data_integrity_status"] = "GOOD"
            elif quality_assessment["quality_score"] >= 0.5:
                quality_assessment["data_integrity_status"] = "ACCEPTABLE"
            else:
                quality_assessment["data_integrity_status"] = "NEEDS_IMPROVEMENT"
            
            self.logger.info(f"✅ Quality assessment completed: {quality_assessment['data_integrity_status']}")
            return quality_assessment
            
        except Exception as e:
            self.logger.error(f"❌ Quality assessment failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    def _evaluate_quality_criterion(self, criterion: str, expected_value: Any, validation_results: Dict[str, Any]) -> bool:
        """Evaluate a specific quality criterion against validation results."""
        
        # This is a simplified evaluation - in practice, you'd implement
        # detailed logic for each criterion based on validation results
        
        if criterion == "no_null_required_fields":
            # Check if validation found any missing required fields
            for job_validation in validation_results.values():
                if isinstance(job_validation, dict):
                    for stage_results in job_validation.values():
                        if isinstance(stage_results, list):
                            for result in stage_results:
                                if hasattr(result, 'status') and result.status == "FAIL":
                                    if "missing" in result.message.lower():
                                        return False
            return True
        
        elif criterion == "all_required_files_present":
            # Check validation results for file existence
            for job_validation in validation_results.values():
                if isinstance(job_validation, dict):
                    for stage_results in job_validation.values():
                        if isinstance(stage_results, list):
                            for result in stage_results:
                                if (hasattr(result, 'check_name') and 
                                    "required_file" in result.check_name and 
                                    hasattr(result, 'status') and 
                                    result.status == "FAIL"):
                                    return False
            return True
        
        # Default: assume criterion is met if no specific logic implemented
        return True
    
    async def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate improvement recommendations based on test results."""
        self.logger.info("💡 Generating Recommendations...")
        
        recommendations = []
        
        try:
            performance_analysis = self.comprehensive_results.get("performance_analysis", {})
            quality_assessment = self.comprehensive_results.get("quality_assessment", {})
            
            # Performance-based recommendations
            for bottleneck in performance_analysis.get("bottlenecks_identified", []):
                recommendations.append({
                    "type": "PERFORMANCE",
                    "priority": "HIGH",
                    "category": bottleneck["stage"],
                    "issue": f"{bottleneck['metric']} not meeting baseline",
                    "recommendation": f"Optimize {bottleneck['stage']} stage to improve {bottleneck['metric']}",
                    "expected_impact": "Improved processing speed and resource efficiency"
                })
            
            # Quality-based recommendations
            for issue in quality_assessment.get("quality_issues", []):
                recommendations.append({
                    "type": "QUALITY",
                    "priority": "MEDIUM",
                    "category": issue["category"],
                    "issue": f"{issue['criterion']} not meeting criteria",
                    "recommendation": f"Implement validation for {issue['criterion']}",
                    "expected_impact": "Improved data integrity and reliability"
                })
            
            # General recommendations based on overall results
            execution_results = self.comprehensive_results.get("execution_results", {})
            if execution_results.get("execution_summary", {}).get("errors_encountered", 0) > 0:
                recommendations.append({
                    "type": "RELIABILITY",
                    "priority": "HIGH",
                    "category": "error_handling",
                    "issue": "Errors encountered during execution",
                    "recommendation": "Implement robust error handling and recovery mechanisms",
                    "expected_impact": "Improved system reliability and user experience"
                })
            
            self.logger.info(f"✅ Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Recommendation generation failed: {e}")
            return [{
                "type": "ERROR",
                "priority": "HIGH",
                "category": "test_framework",
                "issue": f"Recommendation generation failed: {e}",
                "recommendation": "Review test framework implementation",
                "expected_impact": "Improved test coverage and reliability"
            }]
    
    async def _generate_final_report(self) -> None:
        """Generate comprehensive final test report."""
        self.logger.info("📋 Generating Final Test Report...")
        
        try:
            # Add session completion metadata
            self.comprehensive_results["session_metadata"]["end_time"] = datetime.now().isoformat()
            self.comprehensive_results["session_metadata"]["duration_seconds"] = (
                datetime.now() - self.test_start_time
            ).total_seconds()
            
            # Generate executive summary
            execution_status = self.comprehensive_results.get("execution_results", {}).get("execution_summary", {}).get("status", "UNKNOWN")
            performance_score = self.comprehensive_results.get("performance_analysis", {}).get("performance_score", 0.0)
            quality_score = self.comprehensive_results.get("quality_assessment", {}).get("quality_score", 0.0)
            
            executive_summary = {
                "test_session_status": execution_status,
                "performance_score": performance_score,
                "quality_score": quality_score,
                "overall_grade": self._calculate_overall_grade(performance_score, quality_score),
                "total_recommendations": len(self.comprehensive_results.get("recommendations", [])),
                "key_findings": self._extract_key_findings()
            }
            
            self.comprehensive_results["executive_summary"] = executive_summary
            
            # Save comprehensive report
            report_filename = f"comprehensive_test_report_{self.session_id}.json"
            report_path = Path("test_logs") / report_filename
            
            with report_path.open('w') as f:
                json.dump(self.comprehensive_results, f, indent=2, default=str)
            
            # Generate human-readable summary
            await self._generate_human_readable_summary(report_path)
            
            self.logger.info(f"✅ Final report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Final report generation failed: {e}")
    
    def _calculate_overall_grade(self, performance_score: float, quality_score: float) -> str:
        """Calculate overall test grade."""
        combined_score = (performance_score + quality_score) / 2
        
        if combined_score >= 0.9:
            return "A"
        elif combined_score >= 0.8:
            return "B"
        elif combined_score >= 0.7:
            return "C"
        elif combined_score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from test results."""
        findings = []
        
        # Extract from execution results
        execution_results = self.comprehensive_results.get("execution_results", {})
        if execution_results.get("execution_summary", {}).get("status") == "COMPLETED":
            findings.append("End-to-end workflow executed successfully")
        
        # Extract from validation results
        validation_results = self.comprehensive_results.get("validation_results", {})
        if validation_results:
            findings.append("Pipeline validation completed with detailed stage analysis")
        
        # Extract from performance analysis
        performance_analysis = self.comprehensive_results.get("performance_analysis", {})
        if performance_analysis.get("performance_score", 0) > 0.8:
            findings.append("Performance meets or exceeds baseline expectations")
        
        return findings
    
    async def _generate_human_readable_summary(self, report_path: Path) -> None:
        """Generate human-readable test summary."""
        summary_path = report_path.with_suffix('.md')
        
        executive_summary = self.comprehensive_results.get("executive_summary", {})
        
        summary_content = f"""# Comprehensive AI Mapping Workflow Test Report

## Executive Summary

- **Test Session**: {self.session_id}
- **Status**: {executive_summary.get('test_session_status', 'UNKNOWN')}
- **Overall Grade**: {executive_summary.get('overall_grade', 'N/A')}
- **Performance Score**: {executive_summary.get('performance_score', 0):.2f}
- **Quality Score**: {executive_summary.get('quality_score', 0):.2f}
- **Duration**: {self.comprehensive_results['session_metadata']['duration_seconds']:.2f} seconds

## Key Findings

{chr(10).join(f'- {finding}' for finding in executive_summary.get('key_findings', []))}

## Recommendations ({executive_summary.get('total_recommendations', 0)} total)

{chr(10).join(f'- **{rec["type"]}** ({rec["priority"]}): {rec["recommendation"]}' for rec in self.comprehensive_results.get('recommendations', [])[:5])}

## Test Configuration

- **Detailed Logs**: {self.args.detailed_logs}
- **API Testing**: {not self.args.skip_ai_mapping if hasattr(self.args, 'skip_ai_mapping') else True}

---
*Report generated on {datetime.now().isoformat()}*
"""
        
        with summary_path.open('w') as f:
            f.write(summary_content)
        
        self.logger.info(f"Human-readable summary: {summary_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive AI Mapping Workflow Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--detailed-logs",
        action="store_true",
        help="Enable detailed debug logging"
    )
    
    parser.add_argument(
        "--skip-ai-mapping",
        action="store_true", 
        help="Skip AI mapping phase (for testing without API)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_logs"),
        help="Output directory for test reports"
    )
    
    return parser.parse_args()


async def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Ensure output directory exists
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check environment prerequisites
    if not os.getenv("GOOGLE_API_KEY") and not args.skip_ai_mapping:
        print("❌ GOOGLE_API_KEY environment variable required for AI mapping tests")
        print("   Set API key or use --skip-ai-mapping flag")
        sys.exit(1)
    
    # Initialize and run comprehensive test
    test_runner = ComprehensiveTestRunner(args)
    
    try:
        results = await test_runner.run_comprehensive_test()
        
        print("\n🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        executive_summary = results.get("executive_summary", {})
        print(f"Overall Grade: {executive_summary.get('overall_grade', 'N/A')}")
        print(f"Performance Score: {executive_summary.get('performance_score', 0):.2f}")
        print(f"Quality Score: {executive_summary.get('quality_score', 0):.2f}")
        print(f"Total Recommendations: {executive_summary.get('total_recommendations', 0)}")
        
        print(f"\n📁 Check {args.output_dir}/ for detailed reports")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ COMPREHENSIVE TEST FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))