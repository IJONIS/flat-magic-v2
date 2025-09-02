#!/usr/bin/env python3
"""
Test Validation Framework for AI Mapping Workflow

Provides detailed validation criteria, baseline expectations, and quality assessment
for each pipeline stage with comprehensive output verification.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    check_name: str
    status: str  # PASS, FAIL, WARNING, SKIP
    message: str
    details: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None


@dataclass
class StageValidationCriteria:
    """Validation criteria for a specific pipeline stage."""
    
    stage_name: str
    required_files: List[str]
    optional_files: List[str] = field(default_factory=list)
    data_structure_requirements: Dict[str, Any] = field(default_factory=dict)
    performance_baselines: Dict[str, Union[int, float]] = field(default_factory=dict)
    quality_thresholds: Dict[str, Union[int, float]] = field(default_factory=dict)


class BaseValidator(ABC):
    """Base class for pipeline stage validators."""
    
    def __init__(self, criteria: StageValidationCriteria):
        self.criteria = criteria
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.validation_results: List[ValidationResult] = []
    
    @abstractmethod
    async def validate(self, output_dir: Path, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Execute validation for this stage."""
        pass
    
    def _check_file_existence(self, output_dir: Path) -> List[ValidationResult]:
        """Check if required and optional files exist."""
        results = []
        
        # Check required files
        for required_file in self.criteria.required_files:
            file_path = output_dir / required_file
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                results.append(ValidationResult(
                    check_name=f"required_file_{required_file}",
                    status="PASS",
                    message=f"Required file exists: {required_file}",
                    details={"file_path": str(file_path), "size_bytes": file_size}
                ))
            else:
                results.append(ValidationResult(
                    check_name=f"required_file_{required_file}",
                    status="FAIL", 
                    message=f"Required file missing: {required_file}",
                    details={"expected_path": str(file_path)}
                ))
        
        # Check optional files
        for optional_file in self.criteria.optional_files:
            file_path = output_dir / optional_file
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                results.append(ValidationResult(
                    check_name=f"optional_file_{optional_file}",
                    status="PASS",
                    message=f"Optional file exists: {optional_file}",
                    details={"file_path": str(file_path), "size_bytes": file_size}
                ))
        
        return results
    
    def _validate_json_structure(self, file_path: Path, expected_structure: Dict[str, Any]) -> ValidationResult:
        """Validate JSON file structure against expected schema."""
        try:
            with file_path.open('r') as f:
                data = json.load(f)
            
            validation_errors = []
            self._check_structure_recursive(data, expected_structure, validation_errors)
            
            if not validation_errors:
                return ValidationResult(
                    check_name=f"json_structure_{file_path.name}",
                    status="PASS",
                    message=f"JSON structure validation passed for {file_path.name}",
                    details={"file_path": str(file_path)}
                )
            else:
                return ValidationResult(
                    check_name=f"json_structure_{file_path.name}",
                    status="FAIL",
                    message=f"JSON structure validation failed: {', '.join(validation_errors)}",
                    details={"errors": validation_errors, "file_path": str(file_path)}
                )
                
        except json.JSONDecodeError as e:
            return ValidationResult(
                check_name=f"json_structure_{file_path.name}",
                status="FAIL",
                message=f"Invalid JSON format: {e}",
                details={"file_path": str(file_path), "json_error": str(e)}
            )
        except Exception as e:
            return ValidationResult(
                check_name=f"json_structure_{file_path.name}",
                status="FAIL",
                message=f"Validation error: {e}",
                details={"file_path": str(file_path), "error": str(e)}
            )
    
    def _check_structure_recursive(self, data: Any, expected: Dict[str, Any], errors: List[str], path: str = "") -> None:
        """Recursively check data structure against expected structure."""
        for key, expected_type in expected.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in data:
                errors.append(f"Missing required key: {current_path}")
                continue
            
            value = data[key]
            
            if isinstance(expected_type, type):
                if not isinstance(value, expected_type):
                    errors.append(f"Type mismatch at {current_path}: expected {expected_type.__name__}, got {type(value).__name__}")
            elif isinstance(expected_type, dict):
                if isinstance(value, dict):
                    self._check_structure_recursive(value, expected_type, errors, current_path)
                else:
                    errors.append(f"Type mismatch at {current_path}: expected dict, got {type(value).__name__}")


class SKUAnalysisValidator(BaseValidator):
    """Validator for SKU analysis stage."""
    
    async def validate(self, output_dir: Path, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate SKU analysis results."""
        results = []
        
        # File existence check
        results.extend(self._check_file_existence(output_dir))
        
        # Analysis results validation
        analysis_file = output_dir / "analysis_results.json"
        if analysis_file.exists():
            # Structure validation
            expected_structure = {
                "total_skus": int,
                "relationships": dict,
                "processing_metadata": dict
            }
            
            results.append(self._validate_json_structure(analysis_file, expected_structure))
            
            # Quality thresholds
            with analysis_file.open('r') as f:
                data = json.load(f)
            
            total_skus = data.get("total_skus", 0)
            relationship_count = len(data.get("relationships", {}))
            
            # Validate minimum data quality
            if total_skus >= self.criteria.quality_thresholds.get("min_skus", 100):
                results.append(ValidationResult(
                    check_name="sku_quantity_threshold",
                    status="PASS",
                    message=f"SKU quantity meets threshold: {total_skus} SKUs",
                    details={"total_skus": total_skus}
                ))
            else:
                results.append(ValidationResult(
                    check_name="sku_quantity_threshold",
                    status="WARNING",
                    message=f"SKU quantity below expected: {total_skus} SKUs",
                    details={"total_skus": total_skus}
                ))
            
            # Parent-child relationship quality
            if relationship_count > 0:
                results.append(ValidationResult(
                    check_name="parent_child_relationships",
                    status="PASS",
                    message=f"Parent-child relationships detected: {relationship_count} parents",
                    details={"parent_count": relationship_count}
                ))
            else:
                results.append(ValidationResult(
                    check_name="parent_child_relationships",
                    status="WARNING",
                    message="No parent-child relationships detected",
                    details={"parent_count": 0}
                ))
        
        return results


class CSVExportValidator(BaseValidator):
    """Validator for CSV export stage."""
    
    async def validate(self, output_dir: Path, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate CSV export results."""
        results = []
        
        csv_dir = output_dir / "csv_splits"
        
        if csv_dir.exists():
            csv_files = list(csv_dir.glob("*.csv"))
            
            if csv_files:
                results.append(ValidationResult(
                    check_name="csv_export_files",
                    status="PASS",
                    message=f"CSV export successful: {len(csv_files)} files",
                    details={"csv_file_count": len(csv_files), "files": [f.name for f in csv_files]}
                ))
                
                # Validate CSV file sizes
                total_size = sum(f.stat().st_size for f in csv_files)
                if total_size > 0:
                    results.append(ValidationResult(
                        check_name="csv_file_sizes",
                        status="PASS",
                        message=f"CSV files contain data: {total_size:,} bytes total",
                        details={"total_size_bytes": total_size}
                    ))
            else:
                results.append(ValidationResult(
                    check_name="csv_export_files",
                    status="FAIL",
                    message="No CSV files found in export directory",
                    details={"csv_directory": str(csv_dir)}
                ))
        else:
            results.append(ValidationResult(
                check_name="csv_export_directory",
                status="FAIL",
                message="CSV export directory not found",
                details={"expected_directory": str(csv_dir)}
            ))
        
        return results


class CompressionValidator(BaseValidator):
    """Validator for JSON compression stage."""
    
    async def validate(self, output_dir: Path, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate compression results."""
        results = []
        
        # Find compressed files
        compressed_files = list(output_dir.glob("parent_*/step2_compressed.json"))
        
        if compressed_files:
            results.append(ValidationResult(
                check_name="compression_files",
                status="PASS",
                message=f"Compression successful: {len(compressed_files)} files",
                details={"compressed_file_count": len(compressed_files)}
            ))
            
            # Validate compression structure
            for compressed_file in compressed_files[:3]:  # Sample first 3
                expected_structure = {
                    "parent_sku": str,
                    "compressed_data": dict,
                    "compression_metadata": dict
                }
                
                result = self._validate_json_structure(compressed_file, expected_structure)
                results.append(result)
                
                # Check compression ratio if available
                with compressed_file.open('r') as f:
                    data = json.load(f)
                
                metadata = data.get("compression_metadata", {})
                if "compression_ratio" in metadata:
                    ratio = metadata["compression_ratio"]
                    if ratio > 0.5:  # Good compression
                        results.append(ValidationResult(
                            check_name=f"compression_ratio_{compressed_file.parent.name}",
                            status="PASS",
                            message=f"Good compression ratio: {ratio:.2f}",
                            details={"compression_ratio": ratio}
                        ))
                    else:
                        results.append(ValidationResult(
                            check_name=f"compression_ratio_{compressed_file.parent.name}",
                            status="WARNING",
                            message=f"Low compression ratio: {ratio:.2f}",
                            details={"compression_ratio": ratio}
                        ))
        else:
            results.append(ValidationResult(
                check_name="compression_files",
                status="FAIL",
                message="No compressed files found",
                details={"search_pattern": "parent_*/step2_compressed.json"}
            ))
        
        return results


class AIMappingValidator(BaseValidator):
    """Validator for AI mapping stage."""
    
    async def validate(self, output_dir: Path, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate AI mapping results."""
        results = []
        
        # Find AI mapping output files
        ai_mapping_files = list(output_dir.glob("parent_*/step3_ai_mapping.json"))
        
        if ai_mapping_files:
            results.append(ValidationResult(
                check_name="ai_mapping_files",
                status="PASS",
                message=f"AI mapping files found: {len(ai_mapping_files)}",
                details={"ai_mapping_file_count": len(ai_mapping_files)}
            ))
            
            # Validate AI mapping structure and quality
            for ai_file in ai_mapping_files:
                expected_structure = {
                    "parent_sku": str,
                    "mapped_fields": list,
                    "unmapped_mandatory": list,
                    "overall_confidence": float,
                    "processing_notes": str
                }
                
                result = self._validate_json_structure(ai_file, expected_structure)
                results.append(result)
                
                # Quality assessment
                with ai_file.open('r') as f:
                    data = json.load(f)
                
                mapped_fields = data.get("mapped_fields", [])
                overall_confidence = data.get("overall_confidence", 0.0)
                
                # Validate mapping quality
                if len(mapped_fields) > 0:
                    results.append(ValidationResult(
                        check_name=f"ai_mapping_coverage_{ai_file.parent.name}",
                        status="PASS",
                        message=f"Fields mapped: {len(mapped_fields)}",
                        details={"mapped_field_count": len(mapped_fields)}
                    ))
                else:
                    results.append(ValidationResult(
                        check_name=f"ai_mapping_coverage_{ai_file.parent.name}",
                        status="WARNING",
                        message="No fields were mapped",
                        details={"mapped_field_count": 0}
                    ))
                
                # Confidence assessment
                confidence_threshold = self.criteria.quality_thresholds.get("min_confidence", 0.5)
                if overall_confidence >= confidence_threshold:
                    results.append(ValidationResult(
                        check_name=f"ai_confidence_{ai_file.parent.name}",
                        status="PASS",
                        message=f"Confidence meets threshold: {overall_confidence:.2f}",
                        details={"confidence": overall_confidence}
                    ))
                else:
                    results.append(ValidationResult(
                        check_name=f"ai_confidence_{ai_file.parent.name}",
                        status="WARNING",
                        message=f"Confidence below threshold: {overall_confidence:.2f}",
                        details={"confidence": overall_confidence}
                    ))
        else:
            results.append(ValidationResult(
                check_name="ai_mapping_files",
                status="FAIL",
                message="No AI mapping files found",
                details={"search_pattern": "parent_*/step3_ai_mapping.json"}
            ))
        
        return results


class PipelineValidationOrchestrator:
    """Orchestrates validation across all pipeline stages."""
    
    def __init__(self):
        """Initialize validation orchestrator with stage criteria."""
        self.logger = logging.getLogger(__name__)
        
        # Define validation criteria for each stage
        self.stage_criteria = {
            "sku_analysis": StageValidationCriteria(
                stage_name="sku_analysis",
                required_files=["analysis_results.json"],
                data_structure_requirements={
                    "total_skus": int,
                    "relationships": dict
                },
                quality_thresholds={
                    "min_skus": 100,
                    "min_parents": 5
                }
            ),
            
            "csv_export": StageValidationCriteria(
                stage_name="csv_export",
                required_files=["csv_splits"],
                quality_thresholds={
                    "min_csv_files": 1
                }
            ),
            
            "compression": StageValidationCriteria(
                stage_name="compression",
                required_files=[],
                data_structure_requirements={
                    "parent_sku": str,
                    "compressed_data": dict
                },
                quality_thresholds={
                    "min_compression_ratio": 0.3
                }
            ),
            
            "ai_mapping": StageValidationCriteria(
                stage_name="ai_mapping",
                required_files=[],
                data_structure_requirements={
                    "parent_sku": str,
                    "mapped_fields": list,
                    "overall_confidence": float
                },
                quality_thresholds={
                    "min_confidence": 0.5,
                    "min_mapped_fields": 1
                }
            )
        }
        
        # Initialize validators
        self.validators = {
            "sku_analysis": SKUAnalysisValidator(self.stage_criteria["sku_analysis"]),
            "csv_export": CSVExportValidator(self.stage_criteria["csv_export"]),
            "compression": CompressionValidator(self.stage_criteria["compression"]),
            "ai_mapping": AIMappingValidator(self.stage_criteria["ai_mapping"])
        }
    
    async def validate_complete_pipeline(self, job_output_dir: Path) -> Dict[str, List[ValidationResult]]:
        """Validate complete pipeline execution."""
        self.logger.info(f"Starting pipeline validation for: {job_output_dir}")
        
        validation_results = {}
        
        for stage_name, validator in self.validators.items():
            self.logger.info(f"Validating stage: {stage_name}")
            
            try:
                stage_results = await validator.validate(job_output_dir)
                validation_results[stage_name] = stage_results
                
                # Log stage summary
                passed = len([r for r in stage_results if r.status == "PASS"])
                failed = len([r for r in stage_results if r.status == "FAIL"])
                warnings = len([r for r in stage_results if r.status == "WARNING"])
                
                self.logger.info(
                    f"Stage {stage_name}: {passed} passed, {failed} failed, {warnings} warnings"
                )
                
            except Exception as e:
                self.logger.error(f"Validation failed for stage {stage_name}: {e}")
                validation_results[stage_name] = [ValidationResult(
                    check_name=f"{stage_name}_validation_error",
                    status="FAIL",
                    message=f"Validation error: {e}",
                    details={"error": str(e)}
                )]
        
        return validation_results
    
    def generate_validation_report(
        self, 
        validation_results: Dict[str, List[ValidationResult]], 
        output_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        report = {
            "validation_summary": {
                "timestamp": json.dumps(None, default=str),
                "total_stages": len(validation_results),
                "total_checks": sum(len(results) for results in validation_results.values())
            },
            "stage_summaries": {},
            "detailed_results": validation_results,
            "overall_status": "UNKNOWN"
        }
        
        # Calculate summaries
        total_passed = 0
        total_failed = 0
        total_warnings = 0
        
        for stage_name, results in validation_results.items():
            stage_passed = len([r for r in results if r.status == "PASS"])
            stage_failed = len([r for r in results if r.status == "FAIL"])
            stage_warnings = len([r for r in results if r.status == "WARNING"])
            
            total_passed += stage_passed
            total_failed += stage_failed
            total_warnings += stage_warnings
            
            report["stage_summaries"][stage_name] = {
                "passed": stage_passed,
                "failed": stage_failed,
                "warnings": stage_warnings,
                "status": "FAIL" if stage_failed > 0 else ("WARNING" if stage_warnings > 0 else "PASS")
            }
        
        # Overall status
        if total_failed > 0:
            report["overall_status"] = "FAIL"
        elif total_warnings > 0:
            report["overall_status"] = "WARNING"
        else:
            report["overall_status"] = "PASS"
        
        report["validation_summary"]["passed"] = total_passed
        report["validation_summary"]["failed"] = total_failed
        report["validation_summary"]["warnings"] = total_warnings
        
        # Save report if requested
        if output_file:
            with output_file.open('w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Validation report saved: {output_file}")
        
        return report


# Performance baseline expectations
PERFORMANCE_BASELINES = {
    "sku_analysis": {
        "max_processing_time_seconds": 30.0,
        "max_memory_usage_mb": 512,
        "min_throughput_records_per_second": 1000
    },
    "csv_export": {
        "max_processing_time_seconds": 15.0,
        "max_file_size_mb": 100
    },
    "compression": {
        "max_processing_time_seconds": 10.0,
        "min_compression_ratio": 0.3
    },
    "ai_mapping": {
        "max_processing_time_seconds": 60.0,
        "max_api_response_time_seconds": 30.0,
        "min_confidence_score": 0.5
    }
}

# Quality assessment criteria
QUALITY_CRITERIA = {
    "data_integrity": {
        "no_null_required_fields": True,
        "consistent_data_types": True,
        "valid_sku_formats": True
    },
    "mapping_accuracy": {
        "min_mapped_field_percentage": 0.7,
        "max_unmapped_mandatory_fields": 5,
        "consistent_mapping_logic": True
    },
    "output_completeness": {
        "all_required_files_present": True,
        "all_parent_directories_processed": True,
        "no_empty_output_files": True
    }
}


if __name__ == "__main__":
    """Example usage of validation framework."""
    import asyncio
    
    async def demo_validation():
        """Demonstrate validation framework usage."""
        orchestrator = PipelineValidationOrchestrator()
        
        # Example: validate a job output directory
        job_dir = Path("production_output/example_job")
        
        if job_dir.exists():
            validation_results = await orchestrator.validate_complete_pipeline(job_dir)
            
            report = orchestrator.generate_validation_report(
                validation_results,
                Path("test_logs/validation_report.json")
            )
            
            print(f"Validation completed: {report['overall_status']}")
            print(f"Total checks: {report['validation_summary']['total_checks']}")
            print(f"Passed: {report['validation_summary']['passed']}")
            print(f"Failed: {report['validation_summary']['failed']}")
    
    asyncio.run(demo_validation())