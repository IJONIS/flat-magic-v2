"""Validation test suite for format compliance and example loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .example_loader import ExampleFormatLoader, FormatEnforcer
from .models import AITransformationOutput, Metadata, ParentData, VariantRecord


class FormatValidationTester:
    """Test suite for format validation and compliance."""
    
    def __init__(self, example_file_path: Path | None = None):
        """Initialize validation tester.
        
        Args:
            example_file_path: Path to example_output_ai.json
        """
        self.example_loader = ExampleFormatLoader(example_file_path)
        self.format_enforcer = FormatEnforcer(self.example_loader)
        self.logger = logging.getLogger(__name__)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive validation test suite.
        
        Returns:
            Test results summary
        """
        test_results = {
            "example_loading": self.test_example_loading(),
            "pydantic_validation": self.test_pydantic_validation(),
            "structure_validation": self.test_structure_validation(),
            "format_enforcement": self.test_format_enforcement(),
            "edge_cases": self.test_edge_cases()
        }
        
        # Calculate overall results
        total_tests = sum(len(result.get("individual_tests", [])) 
                         for result in test_results.values())
        passed_tests = sum(sum(1 for test in result.get("individual_tests", []) 
                              if test.get("passed", False))
                          for result in test_results.values())
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "overall_passed": passed_tests == total_tests
        }
        
        return test_results
    
    def test_example_loading(self) -> Dict[str, Any]:
        """Test example file loading and parsing."""
        tests = []
        
        # Test 1: Load example structure
        try:
            example_data = self.example_loader.load_example_structure()
            tests.append({
                "name": "Load example structure",
                "passed": isinstance(example_data, dict) and "metadata" in example_data,
                "details": f"Loaded {len(example_data)} top-level keys"
            })
        except Exception as e:
            tests.append({
                "name": "Load example structure",
                "passed": False,
                "error": str(e)
            })
        
        # Test 2: Get example model
        try:
            example_model = self.example_loader.get_example_model()
            tests.append({
                "name": "Parse example as Pydantic model",
                "passed": isinstance(example_model, AITransformationOutput),
                "details": f"Model type: {type(example_model).__name__}"
            })
        except Exception as e:
            tests.append({
                "name": "Parse example as Pydantic model", 
                "passed": False,
                "error": str(e)
            })
        
        # Test 3: Get required fields schema
        try:
            schema = self.example_loader.get_required_fields_schema()
            tests.append({
                "name": "Extract required fields schema",
                "passed": isinstance(schema, dict) and len(schema) > 0,
                "details": f"Schema has {len(schema)} field definitions"
            })
        except Exception as e:
            tests.append({
                "name": "Extract required fields schema",
                "passed": False,
                "error": str(e)
            })
        
        return {
            "category": "Example Loading",
            "individual_tests": tests,
            "passed": all(test.get("passed", False) for test in tests)
        }
    
    def test_pydantic_validation(self) -> Dict[str, Any]:
        """Test Pydantic model validation."""
        tests = []
        
        # Test 1: Valid data validation
        try:
            example_data = self.example_loader.load_example_structure()
            model = AITransformationOutput.model_validate(example_data)
            tests.append({
                "name": "Validate example data with Pydantic",
                "passed": True,
                "details": f"Successfully created {type(model).__name__}"
            })
        except Exception as e:
            tests.append({
                "name": "Validate example data with Pydantic",
                "passed": False,
                "error": str(e)
            })
        
        # Test 2: Invalid data validation
        invalid_data = {"invalid": "structure"}
        try:
            AITransformationOutput.model_validate(invalid_data)
            tests.append({
                "name": "Reject invalid data structure",
                "passed": False,
                "details": "Should have raised ValidationError"
            })
        except Exception as e:
            tests.append({
                "name": "Reject invalid data structure",
                "passed": True,
                "details": f"Correctly rejected: {type(e).__name__}"
            })
        
        # Test 3: Partial data validation
        partial_data = {
            "metadata": {
                "parent_id": "test",
                "job_id": "test_job",
                "transformation_timestamp": "2024-01-01T00:00:00Z",
                "ai_model": "test",
                "mapping_confidence": 0.5,
                "total_variants": 0
            },
            "parent_data": {
                "feed_product_type": "test",
                "brand_name": "test",
                "outer_material_type": "test",
                "target_gender": "test",
                "age_range_description": "test",
                "bottoms_size_system": "test",
                "bottoms_size_class": "test",
                "country_of_origin": "test",
                "department_name": "test",
                "recommended_browse_nodes": "test"
            },
            "variance_data": []
        }
        try:
            model = AITransformationOutput.model_validate(partial_data)
            tests.append({
                "name": "Validate minimal valid structure",
                "passed": True,
                "details": f"Created model with {len(model.variance_data)} variants"
            })
        except Exception as e:
            tests.append({
                "name": "Validate minimal valid structure",
                "passed": False,
                "error": str(e)
            })
        
        return {
            "category": "Pydantic Validation",
            "individual_tests": tests,
            "passed": all(test.get("passed", False) for test in tests)
        }
    
    def test_structure_validation(self) -> Dict[str, Any]:
        """Test structural validation rules."""
        tests = []
        
        # Test 1: Valid structure validation
        try:
            example_data = self.example_loader.load_example_structure()
            is_valid, errors = self.example_loader.validate_output_structure(example_data)
            tests.append({
                "name": "Validate example structure",
                "passed": is_valid and len(errors) == 0,
                "details": f"Valid: {is_valid}, Errors: {len(errors)}"
            })
        except Exception as e:
            tests.append({
                "name": "Validate example structure",
                "passed": False,
                "error": str(e)
            })
        
        # Test 2: Inconsistent metadata validation
        try:
            example_data = self.example_loader.load_example_structure()
            # Modify to create inconsistency
            example_data["metadata"]["total_variants"] = 999
            is_valid, errors = self.example_loader.validate_output_structure(example_data)
            tests.append({
                "name": "Detect metadata inconsistency",
                "passed": not is_valid and len(errors) > 0,
                "details": f"Correctly detected {len(errors)} errors"
            })
        except Exception as e:
            tests.append({
                "name": "Detect metadata inconsistency",
                "passed": False,
                "error": str(e)
            })
        
        return {
            "category": "Structure Validation",
            "individual_tests": tests,
            "passed": all(test.get("passed", False) for test in tests)
        }
    
    def test_format_enforcement(self) -> Dict[str, Any]:
        """Test format enforcement functionality."""
        tests = []
        
        # Test 1: Enforce format on compliant data
        try:
            example_data = self.example_loader.load_example_structure()
            enforced_data, warnings = self.format_enforcer.enforce_format(
                example_data, "test_parent"
            )
            tests.append({
                "name": "Enforce format on compliant data",
                "passed": len(warnings) == 0,
                "details": f"Warnings: {len(warnings)}"
            })
        except Exception as e:
            tests.append({
                "name": "Enforce format on compliant data",
                "passed": False,
                "error": str(e)
            })
        
        # Test 2: Transform legacy format
        legacy_data = {
            "parent_sku": "test_123",
            "parent_data": {"brand_name": "TestBrand", "feed_product_type": "pants"},
            "variance_data": {"colors": ["red", "blue"], "sizes": ["M", "L"]},
            "metadata": {"confidence": 0.8, "total_mapped_fields": 5}
        }
        try:
            enforced_data, warnings = self.format_enforcer.enforce_format(
                legacy_data, "test_123"
            )
            
            # Check if transformed to compliant format
            is_valid, errors = self.example_loader.validate_output_structure(enforced_data)
            tests.append({
                "name": "Transform legacy format",
                "passed": is_valid,
                "details": f"Valid: {is_valid}, Warnings: {len(warnings)}, Errors: {len(errors)}"
            })
        except Exception as e:
            tests.append({
                "name": "Transform legacy format",
                "passed": False,
                "error": str(e)
            })
        
        # Test 3: Handle invalid data gracefully
        invalid_data = {"completely": "invalid"}
        try:
            enforced_data, warnings = self.format_enforcer.enforce_format(
                invalid_data, "test_invalid", strict=False
            )
            tests.append({
                "name": "Handle invalid data gracefully",
                "passed": isinstance(enforced_data, dict) and "metadata" in enforced_data,
                "details": f"Returned template structure, warnings: {len(warnings)}"
            })
        except Exception as e:
            tests.append({
                "name": "Handle invalid data gracefully",
                "passed": False,
                "error": str(e)
            })
        
        return {
            "category": "Format Enforcement",
            "individual_tests": tests,
            "passed": all(test.get("passed", False) for test in tests)
        }
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error conditions."""
        tests = []
        
        # Test 1: Empty variance data
        try:
            template = self.example_loader.create_template_output("empty_test")
            is_valid, errors = self.example_loader.validate_output_structure(template)
            tests.append({
                "name": "Handle empty variance data",
                "passed": is_valid and template["metadata"]["total_variants"] == 0,
                "details": f"Valid: {is_valid}, Total variants: {template['metadata']['total_variants']}"
            })
        except Exception as e:
            tests.append({
                "name": "Handle empty variance data",
                "passed": False,
                "error": str(e)
            })
        
        # Test 2: Large variance data
        try:
            large_variance = [
                {
                    "item_sku": f"test_{i}",
                    "size_name": f"Size{i}",
                    "color_name": f"Color{i}",
                    "size_map": f"Size{i}",
                    "color_map": f"Color{i}"
                }
                for i in range(100)
            ]
            
            large_data = self.example_loader.create_template_output("large_test")
            large_data["variance_data"] = large_variance
            large_data["metadata"]["total_variants"] = len(large_variance)
            
            is_valid, errors = self.example_loader.validate_output_structure(large_data)
            tests.append({
                "name": "Handle large variance data",
                "passed": is_valid,
                "details": f"Valid: {is_valid}, Variants: {len(large_variance)}"
            })
        except Exception as e:
            tests.append({
                "name": "Handle large variance data",
                "passed": False,
                "error": str(e)
            })
        
        # Test 3: Unicode and special characters
        try:
            unicode_data = self.example_loader.create_template_output("unicode_test")
            unicode_data["parent_data"]["brand_name"] = "Größe & Füße™"
            unicode_data["parent_data"]["target_gender"] = "Männlich"
            unicode_data["variance_data"] = [{
                "item_sku": "unicode_test_1",
                "size_name": "größe",
                "color_name": "weiß",
                "size_map": "größe",
                "color_map": "Weiß"
            }]
            unicode_data["metadata"]["total_variants"] = 1
            
            is_valid, errors = self.example_loader.validate_output_structure(unicode_data)
            tests.append({
                "name": "Handle Unicode characters",
                "passed": is_valid,
                "details": f"Valid: {is_valid}, Errors: {len(errors)}"
            })
        except Exception as e:
            tests.append({
                "name": "Handle Unicode characters",
                "passed": False,
                "error": str(e)
            })
        
        return {
            "category": "Edge Cases",
            "individual_tests": tests,
            "passed": all(test.get("passed", False) for test in tests)
        }
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable test report.
        
        Args:
            results: Test results from run_all_tests()
            
        Returns:
            Formatted test report
        """
        lines = []
        lines.append("="*60)
        lines.append("FORMAT VALIDATION TEST REPORT")
        lines.append("="*60)
        
        # Summary
        summary = results.get("summary", {})
        lines.append(f"Total Tests: {summary.get('total_tests', 0)}")
        lines.append(f"Passed: {summary.get('passed_tests', 0)}")
        lines.append(f"Failed: {summary.get('failed_tests', 0)}")
        lines.append(f"Success Rate: {summary.get('success_rate', 0.0):.1%}")
        lines.append(f"Overall: {'PASSED' if summary.get('overall_passed') else 'FAILED'}")
        lines.append("")
        
        # Detailed results by category
        for category_key, category_result in results.items():
            if category_key == "summary":
                continue
                
            category_name = category_result.get("category", category_key)
            category_passed = category_result.get("passed", False)
            individual_tests = category_result.get("individual_tests", [])
            
            lines.append(f"{category_name}: {'PASSED' if category_passed else 'FAILED'}")
            lines.append("-" * len(category_name) + "--------")
            
            for test in individual_tests:
                test_name = test.get("name", "Unknown Test")
                test_passed = test.get("passed", False)
                status = "✓" if test_passed else "✗"
                
                lines.append(f"  {status} {test_name}")
                
                if "details" in test:
                    lines.append(f"    Details: {test['details']}")
                if "error" in test:
                    lines.append(f"    Error: {test['error']}")
            
            lines.append("")
        
        return "\n".join(lines)


def run_format_validation_tests() -> Dict[str, Any]:
    """Run format validation tests and return results.
    
    Returns:
        Complete test results
    """
    tester = FormatValidationTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    # Run tests when executed directly
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    tester = FormatValidationTester()
    results = tester.run_all_tests()
    
    # Print report
    print(tester.generate_test_report(results))
    
    # Exit with appropriate code
    sys.exit(0 if results["summary"]["overall_passed"] else 1)