#!/usr/bin/env python3
"""
Comprehensive test script for validating Gemini API structured output implementation.

This script uses real production data from job 1756744213 to test the complete
end-to-end structured output workflow with the actual Gemini API.

Usage:
    python test_real_gemini_structured_output.py

Requirements:
    - GOOGLE_API_KEY environment variable set
    - Real production data from job 1756744213
    - google-genai library installed
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sku_analyzer.shared.gemini_client import GeminiClient, AIProcessingConfig
from sku_analyzer.prompts.mapping_prompts import MappingPromptManager


class StructuredOutputValidator:
    """Validates structured output against expected schema requirements."""
    
    REQUIRED_PARENT_FIELDS = [
        "age_range_description",
        "bottoms_size_class", 
        "bottoms_size_system",
        "brand_name",
        "country_of_origin",
        "department_name",
        "external_product_id_type",
        "fabric_type",
        "feed_product_type",
        "item_name",
        "main_image_url",
        "outer_material_type",
        "recommended_browse_nodes",
        "target_gender"
    ]
    
    REQUIRED_VARIANT_FIELDS = [
        "color_map",
        "color_name",
        "external_product_id",
        "item_sku",
        "list_price_with_tax",
        "quantity",
        "size_map",
        "size_name",
        "standard_price"
    ]
    
    def __init__(self):
        """Initialize validator."""
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
    
    def validate_response_structure(self, response_data: Dict[str, Any]) -> bool:
        """Validate response has correct top-level structure.
        
        Args:
            response_data: Parsed JSON response from API
            
        Returns:
            True if structure is valid
        """
        # Check top-level required fields
        if not isinstance(response_data, dict):
            self.validation_errors.append("Response is not a dictionary")
            return False
        
        if "parent_data" not in response_data:
            self.validation_errors.append("Missing 'parent_data' field")
            return False
        
        if "variants" not in response_data:
            self.validation_errors.append("Missing 'variants' field")
            return False
        
        return True
    
    def validate_parent_data(self, parent_data: Dict[str, Any]) -> bool:
        """Validate parent data has all required fields.
        
        Args:
            parent_data: Parent data section
            
        Returns:
            True if parent data is valid
        """
        if not isinstance(parent_data, dict):
            self.validation_errors.append("Parent data is not a dictionary")
            return False
        
        # Check all required fields exist
        missing_fields = []
        for field in self.REQUIRED_PARENT_FIELDS:
            if field not in parent_data:
                missing_fields.append(field)
            elif not parent_data[field] or parent_data[field] == "":
                self.validation_warnings.append(f"Parent field '{field}' is empty")
        
        if missing_fields:
            self.validation_errors.append(f"Missing parent fields: {missing_fields}")
            return False
        
        return True
    
    def validate_variants_data(self, variants: List[Dict[str, Any]], expected_count: int = 28) -> bool:
        """Validate variants data structure.
        
        Args:
            variants: List of variant objects
            expected_count: Expected number of variants
            
        Returns:
            True if variants are valid
        """
        if not isinstance(variants, list):
            self.validation_errors.append("Variants is not a list")
            return False
        
        if len(variants) != expected_count:
            self.validation_warnings.append(f"Expected {expected_count} variants, got {len(variants)}")
        
        # Validate each variant
        for i, variant in enumerate(variants):
            if not isinstance(variant, dict):
                self.validation_errors.append(f"Variant {i} is not a dictionary")
                continue
            
            # Check required fields
            missing_fields = []
            for field in self.REQUIRED_VARIANT_FIELDS:
                if field not in variant:
                    missing_fields.append(field)
                elif not variant[field] or variant[field] == "":
                    self.validation_warnings.append(f"Variant {i} field '{field}' is empty")
            
            if missing_fields:
                self.validation_errors.append(f"Variant {i} missing fields: {missing_fields}")
        
        return len(self.validation_errors) == 0
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get complete validation summary.
        
        Returns:
            Validation summary with errors and warnings
        """
        return {
            "is_valid": len(self.validation_errors) == 0,
            "error_count": len(self.validation_errors),
            "warning_count": len(self.validation_warnings),
            "errors": self.validation_errors,
            "warnings": self.validation_warnings
        }


class RealDataTester:
    """Tests structured output using real production data."""
    
    def __init__(self, job_id: str = "1756744213", parent_sku: str = "41282"):
        """Initialize tester with job and parent SKU.
        
        Args:
            job_id: Production job ID
            parent_sku: Parent SKU to test
        """
        self.job_id = job_id
        self.parent_sku = parent_sku
        self.project_root = Path(__file__).parent
        self.job_dir = self.project_root / "production_output" / job_id
        self.parent_dir = self.job_dir / f"parent_{parent_sku}"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.validator = StructuredOutputValidator()
        self.prompt_manager = MappingPromptManager()
        
        # Data storage
        self.loaded_data: Dict[str, Any] = {}
        self.test_results: Dict[str, Any] = {}
    
    def load_production_data(self) -> bool:
        """Load all required production data files.
        
        Returns:
            True if all data loaded successfully
        """
        self.logger.info(f"Loading production data for job {self.job_id}, parent {self.parent_sku}")
        
        # Define required files
        files_to_load = {
            "step4_template": self.job_dir / "flat_file_analysis" / "step4_template.json",
            "step3_mandatory": self.job_dir / "flat_file_analysis" / "step3_mandatory_fields.json",
            "step2_compressed": self.parent_dir / "step2_compressed.json",
            "existing_mapping": self.parent_dir / "step5_ai_mapping.json"
        }
        
        # Load each file
        for file_key, file_path in files_to_load.items():
            if not file_path.exists():
                self.logger.error(f"Required file not found: {file_path}")
                return False
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.loaded_data[file_key] = json.load(f)
                self.logger.info(f"Loaded {file_key}: {len(json.dumps(self.loaded_data[file_key]))} chars")
            except Exception as e:
                self.logger.error(f"Error loading {file_key}: {e}")
                return False
        
        # Validate data structure
        if not self._validate_loaded_data():
            return False
        
        self.logger.info("All production data loaded successfully")
        return True
    
    def _validate_loaded_data(self) -> bool:
        """Validate loaded data has expected structure.
        
        Returns:
            True if data is valid
        """
        # Check step2_compressed has required structure
        step2_data = self.loaded_data.get("step2_compressed", {})
        if "parent_data" not in step2_data:
            self.logger.error("step2_compressed missing parent_data")
            return False
        
        if "data_rows" not in step2_data:
            self.logger.error("step2_compressed missing data_rows")
            return False
        
        variant_count = len(step2_data["data_rows"])
        self.logger.info(f"Found {variant_count} variants in step2_compressed")
        
        # Check step3_mandatory has field definitions (file IS the mandatory fields)
        step3_data = self.loaded_data.get("step3_mandatory", {})
        if not isinstance(step3_data, dict) or len(step3_data) == 0:
            self.logger.error("step3_mandatory is not a valid field dictionary")
            return False
        
        field_count = len(step3_data)
        self.logger.info(f"Found {field_count} mandatory fields")
        
        return True
    
    def create_test_prompt(self) -> str:
        """Create test prompt using real production data.
        
        Returns:
            Formatted prompt for AI mapping
        """
        self.logger.info("Creating test prompt from production data")
        
        # Extract data for prompt
        mandatory_fields = self.loaded_data["step3_mandatory"]  # This IS the mandatory fields
        product_data = self.loaded_data["step2_compressed"]
        template_structure = self.loaded_data.get("step4_template", {})
        
        # Create prompt context
        context = {
            "parent_sku": self.parent_sku,
            "mandatory_fields": mandatory_fields,
            "product_data": product_data,
            "template_structure": template_structure
        }
        
        # Generate the prompt
        prompt = self.prompt_manager.render_mapping_prompt(context)
        
        self.logger.info(f"Generated prompt: {len(prompt)} characters")
        return prompt
    
    async def test_structured_output(self) -> Dict[str, Any]:
        """Test structured output with real Gemini API.
        
        Returns:
            Complete test results
        """
        self.logger.info("Starting structured output API test")
        
        # Validate API key
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable required")
        
        # Create client with structured output enabled
        config = AIProcessingConfig(
            model_name="gemini-2.5-flash",
            temperature=0.3,
            enable_structured_output=True,
            thinking_budget=-1,
            timeout_seconds=30
        )
        
        client = GeminiClient(config=config)
        
        # Create test prompt
        prompt = self.create_test_prompt()
        
        # Record start time
        start_time = time.perf_counter()
        
        try:
            # Make structured API call
            response = await client.generate_structured_mapping(
                prompt=prompt,
                operation_name="real_data_test"
            )
            
            # Record response time
            response_time = (time.perf_counter() - start_time) * 1000  # ms
            
            # Parse JSON response
            response_data = await client.validate_json_response(response)
            
            # Validate structure
            structure_valid = self.validator.validate_response_structure(response_data)
            
            parent_valid = False
            variants_valid = False
            
            if structure_valid:
                parent_valid = self.validator.validate_parent_data(response_data["parent_data"])
                variants_valid = self.validator.validate_variants_data(response_data["variants"])
            
            # Compile results
            test_results = {
                "success": True,
                "response_time_ms": response_time,
                "prompt_length": len(prompt),
                "response_length": len(response.content),
                "api_response": {
                    "finish_reason": response.finish_reason,
                    "usage_metadata": response.usage_metadata,
                    "safety_ratings": response.safety_ratings
                },
                "validation": self.validator.get_validation_summary(),
                "structure_validation": {
                    "structure_valid": structure_valid,
                    "parent_valid": parent_valid,
                    "variants_valid": variants_valid
                },
                "data_summary": {
                    "parent_field_count": len(response_data.get("parent_data", {})),
                    "variant_count": len(response_data.get("variants", [])),
                    "total_fields": len(response_data.get("parent_data", {})) + 
                                   len(response_data.get("variants", [])) * 9 if response_data.get("variants") else 0
                },
                "response_data": response_data
            }
            
            self.logger.info(f"Structured output test completed successfully in {response_time:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"Structured output test failed: {e}")
            test_results = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "response_time_ms": (time.perf_counter() - start_time) * 1000,
                "prompt_length": len(prompt) if 'prompt' in locals() else 0
            }
        
        return test_results
    
    def compare_with_existing(self, new_response: Dict[str, Any]) -> Dict[str, Any]:
        """Compare new structured response with existing mapping.
        
        Args:
            new_response: New structured response data
            
        Returns:
            Comparison results
        """
        existing_data = self.loaded_data.get("existing_mapping", {})
        
        comparison = {
            "has_existing_mapping": bool(existing_data),
            "structure_comparison": {},
            "field_differences": []
        }
        
        if existing_data:
            # Compare field counts
            existing_parent_count = len(existing_data.get("parent_data", {}))
            new_parent_count = len(new_response.get("parent_data", {}))
            
            existing_variant_count = len(existing_data.get("variants", []))
            new_variant_count = len(new_response.get("variants", []))
            
            comparison["structure_comparison"] = {
                "parent_fields": {
                    "existing": existing_parent_count,
                    "new": new_parent_count,
                    "difference": new_parent_count - existing_parent_count
                },
                "variants": {
                    "existing": existing_variant_count,
                    "new": new_variant_count,
                    "difference": new_variant_count - existing_variant_count
                }
            }
            
            self.logger.info(f"Comparison: Parent fields {existing_parent_count}→{new_parent_count}, "
                           f"Variants {existing_variant_count}→{new_variant_count}")
        
        return comparison
    
    def print_test_summary(self, results: Dict[str, Any]) -> None:
        """Print comprehensive test summary.
        
        Args:
            results: Test results to summarize
        """
        print("\n" + "="*80)
        print("REAL GEMINI STRUCTURED OUTPUT TEST RESULTS")
        print("="*80)
        
        if results["success"]:
            print(f"✅ SUCCESS - API Response Time: {results['response_time_ms']:.1f}ms")
            
            # Validation summary
            validation = results["validation"]
            if validation["is_valid"]:
                print("✅ VALIDATION PASSED")
            else:
                print(f"❌ VALIDATION FAILED - {validation['error_count']} errors")
                for error in validation["errors"]:
                    print(f"   ERROR: {error}")
            
            if validation["warning_count"] > 0:
                print(f"⚠️  {validation['warning_count']} warnings:")
                for warning in validation["warnings"]:
                    print(f"   WARNING: {warning}")
            
            # Structure summary
            structure = results["structure_validation"]
            print(f"📊 STRUCTURE: Parent={structure['parent_valid']}, Variants={structure['variants_valid']}")
            
            # Data summary
            data = results["data_summary"]
            print(f"📈 DATA: {data['parent_field_count']} parent fields, {data['variant_count']} variants")
            
            # API metrics
            api = results["api_response"]
            if api.get("usage_metadata"):
                tokens = api["usage_metadata"]
                print(f"🔢 TOKENS: {tokens.get('total_token_count', 0)} total "
                      f"({tokens.get('prompt_token_count', 0)} prompt + {tokens.get('candidates_token_count', 0)} response)")
            
            # Performance check
            if results['response_time_ms'] <= 5000:
                print("🚀 PERFORMANCE: Excellent (<5s)")
            elif results['response_time_ms'] <= 10000:
                print("⚡ PERFORMANCE: Good (<10s)")
            else:
                print("🐌 PERFORMANCE: Slow (>10s)")
            
        else:
            print(f"❌ FAILED - {results.get('error_type', 'Unknown Error')}")
            print(f"   Error: {results.get('error', 'No details')}")
            print(f"   Time: {results.get('response_time_ms', 0):.1f}ms")
        
        print("="*80)
    
    async def run_complete_test(self) -> Dict[str, Any]:
        """Run complete end-to-end test.
        
        Returns:
            Complete test results
        """
        print(f"🧪 Testing Gemini Structured Output with Job {self.job_id}, Parent {self.parent_sku}")
        
        # Load production data
        if not self.load_production_data():
            return {"success": False, "error": "Failed to load production data"}
        
        # Run structured output test
        test_results = await self.test_structured_output()
        
        # Compare with existing if successful
        if test_results["success"] and "response_data" in test_results:
            comparison = self.compare_with_existing(test_results["response_data"])
            test_results["comparison"] = comparison
        
        # Print summary
        self.print_test_summary(test_results)
        
        return test_results


async def main():
    """Main test execution."""
    print("🔬 Real Gemini Structured Output Test")
    print("Using production data from job 1756744213, parent 41282")
    print("-" * 60)
    
    # Check requirements
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable required")
        return 1
    
    try:
        # Run test
        tester = RealDataTester()
        results = await tester.run_complete_test()
        
        # Return appropriate exit code
        return 0 if results["success"] else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Run async test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)