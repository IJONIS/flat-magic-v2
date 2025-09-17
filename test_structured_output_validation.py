#!/usr/bin/env python3
"""
Validation test for structured output schema and data loading.

This script tests the data loading, validation, and schema components
without requiring the actual Gemini API key.

Usage:
    python test_structured_output_validation.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sku_analyzer.step5_mapping.schema import get_ai_mapping_schema, get_schema_field_mappings
from sku_analyzer.prompts.mapping_prompts import MappingPromptManager


class SchemaValidator:
    """Validates schema definitions and structure."""
    
    def __init__(self):
        """Initialize validator."""
        self.logger = logging.getLogger(__name__)
    
    def validate_schema_structure(self) -> Dict[str, Any]:
        """Validate the AI mapping schema structure.
        
        Returns:
            Validation results
        """
        try:
            schema = get_ai_mapping_schema()
            
            # Check top-level structure
            results = {
                "success": True,
                "schema_type": str(schema.type),
                "required_fields": schema.required,
                "has_parent_data": "parent_data" in schema.properties,
                "has_variants": "variants" in schema.properties,
                "errors": []
            }
            
            # Validate parent data schema
            if "parent_data" in schema.properties:
                parent_schema = schema.properties["parent_data"]
                results["parent_required_count"] = len(parent_schema.required) if parent_schema.required else 0
                results["parent_properties_count"] = len(parent_schema.properties) if parent_schema.properties else 0
            
            # Validate variants schema
            if "variants" in schema.properties:
                variants_schema = schema.properties["variants"]
                results["variants_is_array"] = str(variants_schema.type) == "Type.ARRAY"
                
                if hasattr(variants_schema, 'items') and variants_schema.items:
                    variant_item = variants_schema.items
                    results["variant_required_count"] = len(variant_item.required) if variant_item.required else 0
                    results["variant_properties_count"] = len(variant_item.properties) if variant_item.properties else 0
            
            self.logger.info(f"Schema validation successful: {results}")
            return results
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def validate_field_mappings(self) -> Dict[str, Any]:
        """Validate field mappings configuration.
        
        Returns:
            Validation results
        """
        try:
            mappings = get_schema_field_mappings()
            
            parent_mappings = [k for k, v in mappings.items() if v.startswith("parent_data.")]
            variant_mappings = [k for k, v in mappings.items() if v.startswith("variants.")]
            
            results = {
                "success": True,
                "total_mappings": len(mappings),
                "parent_mappings": len(parent_mappings),
                "variant_mappings": len(variant_mappings),
                "parent_fields": parent_mappings,
                "variant_fields": variant_mappings
            }
            
            self.logger.info(f"Field mappings validation successful: {results}")
            return results
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }


class DataValidator:
    """Validates production data loading and structure."""
    
    def __init__(self, job_id: str = "1756744213", parent_sku: str = "41282"):
        """Initialize validator.
        
        Args:
            job_id: Production job ID
            parent_sku: Parent SKU to test
        """
        self.job_id = job_id
        self.parent_sku = parent_sku
        self.project_root = Path(__file__).parent
        self.job_dir = self.project_root / "production_output" / job_id
        self.parent_dir = self.job_dir / f"parent_{parent_sku}"
        
        self.logger = logging.getLogger(__name__)
        self.loaded_data: Dict[str, Any] = {}
    
    def validate_data_files(self) -> Dict[str, Any]:
        """Validate production data files exist and are readable.
        
        Returns:
            Validation results
        """
        files_to_check = {
            "step4_template": self.job_dir / "flat_file_analysis" / "step4_template.json",
            "step3_mandatory": self.job_dir / "flat_file_analysis" / "step3_mandatory_fields.json", 
            "step2_compressed": self.parent_dir / "step2_compressed.json",
            "existing_mapping": self.parent_dir / "step5_ai_mapping.json"
        }
        
        results = {
            "success": True,
            "files_checked": len(files_to_check),
            "files_found": 0,
            "files_readable": 0,
            "file_details": {},
            "errors": []
        }
        
        for file_key, file_path in files_to_check.items():
            file_info = {
                "exists": file_path.exists(),
                "readable": False,
                "size_bytes": 0,
                "json_valid": False,
                "content_preview": ""
            }
            
            if file_info["exists"]:
                results["files_found"] += 1
                file_info["size_bytes"] = file_path.stat().st_size
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.loaded_data[file_key] = data
                    
                    file_info["readable"] = True
                    file_info["json_valid"] = True
                    file_info["content_preview"] = str(list(data.keys())[:5]) if isinstance(data, dict) else str(type(data).__name__)
                    results["files_readable"] += 1
                    
                except Exception as e:
                    file_info["error"] = str(e)
                    results["errors"].append(f"{file_key}: {e}")
            else:
                results["errors"].append(f"{file_key}: File not found at {file_path}")
            
            results["file_details"][file_key] = file_info
        
        results["success"] = results["files_readable"] == len(files_to_check)
        
        self.logger.info(f"Data files validation: {results['files_readable']}/{results['files_checked']} files readable")
        return results
    
    def validate_data_structure(self) -> Dict[str, Any]:
        """Validate loaded data has expected structure.
        
        Returns:
            Validation results
        """
        results = {
            "success": True,
            "structure_checks": {},
            "data_summary": {},
            "errors": []
        }
        
        # Validate step2_compressed structure
        if "step2_compressed" in self.loaded_data:
            step2 = self.loaded_data["step2_compressed"]
            step2_check = {
                "has_parent_data": "parent_data" in step2,
                "has_data_rows": "data_rows" in step2,
                "parent_field_count": len(step2.get("parent_data", {})),
                "variant_count": len(step2.get("data_rows", []))
            }
            results["structure_checks"]["step2_compressed"] = step2_check
            
            if not step2_check["has_parent_data"]:
                results["errors"].append("step2_compressed missing parent_data")
            if not step2_check["has_data_rows"]:
                results["errors"].append("step2_compressed missing data_rows")
        
        # Validate step3_mandatory structure
        if "step3_mandatory" in self.loaded_data:
            step3 = self.loaded_data["step3_mandatory"]
            # The file IS the mandatory fields (dictionary of field definitions)
            step3_check = {
                "has_mandatory_fields": isinstance(step3, dict) and len(step3) > 0,
                "field_count": len(step3) if isinstance(step3, dict) else 0
            }
            results["structure_checks"]["step3_mandatory"] = step3_check
            
            if not step3_check["has_mandatory_fields"]:
                results["errors"].append("step3_mandatory is not a valid field dictionary")
        
        # Validate step4_template structure
        if "step4_template" in self.loaded_data:
            step4 = self.loaded_data["step4_template"]
            step4_check = {
                "has_template_structure": "template_structure" in step4,
                "has_metadata": "metadata" in step4
            }
            results["structure_checks"]["step4_template"] = step4_check
        
        results["success"] = len(results["errors"]) == 0
        
        # Create data summary
        if results["success"]:
            results["data_summary"] = {
                "parent_fields": results["structure_checks"]["step2_compressed"]["parent_field_count"],
                "variants": results["structure_checks"]["step2_compressed"]["variant_count"], 
                "mandatory_fields": results["structure_checks"]["step3_mandatory"]["field_count"],
                "job_id": self.job_id,
                "parent_sku": self.parent_sku
            }
        
        self.logger.info(f"Data structure validation: {'PASSED' if results['success'] else 'FAILED'}")
        return results


class PromptValidator:
    """Validates prompt generation with real data."""
    
    def __init__(self, data_validator: DataValidator):
        """Initialize with data validator.
        
        Args:
            data_validator: Data validator with loaded data
        """
        self.data_validator = data_validator
        self.logger = logging.getLogger(__name__)
    
    def validate_prompt_generation(self) -> Dict[str, Any]:
        """Validate prompt generation with real data.
        
        Returns:
            Validation results
        """
        results = {
            "success": True,
            "prompt_length": 0,
            "contains_parent_sku": False,
            "contains_mandatory_fields": False,
            "contains_product_data": False,
            "errors": []
        }
        
        try:
            # Create prompt manager
            prompt_manager = MappingPromptManager()
            
            # Prepare context from loaded data
            context = {
                "parent_sku": self.data_validator.parent_sku,
                "mandatory_fields": self.data_validator.loaded_data["step3_mandatory"],  # This IS the mandatory fields
                "product_data": self.data_validator.loaded_data["step2_compressed"],
                "template_structure": self.data_validator.loaded_data.get("step4_template", {})
            }
            
            # Generate prompt
            prompt = prompt_manager.render_mapping_prompt(context)
            
            # Validate prompt content
            results["prompt_length"] = len(prompt)
            results["contains_parent_sku"] = self.data_validator.parent_sku in prompt
            results["contains_mandatory_fields"] = "mandatory_fields" in prompt or "required" in prompt.lower()
            results["contains_product_data"] = "parent_data" in prompt and "data_rows" in prompt
            
            # Check minimum prompt requirements
            if results["prompt_length"] < 100:
                results["errors"].append("Prompt too short (<100 chars)")
            
            if not results["contains_parent_sku"]:
                results["errors"].append("Prompt missing parent SKU")
            
            results["success"] = len(results["errors"]) == 0
            
            self.logger.info(f"Prompt generation validation: {results['prompt_length']} chars, "
                           f"success={results['success']}")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Prompt generation failed: {e}")
        
        return results


def run_validation_tests() -> Dict[str, Any]:
    """Run complete validation test suite.
    
    Returns:
        Complete test results
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("🧪 Structured Output Validation Test Suite")
    print("=" * 60)
    
    overall_results = {
        "success": True,
        "tests_run": 0,
        "tests_passed": 0,
        "test_results": {}
    }
    
    # Test 1: Schema validation
    print("\n1. Schema Structure Validation...")
    schema_validator = SchemaValidator()
    schema_results = schema_validator.validate_schema_structure()
    overall_results["test_results"]["schema_structure"] = schema_results
    overall_results["tests_run"] += 1
    if schema_results["success"]:
        overall_results["tests_passed"] += 1
        print(f"✅ Schema valid - Parent: {schema_results.get('parent_required_count', 0)} fields, "
              f"Variants: {schema_results.get('variant_required_count', 0)} fields")
    else:
        print(f"❌ Schema validation failed: {schema_results.get('error', 'Unknown error')}")
        overall_results["success"] = False
    
    # Test 2: Field mappings validation
    print("\n2. Field Mappings Validation...")
    mappings_results = schema_validator.validate_field_mappings()
    overall_results["test_results"]["field_mappings"] = mappings_results
    overall_results["tests_run"] += 1
    if mappings_results["success"]:
        overall_results["tests_passed"] += 1
        print(f"✅ Mappings valid - {mappings_results['total_mappings']} total "
              f"({mappings_results['parent_mappings']} parent, {mappings_results['variant_mappings']} variant)")
    else:
        print(f"❌ Mappings validation failed: {mappings_results.get('error', 'Unknown error')}")
        overall_results["success"] = False
    
    # Test 3: Data files validation
    print("\n3. Production Data Files Validation...")
    data_validator = DataValidator()
    files_results = data_validator.validate_data_files()
    overall_results["test_results"]["data_files"] = files_results
    overall_results["tests_run"] += 1
    if files_results["success"]:
        overall_results["tests_passed"] += 1
        print(f"✅ Data files valid - {files_results['files_readable']}/{files_results['files_checked']} files readable")
    else:
        print(f"❌ Data files validation failed:")
        for error in files_results["errors"]:
            print(f"   - {error}")
        overall_results["success"] = False
    
    # Test 4: Data structure validation (only if files loaded successfully)
    if files_results["success"]:
        print("\n4. Data Structure Validation...")
        structure_results = data_validator.validate_data_structure()
        overall_results["test_results"]["data_structure"] = structure_results
        overall_results["tests_run"] += 1
        if structure_results["success"]:
            overall_results["tests_passed"] += 1
            summary = structure_results["data_summary"]
            print(f"✅ Data structure valid - {summary['parent_fields']} parent fields, "
                  f"{summary['variants']} variants, {summary['mandatory_fields']} mandatory fields")
        else:
            print(f"❌ Data structure validation failed:")
            for error in structure_results["errors"]:
                print(f"   - {error}")
            overall_results["success"] = False
        
        # Test 5: Prompt generation validation (only if data structure is valid)
        if structure_results["success"]:
            print("\n5. Prompt Generation Validation...")
            prompt_validator = PromptValidator(data_validator)
            prompt_results = prompt_validator.validate_prompt_generation()
            overall_results["test_results"]["prompt_generation"] = prompt_results
            overall_results["tests_run"] += 1
            if prompt_results["success"]:
                overall_results["tests_passed"] += 1
                print(f"✅ Prompt generation valid - {prompt_results['prompt_length']} chars")
            else:
                print(f"❌ Prompt generation failed:")
                for error in prompt_results["errors"]:
                    print(f"   - {error}")
                overall_results["success"] = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION TEST SUMMARY")
    print("=" * 60)
    
    if overall_results["success"]:
        print(f"✅ ALL TESTS PASSED ({overall_results['tests_passed']}/{overall_results['tests_run']})")
        print("\n🚀 Ready for real Gemini API testing!")
        print("   Set GOOGLE_API_KEY and run: python test_real_gemini_structured_output.py")
    else:
        print(f"❌ TESTS FAILED ({overall_results['tests_passed']}/{overall_results['tests_run']} passed)")
        print("\n🔧 Fix validation issues before API testing")
    
    print("=" * 60)
    
    return overall_results


if __name__ == "__main__":
    try:
        results = run_validation_tests()
        sys.exit(0 if results["success"] else 1)
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)