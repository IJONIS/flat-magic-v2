#!/usr/bin/env python3
"""Test format enforcement for AI transformation prompts."""

import json
import pytest
from pathlib import Path
from datetime import datetime
from sku_analyzer.ai_mapping.prompts.templates import PromptTemplateManager


class TestFormatEnforcement:
    """Test AI prompt format enforcement."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.template_manager = PromptTemplateManager()
        self.example_output_path = Path(__file__).parent / "example_output_ai.json"
        
        # Load reference format
        with open(self.example_output_path, 'r', encoding='utf-8') as f:
            self.reference_format = json.load(f)
    
    def test_prompt_references_example_file(self):
        """Test that prompt explicitly references example_output_ai.json."""
        context = {
            "parent_sku": "4307",
            "mandatory_fields": {
                "brand_name": {
                    "display_name": "Brand Name",
                    "data_type": "string",
                    "unique_values": ["EIKO", "Nike", "Adidas"],
                    "constraints": {"max_length": 50}
                }
            },
            "product_data": {
                "parent_data": {"MANUFACTURER_NAME": "EIKO"},
                "data_rows": [{"SKU": "4307_40_44", "SIZE": "44", "COLOR": "Schwarz"}]
            }
        }
        
        prompt = self.template_manager.render_mapping_prompt(context)
        
        assert "example_output_ai.json" in prompt
        assert "EXACT JSON structure" in prompt
        assert "REFERENCE FORMAT" in prompt
    
    def test_prompt_shows_correct_structure(self):
        """Test that prompt shows the correct output structure."""
        context = {
            "parent_sku": "4307",
            "mandatory_fields": {},
            "product_data": {}
        }
        
        prompt = self.template_manager.render_mapping_prompt(context)
        
        # Check for correct structure sections
        assert '"metadata"' in prompt
        assert '"parent_data"' in prompt
        assert '"variance_data"' in prompt
        
        # Check for individual SKU format (not arrays)
        assert '"item_sku"' in prompt
        assert '"size_name"' in prompt 
        assert '"color_name"' in prompt
        assert '"size_map"' in prompt
        assert '"color_map"' in prompt
    
    def test_prompt_warns_against_incorrect_format(self):
        """Test that prompt explicitly warns against incorrect formats."""
        context = {
            "parent_sku": "4307",
            "mandatory_fields": {},
            "product_data": {}
        }
        
        prompt = self.template_manager.render_mapping_prompt(context)
        
        # Should warn against array format for variance_data
        assert "INCORRECT FORMAT" in prompt
        assert "DO NOT USE" in prompt
        assert '["44", "46", "48"]' in prompt
        
        # Should show correct individual SKU format
        assert "CORRECT FORMAT" in prompt
        assert "REQUIRED" in prompt
    
    def test_prompt_includes_validation_checklist(self):
        """Test that prompt includes format validation checklist."""
        context = {
            "parent_sku": "4307", 
            "mandatory_fields": {},
            "product_data": {}
        }
        
        prompt = self.template_manager.render_mapping_prompt(context)
        
        assert "FORMAT VALIDATION CHECKLIST" in prompt
        assert "Metadata Section" in prompt
        assert "Parent_data Section" in prompt
        assert "Variance_data Section" in prompt
        assert "[ ]" in prompt  # Checklist items
    
    def test_prompt_includes_transformation_examples(self):
        """Test that prompt includes concrete transformation examples."""
        context = {
            "parent_sku": "4307",
            "mandatory_fields": {},
            "product_data": {}
        }
        
        prompt = self.template_manager.render_mapping_prompt(context)
        
        assert "TRANSFORMATION EXAMPLES" in prompt
        assert "MANUFACTURER_NAME" in prompt
        assert "brand_name" in prompt
        assert "EIKO" in prompt  # Actual example value
    
    def test_format_validation_template_exists(self):
        """Test that format validation template exists and works."""
        context = {
            "ai_output": {
                "metadata": {"parent_id": "4307"},
                "parent_data": {},  # This should trigger validation error
                "variance_data": []
            },
            "validation_errors": ["Empty parent_data object"]
        }
        
        validation_prompt = self.template_manager.render_format_validation_prompt(context)
        
        assert "OUTPUT FORMAT VALIDATION" in validation_prompt
        assert "VALIDATION CHECKLIST" in validation_prompt
        assert "Empty parent_data object" in validation_prompt
        assert "FAILED" in validation_prompt
    
    def test_validation_catches_common_errors(self):
        """Test that validation template identifies common format errors."""
        # Test empty parent_data error
        context_empty_parent = {
            "ai_output": {"parent_data": {}},
            "validation_errors": ["Empty parent_data"]
        }
        
        prompt = self.template_manager.render_format_validation_prompt(context_empty_parent)
        assert "Empty parent_data" in prompt
        
        # Test wrong variance_data format error  
        context_wrong_variance = {
            "ai_output": {"variance_data": {"size_name": ["44", "46"]}},
            "validation_errors": ["Wrong variance_data format"]
        }
        
        prompt = self.template_manager.render_format_validation_prompt(context_wrong_variance)
        assert "Wrong variance_data format" in prompt
    
    def test_reference_format_matches_example_file(self):
        """Test that reference format in prompt matches example file structure."""
        context = {
            "parent_sku": "4307",
            "mandatory_fields": {},
            "product_data": {}
        }
        
        prompt = self.template_manager.render_mapping_prompt(context)
        
        # Extract the reference structure from prompt
        # This is between **REFERENCE STRUCTURE** and ```
        reference_start = prompt.find("**REFERENCE STRUCTURE")
        reference_json_start = prompt.find("```json", reference_start)
        reference_json_end = prompt.find("```", reference_json_start + 7)
        reference_json = prompt[reference_json_start + 7:reference_json_end].strip()
        
        try:
            reference_structure = json.loads(reference_json)
            
            # Check top-level keys match
            assert set(reference_structure.keys()) == set(self.reference_format.keys())
            
            # Check metadata structure
            assert "metadata" in reference_structure
            assert isinstance(reference_structure["metadata"], dict)
            
            # Check parent_data is object with actual values
            assert "parent_data" in reference_structure  
            assert isinstance(reference_structure["parent_data"], dict)
            assert len(reference_structure["parent_data"]) > 0
            
            # Check variance_data is array with SKU objects
            assert "variance_data" in reference_structure
            assert isinstance(reference_structure["variance_data"], list)
            if reference_structure["variance_data"]:
                first_variant = reference_structure["variance_data"][0]
                required_fields = {"item_sku", "size_name", "color_name", "size_map", "color_map"}
                assert set(first_variant.keys()) == required_fields
                
        except json.JSONDecodeError:
            pytest.fail("Reference structure in prompt is not valid JSON")
    
    def test_critical_instructions_present(self):
        """Test that critical format enforcement instructions are present."""
        context = {
            "parent_sku": "4307",
            "mandatory_fields": {},
            "product_data": {}
        }
        
        prompt = self.template_manager.render_mapping_prompt(context)
        
        critical_instructions = [
            "EXACT FORMAT ONLY",
            "NO PLACEHOLDERS", 
            "INDIVIDUAL SKU RECORDS",
            "SHARED VALUES",
            "COMPLETE TRANSFORMATION",
            "example_output_ai.json"
        ]
        
        for instruction in critical_instructions:
            assert instruction in prompt, f"Missing critical instruction: {instruction}"
    
    def test_field_mapping_examples_present(self):
        """Test that field mapping examples are included."""
        context = {
            "parent_sku": "4307",
            "mandatory_fields": {},
            "product_data": {}
        }
        
        prompt = self.template_manager.render_mapping_prompt(context)
        
        # Check for field mapping examples
        mapping_examples = [
            "MANUFACTURER_NAME → brand_name",
            "PRODUCT_TYPE/CATEGORY → feed_product_type",
            "MATERIAL → outer_material_type",
            "COUNTRY → country_of_origin"
        ]
        
        for example in mapping_examples:
            assert example in prompt, f"Missing field mapping example: {example}"


def run_format_enforcement_tests():
    """Run format enforcement tests."""
    test_instance = TestFormatEnforcement()
    
    try:
        print("🧪 Running format enforcement tests...")
        
        test_instance.setup_method()
        test_instance.test_prompt_references_example_file()
        print("✅ Prompt references example file correctly")
        
        test_instance.test_prompt_shows_correct_structure()  
        print("✅ Prompt shows correct output structure")
        
        test_instance.test_prompt_warns_against_incorrect_format()
        print("✅ Prompt warns against incorrect formats")
        
        test_instance.test_prompt_includes_validation_checklist()
        print("✅ Prompt includes validation checklist")
        
        test_instance.test_prompt_includes_transformation_examples()
        print("✅ Prompt includes transformation examples")
        
        test_instance.test_format_validation_template_exists()
        print("✅ Format validation template works")
        
        test_instance.test_validation_catches_common_errors()
        print("✅ Validation catches common errors")
        
        test_instance.test_reference_format_matches_example_file()
        print("✅ Reference format matches example file")
        
        test_instance.test_critical_instructions_present()
        print("✅ Critical instructions present")
        
        test_instance.test_field_mapping_examples_present()
        print("✅ Field mapping examples present")
        
        print("\n🎉 All format enforcement tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_format_enforcement_tests()
    exit(0 if success else 1)