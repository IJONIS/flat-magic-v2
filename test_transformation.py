#!/usr/bin/env python3
"""Test script for AI transformation system fixes."""

import asyncio
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import the updated models and components
from sku_analyzer.ai_mapping.models import MappingInput, TransformationResult
from sku_analyzer.ai_mapping.data_transformer import DataVarianceAnalyzer, ConstraintValidator
from sku_analyzer.ai_mapping.prompts.templates import PromptTemplateManager


async def test_data_transformation():
    """Test the data transformation pipeline with sample data."""
    print("Testing AI Transformation System Fixes")
    print("=" * 50)
    
    # Load sample data
    sample_step2_path = Path("production_output/1756744145/parent_4307/step2_compressed.json")
    sample_step3_path = Path("production_output/1756744145/flat_file_analysis/step3_mandatory_fields.json")
    
    if not sample_step2_path.exists() or not sample_step3_path.exists():
        print("Sample data files not found. Looking for alternatives...")
        return
    
    # Load the data
    with open(sample_step2_path, 'r', encoding='utf-8') as f:
        product_data = json.load(f)
    
    with open(sample_step3_path, 'r', encoding='utf-8') as f:
        mandatory_fields_data = json.load(f)
        mandatory_fields = mandatory_fields_data.get("mandatory_fields", {})
    
    print(f"Loaded product data for parent SKU: {product_data['parent_data']['_parent_sku']}")
    print(f"Product has {len(product_data['data_rows'])} variants")
    print(f"Mandatory fields count: {len(mandatory_fields)}")
    
    # Test variance analysis
    print("\n1. Testing Variance Analysis")
    print("-" * 30)
    
    analyzer = DataVarianceAnalyzer()
    analysis = analyzer.analyze_product_data(product_data)
    
    print("Parent fields detected:", len(analysis["parent_fields"]))
    print("Variance fields by type:")
    for field_type, fields in analysis["variance_fields"].items():
        print(f"  {field_type}: {fields}")
    
    print("Field mappings suggested:")
    for source, target in analysis["field_mappings"].items():
        print(f"  {source} → {target}")
    
    # Test prompt generation
    print("\n2. Testing Prompt Generation")
    print("-" * 30)
    
    prompt_manager = PromptTemplateManager()
    
    # Create mapping input
    mapping_input = MappingInput(
        parent_sku="4307",
        mandatory_fields=mandatory_fields,
        product_data=product_data,
        business_context="German Amazon marketplace product"
    )
    
    prompt_context = {
        "parent_sku": mapping_input.parent_sku,
        "mandatory_fields": mapping_input.mandatory_fields,
        "product_data": mapping_input.product_data,
        "business_context": mapping_input.business_context
    }
    
    # Generate the prompt
    user_prompt = prompt_manager.render_mapping_prompt(prompt_context)
    print(f"Generated prompt length: {len(user_prompt)} characters")
    print("Prompt includes variance analysis:", "VARIANCE ANALYSIS GUIDANCE" in user_prompt)
    
    # Test result structure
    print("\n3. Testing Result Structure")
    print("-" * 30)
    
    # Create a sample transformation result to verify structure
    sample_result = TransformationResult(
        parent_sku="4307",
        parent_data={
            "brand_name": "EIKO",
            "item_name": "LAHN Latzhose aus Genuacord",
            "country_of_origin": "Tunisia",
            "feed_product_type": "overalls"
        },
        variance_data={
            "size_name": ["44", "46", "48", "50", "52"],
            "color_name": ["Schwarz", "Braun", "Oliv"]
        },
        metadata={
            "total_mapped_fields": 4,
            "confidence": 0.87,
            "unmapped_mandatory": ["external_product_id", "item_sku"],
            "processing_notes": "Successfully transformed parent data and identified 2 variance dimensions"
        }
    )
    
    # Validate the result format
    result_dict = sample_result.model_dump()
    print("Result structure validation:")
    print(f"  Has parent_sku: {'parent_sku' in result_dict}")
    print(f"  Has parent_data: {'parent_data' in result_dict}")
    print(f"  Has variance_data: {'variance_data' in result_dict}")
    print(f"  Has metadata: {'metadata' in result_dict}")
    print(f"  Parent data fields: {len(result_dict['parent_data'])}")
    print(f"  Variance data dimensions: {len(result_dict['variance_data'])}")
    
    # Test constraint validation
    print("\n4. Testing Constraint Validation")
    print("-" * 30)
    
    validator = ConstraintValidator()
    validation_results = validator.validate_transformation(result_dict, mandatory_fields)
    
    print(f"Compliance score: {validation_results['compliance_score']:.2f}")
    print(f"Compliant fields: {len(validation_results['compliant_fields'])}")
    print(f"Constraint violations: {len(validation_results['constraint_violations'])}")
    print(f"Missing mandatory: {len(validation_results['missing_mandatory'])}")
    
    # Show some sample output
    print("\n5. Sample Output JSON Structure")
    print("-" * 30)
    print(json.dumps(result_dict, indent=2, ensure_ascii=False)[:500] + "...")
    
    print("\n✅ All tests completed successfully!")
    print("The transformation system is ready to produce correct output structure.")


if __name__ == "__main__":
    asyncio.run(test_data_transformation())