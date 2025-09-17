#!/usr/bin/env python3
"""Test script to validate safety filter fix for parent 4301.

This script tests the content sanitization implementation against
the specific safety filter triggers identified in parent 4301.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sku_analyzer.shared.content_sanitizer import ContentSanitizer
from sku_analyzer.shared.gemini_client import GeminiClient, AIProcessingConfig
from sku_analyzer.step5_mapping.ai_mapper import AIMapper
from sku_analyzer.step5_mapping.models import MappingInput


def load_parent_4301_data() -> Dict[str, Any]:
    """Load parent 4301 compressed data from production output."""
    
    parent_4301_path = project_root / "production_output" / "1756744213" / "parent_4301" / "step2_compressed.json"
    
    if not parent_4301_path.exists():
        raise FileNotFoundError(f"Parent 4301 data not found: {parent_4301_path}")
    
    with open(parent_4301_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def load_mandatory_fields() -> Dict[str, Any]:
    """Load mandatory fields from production output."""
    
    mandatory_fields_path = project_root / "production_output" / "1756744213" / "flat_file_analysis" / "step3_mandatory_fields.json"
    
    if not mandatory_fields_path.exists():
        raise FileNotFoundError(f"Mandatory fields not found: {mandatory_fields_path}")
    
    with open(mandatory_fields_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def test_content_sanitizer():
    """Test the content sanitizer on known trigger phrases."""
    
    print("=== Testing Content Sanitizer ===")
    
    # Test cases with original triggers from parent 4301
    test_cases = [
        "Diese Hose ist ein Dauerbrenner, weil sie einfach praktisch ist.",
        "Zumeist ältere Herren und Skater lieben die weiten Beine.",
        "Egal ob man es Manchester oder Cord nennt.",
        "Mit dieser Hose ist man zur Arbeit, in der Freizeit oder zum Kirchgang immer gut gekleidet."
    ]
    
    sanitizer = ContentSanitizer()
    
    print("Basic Sanitization:")
    for i, test_case in enumerate(test_cases, 1):
        sanitized = sanitizer.sanitize_text(test_case)
        triggers = sanitizer.scan_for_triggers(test_case)
        
        print(f"{i}. Original: {test_case}")
        print(f"   Sanitized: {sanitized}")
        print(f"   Triggers found: {triggers}")
        print()
    
    # Test aggressive mode
    print("Aggressive Sanitization:")
    aggressive_sanitizer = ContentSanitizer(aggressive_mode=True)
    
    combined_text = " ".join(test_cases)
    sanitized_aggressive = aggressive_sanitizer.sanitize_text(combined_text)
    
    print(f"Original combined: {combined_text}")
    print(f"Aggressive sanitized: {sanitized_aggressive}")
    print()


def analyze_parent_4301_content():
    """Analyze parent 4301 content for trigger words."""
    
    print("=== Analyzing Parent 4301 Content ===")
    
    try:
        data = load_parent_4301_data()
        sanitizer = ContentSanitizer()
        
        # Analyze risk score
        risk_score = sanitizer.assess_risk_score(data)
        print(f"Risk Score: {risk_score:.3f}")
        
        # Find all trigger words in the data
        all_triggers = set()
        
        def find_triggers_recursive(obj, path=""):
            nonlocal all_triggers
            
            if isinstance(obj, str):
                triggers = sanitizer.scan_for_triggers(obj)
                if triggers:
                    print(f"Triggers in {path}: {triggers}")
                    print(f"  Text: {obj[:100]}...")
                    all_triggers.update(triggers)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    find_triggers_recursive(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    find_triggers_recursive(item, f"{path}[{i}]")
        
        find_triggers_recursive(data)
        
        print(f"\nAll triggers found: {sorted(all_triggers)}")
        print(f"Total unique triggers: {len(all_triggers)}")
        
        # Test sanitization
        sanitized_data = sanitizer.sanitize_product_data(data)
        sanitized_risk = sanitizer.assess_risk_score(sanitized_data)
        
        print(f"Risk score after sanitization: {sanitized_risk:.3f}")
        print(f"Risk reduction: {((risk_score - sanitized_risk) / risk_score * 100):.1f}%")
        
        return data, sanitized_data
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None


async def test_ai_mapping_with_sanitization():
    """Test actual AI mapping with the sanitization fix."""
    
    print("=== Testing AI Mapping with Sanitization ===")
    
    # Load environment variables
    if not os.getenv('GOOGLE_API_KEY'):
        print("Loading environment from .env file...")
        from dotenv import load_dotenv
        load_dotenv()
        
        if not os.getenv('GOOGLE_API_KEY'):
            print("Error: GOOGLE_API_KEY not found in environment or .env file")
            return
    
    try:
        # Load data
        product_data = load_parent_4301_data()
        mandatory_fields = load_mandatory_fields()
        
        if not product_data or not mandatory_fields:
            print("Failed to load required data")
            return
        
        # Create mapping input
        mapping_input = MappingInput(
            parent_sku="4301",
            mandatory_fields=mandatory_fields,
            product_data=product_data,
            template_structure={}  # Will be generated internally
        )
        
        # Set up AI client
        config = AIProcessingConfig(
            enable_structured_output=True,
            model="gemini-2.0-flash-exp",
            temperature=0.3,
            thinking_budget=-1
        )
        
        ai_client = GeminiClient(config)
        ai_mapper = AIMapper(ai_client)
        
        # Test the AI mapping with progressive sanitization
        job_dir = project_root / "production_output" / "1756744213"
        
        print("Executing AI mapping with progressive sanitization...")
        result = await ai_mapper.execute_ai_mapping(mapping_input, job_dir)
        
        print(f"Mapping result:")
        print(f"  Parent SKU: {result.parent_sku}")
        print(f"  Total variants: {len(result.variant_data)}")
        print(f"  Confidence: {result.metadata.get('confidence', 'N/A')}")
        print(f"  Sanitization level: {result.metadata.get('sanitization_level', 'N/A')}")
        print(f"  Processing notes: {result.metadata.get('processing_notes', 'N/A')}")
        
        # Save successful result
        if len(result.variant_data) > 0:
            output_path = project_root / "production_output" / "1756744213" / "parent_4301" / "step5_ai_mapping_fixed.json"
            
            output_data = {
                "parent_data": result.parent_data,
                "variants": [
                    result.variant_data[key] for key in sorted(result.variant_data.keys())
                ],
                "metadata": result.metadata
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"Fixed result saved to: {output_path}")
            print(f"SUCCESS: Parent 4301 mapping fixed with {len(result.variant_data)} variants!")
        else:
            print("FAILED: No variants were mapped")
            
    except Exception as e:
        print(f"Error during AI mapping test: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    
    print("Parent 4301 Safety Filter Fix Validation")
    print("=" * 50)
    print()
    
    # Test 1: Content sanitizer functionality
    test_content_sanitizer()
    
    # Test 2: Analyze parent 4301 content
    analyze_parent_4301_content()
    
    # Test 3: Full AI mapping test
    await test_ai_mapping_with_sanitization()


if __name__ == "__main__":
    asyncio.run(main())