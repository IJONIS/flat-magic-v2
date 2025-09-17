#!/usr/bin/env python3
"""Test validation of AI mapping fix for accurate mandatory field reporting.

This script validates the critical fix for the AI mapping validation system that was 
reporting false completion while missing 48% of mandatory fields.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Set


def load_template_mandatory_fields(template_path: Path) -> Dict[str, str]:
    """Load ALL mandatory fields from step4_template.json."""
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
        
    with open(template_path, 'r', encoding='utf-8') as f:
        template_data = json.load(f)
    
    mandatory_fields = {}
    template_structure = template_data.get('template_structure', {})
    
    # Extract parent mandatory fields
    parent_product = template_structure.get('parent_product', {})
    parent_fields = parent_product.get('fields', {})
    required_parent_fields = parent_product.get('required_fields', [])
    
    for field_name in required_parent_fields:
        if field_name in parent_fields:
            mandatory_fields[field_name] = 'parent'
    
    # Extract variant mandatory fields  
    child_variants = template_structure.get('child_variants', {})
    child_fields = child_variants.get('fields', {})
    
    for field_name, field_info in child_fields.items():
        if isinstance(field_info, dict):
            validation_rules = field_info.get('validation_rules', {})
            if isinstance(validation_rules, dict) and validation_rules.get('required', False):
                mandatory_fields[field_name] = 'variant'
    
    print(f"✅ Loaded {len(mandatory_fields)} mandatory fields from template:")
    parent_count = sum(1 for v in mandatory_fields.values() if v == 'parent')
    variant_count = sum(1 for v in mandatory_fields.values() if v == 'variant')
    print(f"   - Parent fields: {parent_count}")
    print(f"   - Variant fields: {variant_count}")
    print(f"   - Fields: {list(mandatory_fields.keys())}")
    
    return mandatory_fields


def extract_mapped_fields_from_result(result_data: Dict[str, Any]) -> Set[str]:
    """Extract all mapped fields from AI mapping result."""
    mapped_fields = set()
    
    # Parent fields
    parent_data = result_data.get('parent_data', {})
    if isinstance(parent_data, dict):
        mapped_fields.update(parent_data.keys())
    
    # Variant fields from both structures
    variants = result_data.get('variants', [])
    if isinstance(variants, list):
        for variant in variants:
            if isinstance(variant, dict):
                # Handle both direct fields and nested structures
                if 'data' in variant:
                    variant_fields = variant['data']
                else:
                    # Check for variant_N keys
                    for key, value in variant.items():
                        if key.startswith('variant_') and isinstance(value, dict):
                            variant_fields = value
                            break
                    else:
                        variant_fields = variant
                        
                if isinstance(variant_fields, dict):
                    mapped_fields.update(variant_fields.keys())
    
    # Also check variance_data format
    variance_data = result_data.get('variance_data', {})
    if isinstance(variance_data, dict):
        for variant_data in variance_data.values():
            if isinstance(variant_data, dict):
                mapped_fields.update(variant_data.keys())
    
    return mapped_fields


def validate_ai_mapping_result(result_path: Path, template_path: Path) -> Dict[str, Any]:
    """Validate AI mapping result against complete template requirements."""
    print(f"\n🔍 Validating AI mapping result: {result_path}")
    
    if not result_path.exists():
        return {
            'valid': False,
            'error': f'Result file not found: {result_path}'
        }
    
    # Load data
    mandatory_fields = load_template_mandatory_fields(template_path)
    
    with open(result_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
    
    # Extract mapped fields
    mapped_fields = extract_mapped_fields_from_result(result_data)
    
    print(f"📊 Found {len(mapped_fields)} mapped fields in result")
    print(f"   Mapped fields: {sorted(mapped_fields)}")
    
    # Calculate coverage
    unmapped_mandatory = []
    mapped_mandatory = []
    
    for field_name, field_type in mandatory_fields.items():
        if field_name in mapped_fields:
            mapped_mandatory.append(field_name)
        else:
            unmapped_mandatory.append(field_name)
    
    coverage = len(mapped_mandatory) / len(mandatory_fields) if mandatory_fields else 1.0
    
    # Validate variant count
    total_variants_in_result = 0
    variants = result_data.get('variants', [])
    variance_data = result_data.get('variance_data', {})
    
    if isinstance(variants, list):
        total_variants_in_result = len(variants)
    elif isinstance(variance_data, dict):
        total_variants_in_result = len(variance_data)
    
    # Get metadata reporting
    metadata = result_data.get('metadata', {})
    reported_unmapped = metadata.get('unmapped_mandatory_fields', [])
    reported_total_variants = metadata.get('total_variants', 0)
    
    # Validation results
    validation_results = {
        'valid': len(unmapped_mandatory) == 0,
        'mandatory_field_coverage': coverage,
        'total_mandatory_fields': len(mandatory_fields),
        'mapped_mandatory_fields': len(mapped_mandatory),
        'unmapped_mandatory_fields': len(unmapped_mandatory),
        'unmapped_field_names': unmapped_mandatory,
        'mapped_field_names': mapped_mandatory,
        'total_variants_processed': total_variants_in_result,
        'metadata_reports_accurate': len(reported_unmapped) == len(unmapped_mandatory),
        'reported_unmapped_count': len(reported_unmapped),
        'actual_unmapped_count': len(unmapped_mandatory),
        'reporting_accuracy': 'ACCURATE' if len(reported_unmapped) == len(unmapped_mandatory) else 'FALSE POSITIVE' if len(reported_unmapped) < len(unmapped_mandatory) else 'OVER_REPORTING'
    }
    
    return validation_results


def print_validation_summary(results: Dict[str, Any]):
    """Print formatted validation summary."""
    print(f"\n{'='*60}")
    print("🎯 VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    coverage_pct = results['mandatory_field_coverage'] * 100
    print(f"📈 Mandatory Field Coverage: {coverage_pct:.1f}% ({results['mapped_mandatory_fields']}/{results['total_mandatory_fields']})")
    
    if results['unmapped_mandatory_fields'] > 0:
        print(f"❌ Missing {results['unmapped_mandatory_fields']} mandatory fields:")
        for field in results['unmapped_field_names'][:5]:  # Show first 5
            print(f"   - {field}")
        if len(results['unmapped_field_names']) > 5:
            print(f"   ... and {len(results['unmapped_field_names']) - 5} more")
    else:
        print("✅ All mandatory fields mapped successfully!")
    
    print(f"\n🔢 Variant Processing: {results['total_variants_processed']} variants")
    
    print(f"\n📊 Reporting Accuracy: {results['reporting_accuracy']}")
    print(f"   - Reported unmapped: {results['reported_unmapped_count']}")
    print(f"   - Actually unmapped: {results['actual_unmapped_count']}")
    
    if results['metadata_reports_accurate']:
        print("✅ Metadata reporting is ACCURATE")
    else:
        print("❌ Metadata reporting is INACCURATE - FALSE COMPLETION SIGNAL")
        
    print(f"\n🎯 Overall Status: {'✅ VALID' if results['valid'] else '❌ INVALID'}")


def main():
    """Main validation routine."""
    print("🧪 Testing AI Mapping Validation Fix")
    print("="*50)
    
    # Test data paths
    base_dir = Path("/Users/jaminmahmood/Desktop/Flat Magic v6/production_output")
    
    # Find latest job with AI mapping data
    job_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    test_cases_found = []
    
    for job_dir in job_dirs[-5:]:  # Check last 5 jobs
        template_path = job_dir / "flat_file_analysis" / "step4_template.json"
        
        if not template_path.exists():
            continue
            
        # Find parents with AI mapping results
        for parent_dir in job_dir.iterdir():
            if parent_dir.is_dir() and parent_dir.name.startswith('parent_'):
                result_path = parent_dir / "step5_ai_mapping.json"
                if result_path.exists():
                    parent_sku = parent_dir.name.replace('parent_', '')
                    test_cases_found.append({
                        'job_id': job_dir.name,
                        'parent_sku': parent_sku,
                        'result_path': result_path,
                        'template_path': template_path
                    })
    
    if not test_cases_found:
        print("❌ No AI mapping test cases found!")
        print("   Please run AI mapping first to generate step5_ai_mapping.json files")
        return
    
    print(f"📁 Found {len(test_cases_found)} test cases to validate")
    
    all_results = []
    
    for test_case in test_cases_found:
        print(f"\n🔍 Testing Job {test_case['job_id']} - Parent {test_case['parent_sku']}")
        
        validation_result = validate_ai_mapping_result(
            test_case['result_path'], 
            test_case['template_path']
        )
        
        validation_result.update({
            'job_id': test_case['job_id'],
            'parent_sku': test_case['parent_sku']
        })
        
        all_results.append(validation_result)
        print_validation_summary(validation_result)
    
    # Overall summary
    print(f"\n{'='*70}")
    print("📊 OVERALL VALIDATION RESULTS")
    print(f"{'='*70}")
    
    valid_count = sum(1 for r in all_results if r['valid'])
    accurate_reporting_count = sum(1 for r in all_results if r['metadata_reports_accurate'])
    
    avg_coverage = sum(r['mandatory_field_coverage'] for r in all_results) / len(all_results)
    
    print(f"✅ Valid results: {valid_count}/{len(all_results)} ({valid_count/len(all_results)*100:.1f}%)")
    print(f"📊 Accurate reporting: {accurate_reporting_count}/{len(all_results)} ({accurate_reporting_count/len(all_results)*100:.1f}%)")
    print(f"📈 Average coverage: {avg_coverage*100:.1f}%")
    
    # Check if fix is working
    if accurate_reporting_count == len(all_results):
        print("\n🎉 SUCCESS: All validation reports are accurate - FALSE COMPLETION FIX WORKING!")
    else:
        print(f"\n⚠️  WARNING: {len(all_results) - accurate_reporting_count} cases still have inaccurate reporting")
    
    if avg_coverage > 0.8:
        print(f"✅ Good field coverage achieved ({avg_coverage*100:.1f}%)")
    else:
        print(f"❌ Low field coverage ({avg_coverage*100:.1f}%) - needs improvement")


if __name__ == "__main__":
    main()