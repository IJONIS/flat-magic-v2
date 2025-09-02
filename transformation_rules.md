# AI Transformation Rules and Format Enforcement

## Overview

This document defines the exact transformation rules and format requirements for AI-generated product data transformations, enforcing compliance with `example_output_ai.json` structure.

## Critical Issues Addressed

### Previous Problems
1. **Empty parent_data**: AI was producing `"parent_data": {}` instead of actual shared values
2. **Wrong variance_data format**: AI was producing arrays of unique values instead of individual SKU records
3. **Missing individual SKU records**: No representation of actual variant combinations
4. **Incorrect JSON structure**: Not following the exact format from example file

### Solutions Implemented
1. **Reference-based prompting**: Direct reference to `example_output_ai.json` as canonical format
2. **Format validation checklist**: Step-by-step verification requirements
3. **Clear transformation rules**: Explicit mapping instructions from source to target fields
4. **Concrete examples**: Show correct vs incorrect output formats with actual data

## Transformation Rules

### 1. Metadata Section
```json
{
  "metadata": {
    "parent_id": "extracted_from_source_parent_sku",
    "job_id": "job_YYYYMMDD_HHMMSS_format", 
    "transformation_timestamp": "ISO_8601_current_timestamp",
    "ai_model": "gemini-2.5-flash",
    "mapping_confidence": 0.0_to_1.0_confidence_score,
    "total_variants": count_of_variance_data_array_items
  }
}
```

### 2. Parent Data Section - Shared Attributes Only
```json
{
  "parent_data": {
    "brand_name": "ACTUAL_BRAND_FROM_SOURCE",
    "feed_product_type": "MAPPED_PRODUCT_TYPE",
    "outer_material_type": "ACTUAL_MATERIAL",
    "target_gender": "MAPPED_GENDER",
    "country_of_origin": "MAPPED_COUNTRY"
  }
}
```

**Requirements:**
- Must contain actual values from source data
- Only include fields that are IDENTICAL across all variants
- Apply German-to-English translations where needed
- Map to Amazon marketplace field names
- Never use placeholders or empty objects

### 3. Variance Data Section - Individual SKU Records
```json
{
  "variance_data": [
    {
      "item_sku": "unique_identifier_per_variant",
      "size_name": "size_value_for_this_sku",
      "color_name": "color_value_for_this_sku",
      "size_map": "standardized_size_value",
      "color_map": "standardized_color_value"
    }
  ]
}
```

**Requirements:**
- Must be an array of individual SKU objects
- Each object represents ONE specific variant
- Never use arrays of unique values
- Each item_sku must be unique
- Represent actual variant combinations from source data

## Field Mapping Examples

### Source to Target Field Mappings
| Source Field | Target Field | Example Transformation |
|--------------|--------------|----------------------|
| MANUFACTURER_NAME | brand_name | "EIKO" → "EIKO" |
| PRODUCT_TYPE | feed_product_type | "Hose" → "pants" |
| MATERIAL | outer_material_type | "Cord" → "Cord" |
| GENDER | target_gender | "Herren" → "Männlich" |
| COUNTRY_OF_ORIGIN | country_of_origin | "Tunesien" → "Tunesien" |
| SIZE | size_name | "44" → "44" |
| COLOR | color_name | "Schwarz" → "Schwarz" |

### German-to-English Translations
| German | English |
|--------|---------|
| Hose | pants |
| Herren | Männlich (keep German for target market) |
| Erwachsener | Adult |
| Tunesien | Tunisia (if needed for English markets) |

## Format Validation Checklist

### Before Submission Verification
✅ **Structure Check:**
- [ ] Exactly 3 top-level keys: metadata, parent_data, variance_data
- [ ] Valid JSON format
- [ ] Correct data types for all fields

✅ **Metadata Validation:**
- [ ] 6 required fields present
- [ ] mapping_confidence is number 0.0-1.0
- [ ] total_variants matches variance_data length

✅ **Parent Data Validation:**
- [ ] Contains actual values (not empty object)
- [ ] Only shared attributes across all variants
- [ ] Proper field name mapping
- [ ] No placeholder values

✅ **Variance Data Validation:**
- [ ] Array format (not object)
- [ ] Individual SKU records (not arrays of values)
- [ ] Each record has 5 required fields
- [ ] Unique item_sku values
- [ ] Actual variant combinations

## Common Mistakes to Avoid

### ❌ Incorrect Formats
```json
// WRONG - Empty parent_data
"parent_data": {}

// WRONG - Arrays of unique values
"variance_data": {
  "size_name": ["44", "46", "48"],
  "color_name": ["Schwarz", "Braun"]
}

// WRONG - Placeholder values
"parent_data": {
  "brand_name": "TRANSFORMED_BRAND_VALUE"
}
```

### ✅ Correct Formats
```json
// CORRECT - Actual shared values
"parent_data": {
  "brand_name": "EIKO",
  "feed_product_type": "pants"
}

// CORRECT - Individual SKU records
"variance_data": [
  {
    "item_sku": "4307_40_44",
    "size_name": "44",
    "color_name": "Schwarz",
    "size_map": "44",
    "color_map": "Schwarz"
  }
]
```

## Implementation Notes

### Template Usage
1. Use `product_transformation.jinja2` for main transformation prompt
2. Use `format_validation.jinja2` for output verification
3. Reference `example_output_ai.json` as canonical structure
4. Apply transformation rules consistently

### Quality Assurance
1. Always validate output against example format
2. Check that parent_data contains actual values
3. Verify variance_data as array of SKU objects
4. Ensure metadata fields are complete and correct
5. Confirm all transformations are applied properly

## Testing and Validation

### Test Cases
1. **Empty parent_data detection**: Fail if parent_data is {}
2. **Wrong variance_data format**: Fail if not array of SKU objects  
3. **Missing metadata**: Fail if required fields absent
4. **Placeholder values**: Fail if placeholder text present
5. **Format compliance**: Pass only if exact structure match

### Validation Process
1. Parse JSON for structural validity
2. Check field presence and data types
3. Verify parent_data has actual values
4. Validate variance_data array structure
5. Confirm metadata completeness
6. Compare against example_output_ai.json structure

This comprehensive approach ensures AI outputs match the exact format required for successful processing.