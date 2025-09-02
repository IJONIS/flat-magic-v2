# Template-Driven AI Mapping Workflow

## Overview

The SKU analyzer pipeline now includes a new **Step 4: Template Generator** that creates reusable parent-child structure templates from mandatory fields analysis. This enables more efficient and structured AI mapping in Step 5.

## Pipeline Architecture

### Step 1-2: Core Analysis
- **Step 1**: SKU pattern analysis and hierarchy extraction
- **Step 2**: CSV export and data compression

### Step 1-3: Flat File Analysis (Template Mode)
- **Step 1 (Template)**: Column analysis from XLSM template
- **Step 2 (Template)**: Valid values extraction
- **Step 3**: Mandatory fields processing

### Step 4: Template Generation (NEW)
- **Input**: `step3_mandatory_fields.json`
- **Output**: `step4_template.json`
- **Function**: Creates structured parent-child templates

### Step 5: AI Mapping (REFACTORED)
- **Input**: `step4_template.json` + `step2_compressed.json`
- **Output**: `step5_ai_mapping.json`
- **Function**: Template-driven AI product mapping

## Key Features

### Template Generator (`TemplateGenerator`)

**Location**: `sku_analyzer/flat_file/template_generator.py`

**Core Methods**:
- `generate_template_from_mandatory_fields()` - Main template generation logic
- `categorize_field_levels()` - Determines parent vs variant field placement
- `create_template_structure()` - Builds reusable template JSON structure
- `validate_template()` - Ensures template quality and completeness

**Field Categorization Logic**:
- **Parent Fields**: Brand, category, product type, classification fields
- **Variant Fields**: Size, color, material, style, specification fields
- **Smart Analysis**: Uses field characteristics and naming patterns

### AI Mapping Integration

**Updated Methods**:
- `AIMappingProcessor.process_parent_directory()` - Now uses templates
- `MappingInput` model - Extended with `template_structure` field
- Output files renamed from `step3_ai_mapping.json` to `step5_ai_mapping.json`

## Usage

### Basic Template Generation
```python
from sku_analyzer.flat_file.template_generator import TemplateGenerator

generator = TemplateGenerator()
template_result = await generator.generate_template_from_mandatory_fields(
    "step3_mandatory_fields.json",
    "step4_template.json"
)
```

### Complete Workflow
```python
from sku_analyzer.core.analyzer import SkuPatternAnalyzer

analyzer = SkuPatternAnalyzer()

# Full workflow with template generation
job_id = await analyzer.process_file_with_template(
    input_path="data.xlsx",
    template_path="template.xlsm"
)

# Add AI mapping to existing job
ai_results = await analyzer.add_ai_mapping_to_job(job_id)
```

## Template Structure

### Generated Template Format
```json
{
  "metadata": {
    "generation_timestamp": "2025-09-02T15:08:10.310356",
    "template_version": "1.0",
    "field_distribution": {
      "parent_fields": 15,
      "variant_fields": 8,
      "parent_ratio": 0.65
    },
    "quality_score": 1.0
  },
  "template_structure": {
    "parent_product": {
      "fields": { /* parent-level field definitions */ },
      "required_fields": ["feed_product_type", "brand_name"],
      "field_count": 15
    },
    "child_variants": {
      "fields": { /* variant-level field definitions */ },
      "variable_fields": ["color_name", "size_name"],
      "inherited_fields": ["material_type"],
      "field_count": 8
    },
    "field_relationships": {
      "parent_defines": [ /* inheritance rules */ ],
      "variant_overrides": [ /* override permissions */ ],
      "shared_constraints": { /* cross-level constraints */ }
    }
  }
}
```

## Validation

### Pipeline Validation
The `completion_check()` method now validates:
- Step 4: `step4_template.json` (when flat file analysis enabled)
- Step 5: `step5_ai_mapping.json` (when AI mapping validation enabled)

### Quality Metrics
- **Quality Score**: 0.0-1.0 based on field distribution and relationships
- **Field Balance**: Optimal 30-70% parent field ratio
- **Relationship Complexity**: Inheritance and override rule coverage

## Benefits

1. **Structured Mapping**: Clear parent-child field categorization
2. **Reusable Templates**: Generated templates can guide multiple AI mappings
3. **Quality Validation**: Built-in template quality assessment
4. **Backward Compatibility**: Existing workflows continue to function
5. **Flexible Integration**: Optional step that enhances but doesn't require changes

## File Locations

### Implementation Files
- `/sku_analyzer/flat_file/template_generator.py` - NEW: Template generator
- `/sku_analyzer/core/analyzer.py` - UPDATED: Integration methods
- `/sku_analyzer/ai_mapping/processor.py` - UPDATED: Template-driven processing
- `/sku_analyzer/ai_mapping/models.py` - UPDATED: Template structure support

### Output Files
- `production_output/{job_id}/flat_file_analysis/step4_template.json` - NEW
- `production_output/{job_id}/parent_{sku}/step5_ai_mapping.json` - RENAMED

## Migration Notes

- AI mapping step moved from step 3 to step 5
- Old `step3_ai_mapping.json` files renamed to `step5_ai_mapping.json`
- Template generation is optional - pipeline continues without it
- Existing jobs can have template generation added retroactively