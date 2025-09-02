# AI Mapping Prompt Engineering Correction

## Problem Analysis

### Critical Issue Identified
The current AI prompts were requesting **field mappings** instead of **actual data transformation**, causing the AI to produce metadata about relationships rather than transformed Amazon-format data.

### Specific Problems:
1. **Wrong Output Request**: Prompts asked for `source_field->target_field` relationships
2. **Missing Target Structure**: No guidance on `parent_data` vs `variance_data` Amazon format
3. **Constraints Underutilized**: `valid_values` from mandatory fields not used as transformation constraints
4. **Field Identification vs Data Transformation**: AI produced mapping instructions, not transformed values

### Source Data Structure Analysis:
```json
// Current step2_compressed.json format:
{
  "parent_data": {
    "MANUFACTURER_NAME": "EIKO",
    "DESCRIPTION_LONG": "Diese Hose ist...",
    "COUNTRY_OF_ORIGIN": "Tunesien"
  },
  // Array of variant objects with specific values
  [
    {"FVALUE_3_2": 48, "FVALUE_3_3": "Schwarz", "SUPPLIER_PID": "4301_40_48"},
    {"FVALUE_3_2": 50, "FVALUE_3_3": "Schwarz", "SUPPLIER_PID": "4301_40_50"}
  ]
}
```

## Solution Implementation

### 1. Corrected Prompt Template
**File**: `/sku_analyzer/ai_mapping/prompts/files/product_transformation.jinja2`

**Key Changes**:
- Requests **DATA TRANSFORMATION** instead of field mappings
- Integrates `valid_values` constraints from mandatory fields
- Specifies `parent_data` vs `variance_data` structure requirements
- Includes German-to-English translation guidance
- Provides concrete transformation examples

**New Output Format**:
```json
{
  "parent_sku": "4301",
  "transformed_data": {
    "parent_data": {
      "brand_name": "EIKO",           // TRANSFORMED from MANUFACTURER_NAME
      "item_name": "EIKO ALLER Cordhose", // CONSTRUCTED from multiple fields
      "country_of_origin": "Tunisia"     // TRANSLATED from "Tunesien"
    },
    "variant_fields_identified": ["FVALUE_3_2", "FVALUE_3_3"]
  },
  "transformation_summary": {
    "parent_fields_transformed": 3,
    "variant_fields_identified": 2,
    "untransformed_fields": [],
    "overall_confidence": 0.87
  }
}
```

### 2. Updated System Prompt
**File**: `/sku_analyzer/ai_mapping/prompts/files/system_prompt.jinja2`

**Core Changes**:
- Emphasizes **VALUE TRANSFORMATION** over field identification
- Requires German-to-Amazon format conversion
- Mandates constraint validation against `valid_values`
- Specifies parent/child product data structure

### 3. New Data Models
**File**: `/sku_analyzer/ai_mapping/models.py`

**Added Models**:
```python
class FieldTransformation(BaseModel):
    """Individual field transformation result."""
    target_field: str
    source_field: str  
    transformed_value: Any
    confidence: float
    reasoning: str

class TransformedData(BaseModel):
    """Transformed product data structure."""
    parent_data: Dict[str, Any]
    variant_fields_identified: List[str]

class TransformationResult(BaseModel):
    """Complete AI transformation result."""
    parent_sku: str
    transformed_data: TransformedData
    transformation_summary: TransformationSummary
    transformation_details: List[FieldTransformation]
    processing_notes: str
```

### 4. Template Manager Updates
**File**: `/sku_analyzer/ai_mapping/prompts/templates.py`

**Changes**:
- `render_mapping_prompt()` now uses `product_transformation.jinja2`
- Added `render_legacy_mapping_prompt()` for backward compatibility
- Updated docstrings to emphasize data transformation focus

## Integration Impact

### Template System
- ✅ **product_transformation.jinja2**: New transformation-focused template
- ✅ **system_prompt.jinja2**: Updated to emphasize value transformation
- ✅ **templates.py**: Modified to use transformation template by default

### Data Models
- ✅ **TransformationResult**: New model for transformation output
- ✅ **FieldTransformation**: Individual transformation details
- ✅ **TransformedData**: Amazon-format data structure
- ✅ **Backward Compatibility**: Legacy models marked as deprecated

### Constraint Integration
- ✅ **valid_values**: Integrated as transformation constraints
- ✅ **max_length**: Applied during transformation validation
- ✅ **data_type**: Enforced in output structure

## Expected Transformation Examples

### Example 1: Brand Name Transformation
```
INPUT:  "MANUFACTURER_NAME": "EIKO"
OUTPUT: "brand_name": "EIKO"
LOGIC:  Direct mapping from manufacturer to brand
```

### Example 2: Product Name Construction
```  
INPUT:  "DESCRIPTION_LONG": "Diese Hose ist ein Dauerbrenner..."
        "MANUFACTURER_TYPE_DESCRIPTION": "ALLER"
        "MANUFACTURER_NAME": "EIKO"
OUTPUT: "item_name": "EIKO ALLER Cordhose"
LOGIC:  Construct from brand + type + product category
```

### Example 3: Country Translation
```
INPUT:  "COUNTRY_OF_ORIGIN": "Tunesien"  
OUTPUT: "country_of_origin": "Tunisia"
LOGIC:  German to English translation
```

### Example 4: Variant Field Identification
```
INPUT:  variance_data with FVALUE_3_2: [48, 50, 52] and FVALUE_3_3: ["Schwarz", "Oliv"]
OUTPUT: "variant_fields_identified": ["FVALUE_3_2", "FVALUE_3_3"]
LOGIC:  Fields that vary across product variants
```

## Validation & Testing

### Test Implementation
**File**: `test_transformation_correction.py`

**Demonstrates**:
- Comparison between old mapping vs new transformation approach
- Expected output format for Amazon marketplace data
- Integration of constraints and valid_values
- Parent vs variant data classification

### Key Validation Points
1. **Data Values**: AI produces actual transformed values, not field mappings
2. **Structure Compliance**: Output matches Amazon parent/child format requirements
3. **Constraint Validation**: Respects max_length and valid_values constraints
4. **Translation Accuracy**: German-to-English conversions are contextually correct
5. **Confidence Scoring**: Transformation decisions include confidence metrics

## Rollout Strategy

### Phase 1: Template Integration (COMPLETED)
- ✅ New transformation template created
- ✅ System prompt updated for transformation focus
- ✅ Data models extended with transformation structures

### Phase 2: Processor Updates (PENDING)
- Update `AIMappingProcessor` to use `TransformationResult` models
- Modify output validation for transformation format
- Update integration points to handle new data structure

### Phase 3: Testing & Validation (PENDING)
- Run end-to-end tests with new transformation prompts
- Validate Amazon format compliance
- Performance testing with real Gemini API

### Phase 4: Production Rollout (PENDING)
- Deploy corrected prompts to production
- Monitor transformation quality and confidence scores
- Compare mapping success rates: old vs new approach

## Files Modified

### Core Implementation
- `/sku_analyzer/ai_mapping/prompts/files/product_transformation.jinja2` (CREATED)
- `/sku_analyzer/ai_mapping/prompts/files/system_prompt.jinja2` (MODIFIED)
- `/sku_analyzer/ai_mapping/prompts/templates.py` (MODIFIED)
- `/sku_analyzer/ai_mapping/models.py` (EXTENDED)

### Testing & Documentation  
- `test_transformation_correction.py` (CREATED)
- `PROMPT_ENGINEERING_CORRECTION_SUMMARY.md` (CREATED)

## Success Criteria

### Technical Validation
1. **Output Format**: AI returns transformed Amazon-format data structure
2. **Constraint Compliance**: Transformed values respect valid_values and max_length
3. **Translation Quality**: German fields accurately translated to English Amazon format
4. **Structure Classification**: Parent vs variant data correctly identified

### Business Impact
1. **Mapping Success Rate**: Improved successful transformation percentage
2. **Data Quality**: Higher confidence scores for transformed values
3. **Pipeline Efficiency**: Reduced manual intervention requirements
4. **Amazon Compliance**: Better adherence to marketplace data requirements

## Next Steps

1. **Update Processing Components**: Modify `AIMappingProcessor` to use new models
2. **Integration Testing**: Run full pipeline tests with transformation prompts
3. **Performance Validation**: Compare transformation quality vs old mapping approach
4. **Production Deployment**: Roll out corrected prompts with monitoring

---

**Implementation Status**: CORE TEMPLATE CORRECTION COMPLETED
**Next Phase**: Processor integration and end-to-end testing
**Impact**: Fundamental improvement from field mapping to actual data transformation