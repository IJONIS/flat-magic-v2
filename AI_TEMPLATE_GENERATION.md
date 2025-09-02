# AI-Powered Template Generation

This document describes the new AI-powered field categorization feature for template generation in the SKU Analyzer pipeline.

## Overview

The template generator now supports intelligent field categorization using Google's Gemini AI model, replacing the deterministic keyword-matching approach with business-logic-driven AI analysis.

## Key Features

- **AI-Powered Categorization**: Uses Gemini-2.5-flash to analyze field characteristics and categorize them intelligently
- **Fallback Support**: Automatically falls back to deterministic approach if AI fails
- **Transparent Logging**: Logs AI reasoning and confidence scores for debugging
- **Performance Monitoring**: Integrates with existing performance tracking
- **Flexible Configuration**: Can be enabled/disabled via initialization parameters

## Architecture Changes

### Files Modified

1. **`sku_analyzer/flat_file/template_generator.py`**:
   - Added AI categorization methods
   - Enhanced initialization with AI configuration
   - Added fallback mechanisms
   - Enhanced metadata tracking

2. **`sku_analyzer/ai_mapping/gemini_client.py`**:
   - Fixed missing `List` import for type annotations

### New Components

- `AICategorization` exception class for AI-specific errors
- `_ai_categorize_fields()` - Main AI categorization method
- `_create_categorization_prompt()` - Business-logic-aware prompt creation
- `_validate_ai_categorization()` - AI response validation and cleanup

## Usage

### Basic Usage

```python
from sku_analyzer.flat_file.template_generator import TemplateGenerator

# Enable AI categorization (default)
generator = TemplateGenerator(enable_ai_categorization=True)

# Disable AI categorization (use deterministic approach)
generator = TemplateGenerator(enable_ai_categorization=False)

# Generate template with AI analysis
result = await generator.generate_template_from_mandatory_fields(
    step3_mandatory_path="path/to/step3_mandatory_fields.json",
    output_path="path/to/step4_template.json"
)
```

### Configuration

The AI integration uses the following configuration:

```python
ai_config = AIProcessingConfig(
    model_name="gemini-2.5-flash",
    temperature=0.1,           # Low temperature for consistent results
    max_tokens=4096,
    timeout_seconds=30,
    max_concurrent=1           # Sequential processing for categorization
)
```

### Environment Setup

1. **Install Dependencies**:
   ```bash
   pip install google-generativeai pydantic
   ```

2. **Set API Key**:
   ```bash
   export GOOGLE_API_KEY="your_gemini_api_key_here"
   ```

3. **Test Integration**:
   ```bash
   python test_ai_template_generator.py
   ```

## AI Categorization Logic

### Business Context

The AI understands German Amazon marketplace product structures with parent-child hierarchies:

- **Parent Fields**: Define product families (brand, category, material, demographics)
- **Variant Fields**: Define individual SKU variations (size, color, identifiers)

### Categorization Rules

**Parent-Level Fields**:
- Brand information (brand_name, manufacturer)
- Product classification (feed_product_type, department)
- Material properties (outer_material_type, fabric_type)
- Demographics (target_gender, age_range_description)
- System classifications (size_system, country_of_origin)
- Shared attributes (standard_price, main_image_url)
- Fields with limited valid values (≤ 10 options)

**Variant-Level Fields**:
- Unique identifiers (item_sku, external_product_id)
- Size variations (size_name, size_map)
- Color variations (color_name, color_map)
- SKU-specific names (item_name)
- Fields with many possible values or no constraints

### Critical Field Rules

The AI enforces business-critical field placement:

- `feed_product_type` → Always parent (product category)
- `brand_name` → Always parent (brand family)
- `target_gender` → Always parent (demographic)
- `department_name` → Always parent (business category)
- `item_sku` → Always variant (unique identifier)
- `color_name`, `color_map` → Always variant (per-SKU variation)
- `size_name`, `size_map` → Always variant (per-SKU variation)
- `external_product_id` → Always variant (unique barcode)

## Error Handling

### AI Failure Scenarios

1. **API Key Missing**: Falls back to deterministic approach
2. **Network Issues**: Retries with timeout, then falls back
3. **Invalid Response**: Validates and cleans AI output, adds missing fields
4. **Rate Limiting**: Built-in rate limiting and retry logic

### Validation & Cleanup

- **Completeness Check**: Ensures all fields are categorized
- **Duplication Removal**: Resolves conflicts (keeps in parent level)
- **Unknown Field Handling**: Removes fields not in input data
- **Missing Field Recovery**: Adds missed fields to variant level

## Output Enhancements

### New Metadata Fields

The template output now includes:

```json
{
  "metadata": {
    "categorization_method": "ai|deterministic",
    "ai_confidence": 0.95,
    "generation_timestamp": "2024-01-17T16:45:00Z",
    "quality_score": 0.89
  }
}
```

### Logging Enhancements

- AI categorization timing and performance
- AI reasoning and confidence scores (debug level)
- Fallback triggers and reasons
- Field placement decisions and corrections

## Performance Considerations

### AI Request Optimization

- **Single Request**: All fields categorized in one API call
- **Low Temperature**: Consistent results (0.1)
- **Rate Limiting**: 100ms minimum between requests
- **Async Processing**: Non-blocking AI calls
- **Performance Tracking**: Full monitoring integration

### Expected Performance

- **AI Categorization**: ~500-2000ms (depending on field count)
- **Fallback Speed**: <1ms (deterministic approach)
- **Memory Usage**: Minimal (lazy client initialization)

## Testing

### Test Suite

```bash
# Run comprehensive tests
python test_ai_template_generator.py

# Test with actual API key
python test_ai_with_key.py
```

### Test Coverage

- Deterministic categorization (baseline)
- AI categorization with validation
- Full template generation workflow
- Error handling and fallback mechanisms
- Output format validation

## Integration Points

### Pipeline Integration

The AI template generator integrates seamlessly with the existing pipeline:

1. **Step 3 Output**: Reads `step3_mandatory_fields.json`
2. **AI Analysis**: Categorizes fields using business logic
3. **Template Creation**: Generates structured parent-child template
4. **Step 4 Output**: Saves `step4_template.json` with enhanced metadata

### Backward Compatibility

- **API**: No breaking changes to existing interfaces
- **Output Format**: Enhanced metadata, same core structure
- **Configuration**: AI can be disabled for deterministic behavior
- **Dependencies**: New dependencies are optional (graceful degradation)

## Troubleshooting

### Common Issues

1. **"GOOGLE_API_KEY not found"**:
   - Set environment variable or falls back to deterministic approach
   - No pipeline disruption

2. **"AI categorization failed"**:
   - Check network connectivity and API key validity
   - Pipeline continues with deterministic approach

3. **"Invalid JSON response from Gemini"**:
   - AI response parsing failed, likely model output issue
   - Falls back automatically with warning logged

4. **Field categorization inconsistencies**:
   - Check debug logs for AI reasoning
   - Critical field rules override AI decisions
   - Validation cleans up any issues

### Debug Information

Enable debug logging to see AI decision-making:

```python
import logging
logging.getLogger('sku_analyzer.flat_file.template_generator').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Improvements

1. **Model Fine-Tuning**: Domain-specific training for better accuracy
2. **Batch Processing**: Multiple field sets in single request
3. **Caching**: Cache categorization decisions for similar field sets
4. **A/B Testing**: Compare AI vs deterministic approaches
5. **Feedback Loop**: Learn from manual corrections

### Configuration Options

Future versions may include:
- Model selection (different Gemini variants)
- Temperature tuning for creativity vs consistency
- Custom categorization rules per domain
- Multi-language support for international markets

---

## Quick Start Example

```python
#!/usr/bin/env python3
import asyncio
import os
from pathlib import Path
from sku_analyzer.flat_file.template_generator import TemplateGenerator

async def main():
    # Set up AI (optional - will fallback if not available)
    os.environ["GOOGLE_API_KEY"] = "your_api_key_here"
    
    # Create generator with AI enabled
    generator = TemplateGenerator(enable_ai_categorization=True)
    
    # Generate template
    result = await generator.generate_template_from_mandatory_fields(
        step3_mandatory_path="data/step3_mandatory_fields.json",
        output_path="output/step4_template.json"
    )
    
    # Check if AI was used
    method = result['metadata'].get('categorization_method', 'unknown')
    print(f"Used {method} categorization")
    
    if method == 'ai':
        confidence = result['metadata'].get('ai_confidence', 0.0)
        print(f"AI confidence: {confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

This implementation provides a robust, intelligent field categorization system that enhances template generation while maintaining full backward compatibility and reliability through comprehensive fallback mechanisms.