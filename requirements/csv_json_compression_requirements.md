# CSV-to-JSON Optimization Requirements

## Overview

Transform parent-group CSV files into compressed JSON format by separating parent-level (common) data from child-level (varying) data, eliminating blank columns, and preserving data integrity while achieving significant size reduction.

## Data Analysis Results

Based on analysis of sample data (164 columns, 81 rows in parent_4307):
- **Parent-level columns**: 49 (identical across all children)
- **Child-level columns**: 7 (varying values between children)
- **Empty columns**: 108 (all null/empty values)
- **Compression potential**: ~66% size reduction from blank column removal alone

## Function List

### Core Functions

1. **analyze_csv_structure(csv_path)** - Analyze column patterns and categorize as parent/child/empty
2. **detect_blank_columns(csv_data)** - Identify columns with all null/empty values
3. **separate_parent_child_data(csv_data)** - Split data into parent-level vs child-level attributes  
4. **compress_to_json(parent_data, children_data)** - Generate optimized JSON structure
5. **validate_compression(original_csv, compressed_json)** - Verify data integrity and compression ratio

## Acceptance Criteria

### AC-F1: Blank Column Detection
**Given** a parent group CSV with mixed populated and empty columns  
**When** analyzing column structure  
**Then** remove all columns where ≥95% of values are null/empty/whitespace  

### AC-F2: Parent Data Extraction  
**Given** a parent group with multiple children  
**When** analyzing column values across all rows  
**Then** extract columns with identical values across ALL children as parent-level data

### AC-F3: Child Data Preservation
**Given** columns that vary between children  
**When** compressing data structure  
**Then** preserve only varying attributes per child with original data types

### AC-F4: JSON Structure Generation
**Given** separated parent and child data  
**When** generating compressed JSON  
**Then** output structure: `{parent_data: {}, children_data: [{id, varying_attrs}]}`

### AC-E1: Data Type Preservation
**Given** mixed data types (strings, numbers, booleans, nulls)  
**When** compressing to JSON  
**Then** preserve original types without string coercion

### AC-E2: Empty Value Handling  
**Given** null, empty string, or whitespace-only values  
**When** processing data  
**Then** normalize to null in JSON and exclude from blank column detection if <5% of values

### AC-E3: Unicode and Special Characters
**Given** text with unicode, quotes, or special characters  
**When** converting to JSON  
**Then** preserve original encoding and escape properly

### AC-D1: Idempotent Compression
**Given** the same parent group CSV  
**When** compressed multiple times  
**Then** produce identical JSON output (deterministic)

### AC-D2: Lossless Conversion  
**Given** original CSV data  
**When** compressed and reconstructed  
**Then** all non-empty original values must be recoverable

### AC-N1: Compression Ratio
**Given** a 126-row parent group (largest test case)  
**When** compressing CSV to JSON  
**Then** achieve ≥50% size reduction while maintaining readability

### AC-N2: Processing Performance
**Given** parent groups of varying sizes (26-126 rows)  
**When** processing compression  
**Then** complete in ≤2 seconds per parent group

### AC-N3: Memory Efficiency
**Given** large CSV files  
**When** processing compression  
**Then** avoid loading entire dataset into memory simultaneously

## Detailed Specifications

### 1. Blank Column Detection Algorithm

```python
def detect_blank_columns(csv_data: pd.DataFrame) -> List[str]:
    """
    Criteria for blank columns:
    1. ≥95% of values are null, empty string, or whitespace-only
    2. Exclude columns with meaningful structure (e.g., all zeros vs all empty)
    3. Handle edge case: single non-empty value in otherwise empty column
    """
    blank_threshold = 0.95
    blank_columns = []
    
    for column in csv_data.columns:
        non_empty_values = csv_data[column].dropna()
        non_empty_values = non_empty_values[non_empty_values.str.strip() != '']
        
        empty_ratio = 1 - (len(non_empty_values) / len(csv_data))
        
        if empty_ratio >= blank_threshold:
            blank_columns.append(column)
    
    return blank_columns
```

### 2. Parent vs Child Data Separation Logic

**Parent-level data criteria:**
- Values are identical across ALL children in the parent group
- Includes metadata: MASTER, MANUFACTURER_NAME, PRODUCT_STATUS, etc.
- Structural attributes: GROUP_STRING, ORDER_UNIT, CONTENT_UNIT

**Child-level data criteria:**  
- Values vary between at least 2 children in the group
- Unique identifiers: SUPPLIER_PID, INTERNATIONAL_PID
- Variable attributes: sizes, colors, specific feature values
- URLs and references that differ per child

### 3. Compressed JSON Structure

```json
{
  "parent_sku": "4307",
  "parent_data": {
    "group_string": "Root|Arbeitskleidung|Latzhosen",
    "product_status": "ACTIVE",
    "master": "4307",
    "description_long": "Wer den beliebten Genua-Cord als Latzhose...",
    "manufacturer_name": "EIKO",
    "manufacturer_type_description": "LAHN",
    "order_unit": "C62",
    "content_unit": "C62"
  },
  "children_data": [
    {
      "child_id": "4307_40_102",
      "supplier_pid": "4307_40_102",
      "description_short": "LAHN - Latzhose - Genuacord - Schwarz - Größe: 102",
      "international_pid": "4033976004693",
      "size": "102",
      "color_code": "40",
      "mime_thumb_1": "https://blob.redpim.de/..."
    }
  ],
  "metadata": {
    "total_children": 81,
    "compression_ratio": 0.68,
    "blank_columns_removed": 108,
    "parent_attributes": 49,
    "child_attributes": 7
  }
}
```

### 4. Data Type Preservation Rules

- **Integers**: Preserve as JSON numbers (not strings)
- **Floats**: Maintain precision, avoid scientific notation
- **Booleans**: Convert to JSON true/false  
- **Null/Empty**: Standardize to JSON null
- **Strings**: Preserve with proper JSON escaping
- **Arrays**: Convert delimited strings to JSON arrays where appropriate

### 5. Deterministic JSON Serialization

```python
def serialize_deterministically(data: dict) -> str:
    """Ensure consistent output across runs"""
    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,  # Alphabetical key ordering
        separators=(',', ': ')
    )
```

## Validation and Quality Metrics

### Data Integrity Checks
1. **Completeness**: All non-empty original values present in compressed format
2. **Accuracy**: Values match original types and content
3. **Consistency**: Parent data correctly extracted (identical across children)
4. **Structure**: JSON schema validation passes

### Compression Quality Metrics
1. **Size Reduction**: Percentage decrease in file size
2. **Column Efficiency**: Blank columns removed / total columns
3. **Redundancy Elimination**: Parent data extracted / total duplicate values
4. **Processing Speed**: Milliseconds per row processed

### Validation Test Cases
- **Small parent group**: 26-28 children (parents 41282, 41285, 41382, 41385)
- **Medium parent group**: 81 children (parent 4307)  
- **Large parent group**: 126 children (parent 4301)
- **Mixed data types**: Numbers, strings, booleans, nulls
- **Unicode content**: German text, special characters
- **Edge cases**: Single varying column, mostly empty data

## Error Handling Requirements

### Input Validation
- Verify CSV file exists and is readable
- Check minimum required columns present
- Validate parent group consistency

### Processing Errors  
- Handle malformed CSV data gracefully
- Report columns that cannot be categorized
- Provide fallback for ambiguous parent/child classification

### Output Validation
- Verify JSON structure matches schema
- Check compression ratio meets minimum threshold
- Validate data type preservation

## Performance Targets

- **Processing Speed**: ≤25ms per row
- **Memory Usage**: ≤500MB peak for largest parent group
- **Compression Ratio**: ≥50% size reduction
- **Accuracy**: 100% data preservation for non-empty values
- **Determinism**: Identical output across multiple runs

## Success Criteria Summary

1. **Functional**: Remove 95%+ blank columns, separate parent/child data correctly
2. **Quality**: Preserve all data types and values without loss
3. **Performance**: Achieve ≥50% size reduction in ≤2 seconds per parent group
4. **Reliability**: 100% deterministic output with comprehensive error handling
5. **Maintainability**: Clear separation of concerns, comprehensive validation