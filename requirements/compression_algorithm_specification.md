# CSV-to-JSON Compression Algorithm Specification

## Algorithm Overview

Multi-phase compression pipeline that eliminates redundancy through blank column removal and parent-child data separation while preserving full data integrity.

## Phase 1: Column Analysis and Classification

### Blank Column Detection Algorithm

```python
def analyze_column_patterns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Classify each column as: 'blank', 'parent', or 'child'
    
    Returns:
        Dict mapping column_name -> classification
    """
    BLANK_THRESHOLD = 0.95  # 95% empty = blank column
    
    classifications = {}
    
    for column in df.columns:
        # Step 1: Check for blank column
        non_null_values = df[column].dropna()
        non_empty_values = non_null_values[
            non_null_values.astype(str).str.strip() != ''
        ]
        
        empty_ratio = 1 - (len(non_empty_values) / len(df))
        
        if empty_ratio >= BLANK_THRESHOLD:
            classifications[column] = 'blank'
            continue
        
        # Step 2: Check for parent vs child data
        unique_non_empty = set(non_empty_values.astype(str))
        
        if len(unique_non_empty) <= 1:
            # All non-empty values are identical = parent data
            classifications[column] = 'parent'
        else:
            # Multiple unique values = child data  
            classifications[column] = 'child'
    
    return classifications
```

### Enhanced Pattern Detection

```python
def detect_structured_blanks(df: pd.DataFrame, column: str) -> bool:
    """
    Detect if 'blank' column has meaningful structure
    (e.g., all zeros vs truly empty)
    """
    values = df[column].fillna('')
    non_empty = values[values != '']
    
    if len(non_empty) == 0:
        return True  # Truly blank
    
    # Check if all non-empty values are structural zeros/defaults
    structural_defaults = {'0', '0.0', 'false', 'null', 'none', ''}
    unique_values = set(str(v).lower().strip() for v in non_empty)
    
    return unique_values.issubset(structural_defaults)
```

## Phase 2: Data Separation and Extraction

### Parent Data Extraction

```python
def extract_parent_data(df: pd.DataFrame, parent_columns: List[str]) -> Dict[str, Any]:
    """
    Extract parent-level data (identical across all children)
    """
    parent_data = {}
    
    for column in parent_columns:
        # Get first non-null value as the parent value
        non_null_values = df[column].dropna()
        
        if len(non_null_values) > 0:
            parent_value = non_null_values.iloc[0]
            
            # Preserve data types
            parent_data[column.lower()] = convert_data_type(parent_value)
    
    return parent_data

def convert_data_type(value: Any) -> Any:
    """
    Preserve original data types in JSON-compatible format
    """
    if pd.isna(value):
        return None
    
    # Try numeric conversion
    try:
        # Integer
        if str(value).isdigit() or (str(value).startswith('-') and str(value)[1:].isdigit()):
            return int(value)
        
        # Float
        float_val = float(value)
        if not math.isnan(float_val):
            return float_val
    except (ValueError, TypeError):
        pass
    
    # Boolean conversion
    str_val = str(value).lower().strip()
    if str_val in ('true', '1', 'yes', 'on'):
        return True
    elif str_val in ('false', '0', 'no', 'off'):
        return False
    
    # Default to string
    return str(value).strip()
```

### Child Data Processing

```python
def extract_children_data(df: pd.DataFrame, child_columns: List[str]) -> List[Dict[str, Any]]:
    """
    Extract child-specific varying data
    """
    children_data = []
    
    for _, row in df.iterrows():
        child_record = {}
        
        # Always include primary identifier
        if 'SUPPLIER_PID' in df.columns:
            child_record['child_id'] = row['SUPPLIER_PID']
        
        # Add varying attributes
        for column in child_columns:
            value = row[column]
            
            if pd.notna(value) and str(value).strip():
                # Convert column name to JSON-friendly format
                json_key = column.lower().replace('_', '_')
                child_record[json_key] = convert_data_type(value)
        
        children_data.append(child_record)
    
    return children_data
```

## Phase 3: JSON Structure Generation

### Optimized JSON Schema

```python
def generate_compressed_json(parent_sku: str, parent_data: Dict, 
                           children_data: List[Dict], metadata: Dict) -> Dict:
    """
    Generate final compressed JSON structure
    """
    compressed_structure = {
        "parent_sku": parent_sku,
        "parent_data": parent_data,
        "children_data": children_data,
        "metadata": {
            "total_children": len(children_data),
            "compression_stats": metadata,
            "generated_at": datetime.utcnow().isoformat(),
            "schema_version": "1.0"
        }
    }
    
    return compressed_structure
```

### Deterministic Serialization

```python
def serialize_json_deterministically(data: Dict) -> str:
    """
    Ensure consistent output across multiple runs
    """
    return json.dumps(
        data,
        ensure_ascii=False,        # Preserve Unicode
        indent=2,                  # Human readable
        sort_keys=True,           # Alphabetical ordering
        separators=(',', ': '),   # Clean formatting
        default=handle_special_types
    )

def handle_special_types(obj):
    """Handle non-standard types for JSON serialization"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)
```

## Phase 4: Validation and Quality Metrics

### Compression Ratio Calculation

```python
def calculate_compression_metrics(original_csv_size: int, 
                                compressed_json_size: int,
                                column_stats: Dict) -> Dict:
    """
    Calculate comprehensive compression metrics
    """
    total_columns = sum(column_stats.values())
    
    return {
        "file_size_reduction": {
            "original_bytes": original_csv_size,
            "compressed_bytes": compressed_json_size,
            "reduction_ratio": 1 - (compressed_json_size / original_csv_size),
            "reduction_percentage": f"{((original_csv_size - compressed_json_size) / original_csv_size) * 100:.1f}%"
        },
        "column_optimization": {
            "total_columns": total_columns,
            "blank_columns_removed": column_stats.get('blank', 0),
            "parent_columns_extracted": column_stats.get('parent', 0),
            "child_columns_preserved": column_stats.get('child', 0),
            "column_reduction_ratio": column_stats.get('blank', 0) / total_columns
        },
        "data_efficiency": {
            "redundancy_eliminated": column_stats.get('parent', 0) * column_stats.get('rows', 0),
            "data_density_improvement": f"{(column_stats.get('child', 0) / total_columns) * 100:.1f}%"
        }
    }
```

### Data Integrity Validation

```python
def validate_compression_integrity(original_df: pd.DataFrame,
                                 compressed_json: Dict) -> Dict[str, bool]:
    """
    Comprehensive validation of compression accuracy
    """
    validation_results = {}
    
    # 1. Row count preservation
    original_rows = len(original_df)
    compressed_rows = len(compressed_json['children_data'])
    validation_results['row_count_preserved'] = (original_rows == compressed_rows)
    
    # 2. Parent data consistency
    parent_data = compressed_json['parent_data']
    validation_results['parent_data_consistent'] = validate_parent_consistency(
        original_df, parent_data
    )
    
    # 3. Child data completeness  
    validation_results['child_data_complete'] = validate_child_completeness(
        original_df, compressed_json['children_data']
    )
    
    # 4. Data type preservation
    validation_results['data_types_preserved'] = validate_data_types(
        original_df, compressed_json
    )
    
    return validation_results

def validate_parent_consistency(df: pd.DataFrame, parent_data: Dict) -> bool:
    """Verify parent data represents true constants across all children"""
    for json_key, expected_value in parent_data.items():
        # Convert JSON key back to CSV column name
        csv_column = json_key.upper()
        
        if csv_column in df.columns:
            unique_values = df[csv_column].dropna().unique()
            
            # Should have exactly one unique value
            if len(unique_values) != 1:
                return False
                
            # Value should match extracted parent data
            if convert_data_type(unique_values[0]) != expected_value:
                return False
    
    return True
```

## Example Processing Flow

### Input CSV Analysis
```
Original CSV: parent_4307/data.csv
- Rows: 81 children
- Columns: 164 total
- Size: ~847KB

Column Analysis Results:
- Blank columns: 108 (65.8%)
- Parent columns: 49 (29.9%) 
- Child columns: 7 (4.3%)
```

### Compression Process
```python
# Step 1: Analyze structure
classifications = analyze_column_patterns(df)

# Step 2: Extract data
parent_data = extract_parent_data(df, parent_columns)
children_data = extract_children_data(df, child_columns)

# Step 3: Generate JSON
compressed_json = generate_compressed_json(
    parent_sku="4307",
    parent_data=parent_data,
    children_data=children_data,
    metadata=compression_stats
)

# Step 4: Serialize deterministically
json_output = serialize_json_deterministically(compressed_json)
```

### Output JSON Example
```json
{
  "parent_sku": "4307",
  "parent_data": {
    "group_string": "Root|Arbeitskleidung|Latzhosen",
    "manufacturer_name": "EIKO",
    "master": "4307",
    "order_unit": "C62",
    "product_status": "ACTIVE"
  },
  "children_data": [
    {
      "child_id": "4307_40_102",
      "description_short": "LAHN - Latzhose - Genuacord - Schwarz - Größe: 102",
      "fvalue_3_2": "102",
      "international_pid": "4033976004693",
      "supplier_pid": "4307_40_102"
    }
  ],
  "metadata": {
    "compression_stats": {
      "file_size_reduction": {
        "reduction_percentage": "68.4%",
        "reduction_ratio": 0.684
      },
      "column_optimization": {
        "blank_columns_removed": 108,
        "column_reduction_ratio": 0.658,
        "parent_columns_extracted": 49
      }
    },
    "total_children": 81
  }
}
```

## Performance Optimization Strategies

### Memory Efficiency
- Stream CSV processing for large files
- Lazy evaluation of column classifications  
- Chunked data processing for parent groups >500 rows

### Processing Speed
- Vectorized operations using pandas
- Parallel processing for multiple parent groups
- Caching of column classification results

### Output Optimization  
- Minimal JSON formatting for production use
- Optional pretty-printing for development
- Compression-friendly key naming conventions