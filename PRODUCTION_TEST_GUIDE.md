# Production Workflow Test System

## Overview

The Production Workflow Test System provides comprehensive end-to-end testing for the AI mapping pipeline with real Gemini-2.5-flash API connectivity. This system validates the complete workflow from data loading through AI mapping with production-quality monitoring and reporting.

## Features

- **Complete Pipeline Testing**: End-to-end validation of all workflow stages
- **Real API Integration**: Tests actual Gemini-2.5-flash connectivity and responses
- **Step-by-Step Validation**: File existence checks for each pipeline stage
- **Performance Monitoring**: Comprehensive metrics collection and reporting
- **Production-Quality Error Handling**: Robust error recovery and detailed logging
- **Scalable Parent Processing**: Tests with parent 4301 first, then scales to all parents

## Pipeline Stages Tested

1. **SKU Analysis**: Process input files with CSV export and JSON compression
2. **CSV Splitting Validation**: Verify CSV files are created for each parent group
3. **JSON Compression Validation**: Confirm compressed data files exist
4. **Flat File Analysis Validation**: Check template processing outputs
5. **API Connectivity Testing**: Real Gemini-2.5-flash API validation
6. **AI Mapping Execution**: Complete AI mapping for all parent directories

## Quick Start

### Prerequisites

1. **Environment Setup**:
   ```bash
   # Ensure .env file exists with your Google API key
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

2. **Input Files**: Required test files must be present:
   - `test-files/EIKO Stammdaten.xlsx` (primary data)
   - `test-files/PANTS (3).xlsm` (template data)

### Running Tests

#### System Validation Only
```bash
python run_production_test.py --validate
```
Checks system readiness without executing the full workflow.

#### Quick Test (Validation + API Check)
```bash
python run_production_test.py --quick
```
Validates system and tests basic API connectivity.

#### Complete Production Test
```bash
python run_production_test.py --full
```
Executes the complete end-to-end workflow test.

#### Direct Execution
```bash
python production_workflow_test.py
```
Runs the complete production workflow directly.

## Test Execution Flow

### Phase 1: System Validation
- ✅ Environment configuration check
- ✅ Input file validation  
- ✅ Core module imports
- ✅ Output directory structure
- ✅ Basic component initialization

### Phase 2: Pipeline Execution
```
Step 1: SKU Analysis with CSV Export
├── Process EIKO Stammdaten.xlsx
├── Process PANTS (3).xlsm template
├── Generate parent-child relationships
└── Create job with unique timestamp ID

Step 2: CSV Splitting Validation  
├── Verify csv_splits/ directory exists
├── Count generated CSV files
└── Validate file sizes and content

Step 3: JSON Compression Validation
├── Find all parent_* directories  
├── Check step2_compressed.json files
└── Validate compression completion

Step 4: Flat File Analysis Validation
├── Check flat_file_analysis/ directory
├── Validate step1_template_columns.json
├── Validate step2_extracted_values.json
└── Validate step3_mandatory_fields.json

Step 5: API Connectivity Testing
├── Load Google API key
├── Test Gemini-2.5-flash connectivity
├── Measure response times
└── Validate API responses

Step 6: AI Mapping All Parents
├── Start with parent 4301 (validation)
├── Process remaining parents if successful
├── Generate step3_ai_mapping.json files
└── Collect performance metrics
```

### Phase 3: Reporting
- 📊 Comprehensive execution report
- ⏱️ Performance metrics and timing
- 📁 List of all created files
- 🔍 Validation results for each step
- ❌ Error details and recovery information

## Output Structure

After successful execution, the test creates:

```
production_output/{job_id}/
├── analysis_{job_id}.json              # SKU analysis results
├── csv_splits/                         # CSV exports by parent
│   ├── parent_4301_data.csv
│   └── parent_{sku}_data.csv...
├── parent_{sku}/                       # Per-parent directories
│   ├── step2_compressed.json           # Compressed product data
│   └── step3_ai_mapping.json          # AI mapping results
├── flat_file_analysis/                 # Template analysis
│   ├── step1_template_columns.json
│   ├── step2_extracted_values.json
│   └── step3_mandatory_fields.json
└── workflow_test_report.json           # Comprehensive test report
```

## Error Handling

The system includes comprehensive error handling:

- **Pipeline Validation Errors**: Missing files stop execution immediately
- **API Connectivity Issues**: Fallback validation and detailed error reporting  
- **Processing Failures**: Individual parent failures don't stop the pipeline
- **Recovery Information**: Detailed error context for debugging

## Performance Monitoring

Metrics collected include:

- **Step Duration**: Time for each pipeline stage
- **File Creation**: Count and size of generated files
- **API Performance**: Response times and success rates
- **AI Mapping Quality**: Confidence scores and mapping success rates
- **Resource Usage**: Memory and processing efficiency

## Configuration

### Environment Variables
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Input File Requirements
- **Primary Data**: Excel file with product/SKU information
- **Template Data**: Excel template with field definitions
- **File Format**: .xlsx or .xlsm supported
- **Size Limits**: Tested with files up to several MB

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   Solution: Ensure .env file exists and contains GOOGLE_API_KEY
   ```

2. **Input Files Missing**
   ```
   Solution: Verify test-files/ directory contains required Excel files
   ```

3. **Import Errors**
   ```
   Solution: Ensure all dependencies are installed and paths are correct
   ```

4. **Pipeline Validation Failures**
   ```
   Solution: Check previous steps completed successfully, verify file permissions
   ```

### Debug Mode

For detailed debugging, check the log files:
```bash
# Log files are created with timestamp
production_test_YYYYMMDD_HHMMSS.log
```

## Integration with Existing Workflow

The production test system integrates with:

- **SkuPatternAnalyzer**: Core analysis engine
- **AIMapingIntegration**: AI mapping coordination
- **JobManager**: Job tracking and management
- **Existing Templates**: Reuses current prompt and processing templates

## Development Notes

### Architecture

The system follows clean code principles:
- **Single Responsibility**: Each test method validates one specific aspect
- **Error Boundaries**: Comprehensive exception handling at each level  
- **Performance Monitoring**: Built-in metrics collection
- **Modular Design**: Easy to extend with additional test scenarios

### Testing Approach

- **Production-First**: Real API calls, actual file processing
- **Validation-Driven**: File existence checks prevent partial failures
- **Performance-Aware**: Monitoring ensures acceptable response times
- **Error-Resilient**: Individual failures don't crash the entire test suite

## Future Enhancements

Planned improvements:
- **Batch Size Configuration**: Adjustable concurrency limits
- **Custom Model Selection**: Support for different AI models
- **Extended Validation**: Content validation beyond file existence
- **Performance Benchmarking**: Historical performance comparisons