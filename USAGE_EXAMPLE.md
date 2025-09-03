# Test File Workflow Executor - Usage Examples

## Quick Start

Run the interactive workflow executor:

```bash
python test_file_workflow_executor.py
```

## Usage Examples

### 1. Interactive File Selection
```
🚀 Test File Workflow Executor
==================================================

📋 Available Test File Combinations:
==================================================
1. SKU: EIKO Stammdaten copy.xlsx (0.1MB)
   Flat: PANTS (3).xlsm (1.1MB)

Enter selection (number, 'all', or 'quit'): 1
```

### 2. Process All Files
```
Enter selection (number, 'all', or 'quit'): all
```

### 3. Cancel Execution
```
Enter selection (number, 'all', or 'quit'): quit
👋 Workflow execution cancelled by user
```

## File Structure Requirements

Place test files in the `/test-files` directory:
- `*.xlsx` files (SKU data files)
- `*.xlsm` files (flat template files)

The executor will automatically discover and pair these files for processing.

## Integration with Existing Workflow

The executor seamlessly integrates with `ProductionWorkflowTester` to run:
1. Input file validation
2. SKU analysis with CSV export
3. JSON compression validation  
4. Flat file analysis validation
5. API connectivity testing
6. AI mapping execution

## Clean Code Features

- **Single Responsibility**: Each class has one clear purpose
- **Dynamic Discovery**: No hardcoded file paths
- **Type Safety**: Full type hints throughout
- **Error Handling**: Comprehensive exception handling
- **Clean Architecture**: KISS and DRY principles applied