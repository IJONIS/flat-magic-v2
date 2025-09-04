# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CORE PROGRAMMING PRINCIPLES - HIGHEST PRIORITY

**Embody the principles of Clean Code, KISS and DRY**

- **Clean Code**: Write code that is readable, maintainable, and expressive
- **KISS (Keep It Simple, Stupid)**: Favor simplicity over complexity in all design decisions
- **DRY (Don't Repeat Yourself)**: Eliminate code duplication through abstraction and reuse

These principles override all other considerations when programming new code or updating existing files.

## CRITICAL: SCOPE DISCIPLINE - AVOID OVER-COMPLICATION

**NEVER add functionality beyond explicit requirements**

- **Build ONLY what's asked**: If requirement is "extract valid values from file", do ONLY that
- **No feature creep**: Never add performance monitoring, metrics, analytics, or logging unless explicitly requested
- **No additional layers**: Avoid extra classes, modules, or abstractions not needed for the core requirement
- **No assumed requirements**: Don't implement error recovery, caching, optimization, or validation unless specified
- **Single purpose implementation**: Each function should solve exactly one stated problem
- **Question before expanding**: If unclear whether something is needed, ASK rather than implement

**Examples of over-complication to AVOID**:
- Requirement: "extract values from file" → DON'T add: performance metrics, multiple file formats, caching, logging
- Requirement: "validate input" → DON'T add: multiple validation strategies, performance tracking, error analytics  
- Requirement: "save data" → DON'T add: backup systems, compression, encryption, audit trails

**This is a CRITICAL requirement - over-complication has been a persistent issue.**

### 2. NO BACKWARDS COMPATIBILITY
- **NEVER** include backwards compatibility measures
- **NEVER** suggest legacy fallbacks or support for older systems
- **NEVER** implement polyfills, shims, or compatibility layers

### 3. SIMPLICITY FIRST
- **Prefer simple, elegant solutions** over complex architectures
- **Minimize file count** - consolidate when possible
- **Reduce dependencies** - use built-in features when available
- **Avoid over-engineering** - solve the immediate problem, not hypothetical ones
- **Single responsibility** - each file/module should have one clear purpose

**File Change Tracking (MANDATORY)**
- List **all** created/edited/modified files after each dev cycle.
- **Format**: `path/to/file.py` – [created|modified|updated]
- Reconcile sub-agent change reports; cross-check timestamps.

**File Management Rules**
- **Single Source of Truth**: Edit the canonical file directly.
- **No file duplication**: Avoid suffixes like `_fix`, `_v2`, etc.
- **No backup copies**: Use version control for history.
- **No MD summaries**: Do NOT create markdown files after development steps unless specifically requested. Avoid creating analysis.md, summary.md, or similar documentation files that become outdated and unmaintained.

---

## Clean Code Standards

### File Size Constraints
- **Max lines per file**: 400 (excl. comments/docstrings)
- **Max 50 lines** per function
- **Max 200 lines** per class
- Split modules that exceed limits.

### Code Quality Requirements
- **Single Responsibility Principle** per function/class.
- **Descriptive Naming**
  - Functions: `process_input()` not `process()`
  - Vars: `validated_schema` not `schema1`
  - Classes: `InputValidator` not `Validator`
- **No Magic Numbers**: Use named constants.
- **Type Hints**: Full annotations on all functions.
- **Docstrings**: Public functions/classes need examples.

### Maintainability
- **Cyclomatic Complexity** ≤ 10 per function.
- **Dependency Injection** over hard-coded deps.
- **Error Boundaries**: Wrap I/O & external calls with try/except.
- **Immutable Data** preferred; avoid in-place mutation.
- **Pure Functions** where possible.
- **Configuration**: Extract magic strings/values to config.

### Repository Organization
- Logical grouping by domain/module.
- Clear, absolute imports; avoid cycles.
- Consistent formatting: black, isort, flake8 (or equivalents).
- No dead code: remove unused imports/vars/functions.
- Meaningful commit messages that link AC/Test IDs.
- Self-documenting code via clear naming and structure.

### Performance & Security
- **Memory**: Stream/chunk large datasets; avoid loading all into RAM.
- **Input Validation** at module boundaries.
- **Logging**: Structured logs with levels (DEBUG/INFO/WARN/ERROR).
- **Resource Management**: Use context managers for files/conns.
- **Security**: No secrets in code; sanitize all inputs.

## Project Overview

SKU Pattern Analyzer - A production-ready Python 3.12+ application for analyzing SKU parent-child relationships in Excel files with AI-enhanced mapping and compression optimization.

## Architecture & Pipeline

### Processing Pipeline (5 Steps)
1. **Step 1**: Template analysis (`flat_file_analysis/step1_template_columns.json`)
2. **Step 2**: Value extraction (`flat_file_analysis/step2_valid_values.json`)
3. **Step 3**: Mandatory fields (`flat_file_analysis/step3_mandatory_fields.json`)
4. **Step 4**: Template structure (`flat_file_analysis/step4_template.json`)
5. **Step 5**: AI mapping (`parent_*/step5_ai_mapping.json`)

### Core Modules
- `sku_analyzer/core/`: Main processing (analyzer.py, hierarchy.py, compressor.py)
- `sku_analyzer/flat_file/`: Template analysis and extraction
- `sku_analyzer/step5_mapping/`: AI-powered mapping with Gemini API
- `sku_analyzer/optimization/`: Performance optimization and compression

### Output Structure
```
production_output/
└── [job_id]/
    ├── flat_file_analysis/
    │   ├── step1_template_columns.json
    │   ├── step2_valid_values.json
    │   ├── step3_mandatory_fields.json
    │   └── step4_template.json
    ├── parent_[sku]/
    │   ├── data.csv
    │   ├── step2_compressed.json
    │   └── step5_ai_mapping.json
    └── analysis_[job_id].json
```

## Common Commands

### Running the Application
```bash
# Standard processing with compression
python main.py "path/to/file.xlsx"

# With performance benchmarking
python main.py "path/to/file.xlsx" --full-bench

# Without CSV export
python main.py "path/to/file.xlsx" --no-csv

# Test with sample files
python main.py "test-files/EIKO Stammdaten.xlsx"
```

### Testing
```bash
# Run performance tests
python test_optimized_performance.py

# Test AI mapping safety filters
python test_safety_filter_robustness.py

# Execute complete workflow test
python test_file_workflow_executor.py
```

### Development
```bash
# Code formatting
black . --target-version py312 --line-length 100

# Linting
ruff check . --target-version py312 --line-length 100 --select E,F,W,I,N,UP

# Type checking
mypy . --python-version 3.12 --strict

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_performance.txt
```

## Key Implementation Details

### Pipeline Validation
The `completion_check()` function in `analyzer.py` enforces strict validation:
- Verifies all expected files exist at each step
- Fails fast if any required output is missing
- Configuration-aware (only validates enabled features)

### AI Mapping Integration
- Uses Google Gemini API for intelligent field mapping
- Implements robust safety filter handling
- Automatic retry logic for blocked content
- Format enforcement for consistent output

### Performance Optimizations
- Compression achieves ~71% reduction
- JSON serialization 15x faster with orjson
- Parallel processing for multi-parent datasets
- Automatic blank column removal

## Critical Patterns

### Error Handling
- All AI API calls wrapped with safety filter recovery
- Structured logging at each pipeline step
- Job-based isolation prevents conflicts
- Automatic validation after each step

### Data Flow
1. Excel → Pandas DataFrame (with dtype optimization)
2. Pattern extraction using delimiter-based hierarchy
3. Parent grouping with confidence scoring
4. CSV export with blank column removal
5. JSON compression with redundancy analysis
6. AI mapping with template-based transformation

## Dependencies

### Core Requirements
- Python 3.12+
- pandas>=2.2.0 (DataFrame operations)
- openpyxl>=3.1.0 (Excel reading)
- orjson>=3.11.3 (Fast JSON serialization)
- pydantic>=2.11.7 (Data validation)
- google-generativeai (AI mapping)

### Performance Libraries
- ujson (Alternative JSON encoder)
- orjson (Primary high-performance JSON)

## Important Notes

1. **Job Management**: Each run creates a unique job ID to prevent conflicts
2. **Step Dependencies**: Steps 3-5 require successful completion of previous steps
3. **AI Safety**: Gemini API may block responses - automatic retry handles this
4. **Memory Usage**: Large files processed with chunking to avoid memory issues
5. **Deterministic Output**: All operations sorted for reproducible results

## Testing Strategy

When modifying the pipeline:
1. Run `test_optimized_performance.py` to verify performance targets
2. Test with `test-files/EIKO Stammdaten.xlsx` for baseline validation
3. Check `completion_check()` passes for all steps
4. Verify AI mapping with `test_safety_filter_robustness.py`

## Global Expert Agent Pool
Your professional agents are available globally via Claude Code:
- **requirements-analyst** - Transform ambiguous ideas into concrete specifications
- **system-architect** - Design scalable architecture with maintainability focus  
- **backend-architect** - Reliable backend systems, APIs, databases, security
- **frontend-architect** - Accessible, performant UI with modern frameworks
- **python-expert** - Production-ready Python code following SOLID principles
- **security-engineer** - Vulnerability assessment and compliance verification
- **performance-engineer** - Measurement-driven optimization and bottleneck elimination
- **quality-engineer** - Comprehensive testing strategies and systematic QA
- **refactoring-expert** - Code quality improvement and technical debt reduction
- **root-cause-analyst** - Systematic problem investigation and evidence-based analysis
- **technical-writer** - Clear, comprehensive documentation for specific audiences

## Context Management Protocol
**MANDATORY for ALL agents:**
1. **READ FIRST**: `@context/current-state.md` - Current project status
2. **CHECK HANDOFF**: `@context/handoff-{your-agent-name}.md` - Your specific tasks
3. **UPDATE STATUS**: Add your work to current-state.md when starting/finishing
4. **CREATE HANDOFFS**: Write `context/handoff-{next-agent}.md` when passing work

## Agent Invocation
Since your agents are global, use:
```bash
claude @requirements-analyst
claude @backend-architect  
claude @frontend-architect
# etc.