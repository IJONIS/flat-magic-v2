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

## Development Workflow – Phase-Based Execution

Follow a unified development framework with automatic phase detection, parallel agent orchestration, and MCP server utilization.

### CRITICAL: Parallel Agent Orchestration
**Evaluate after EVERY message**: Identify tasks that can be delegated to specialized agents in `/.claude/agents/` for parallel execution.

**Agent Delegation Strategy**
- **Multi-file operations** (>3 files): `general-purpose`
- **Python expertise needed**: `python-expert`
- **Architecture decisions**: `system-architect`
- **Code quality/refactoring**: `refactoring-expert`
- **Testing strategy**: `quality-engineer`
- **Performance optimization**: `performance-engineer`
- **Requirements analysis**: `requirements-analyst`
- **Documentation**: `technical-writer`
- **Security validation**: `security-engineer`

**MCP Server Integration** (evaluate and use as needed)
- **Context7**: Official library docs (e.g., pandas, openpyxl, Google AI SDK)
- **Sequential**: Complex pipeline analysis & system design
- **Serena**: Semantic code ops & project memory
- **Playwright**: Browser-based testing/validation

**Parallel Execution Rules**
1. Always assess if current work can be split across agents.
2. Delegate immediately when >3 files or complex analysis is involved.
3. Use MCP servers for specialized capabilities (docs, analysis, memory).
4. Coordinate and reconcile agent outputs before proceeding.
5. Document delegation in commit messages with agent IDs.

**Delegation Decision Matrix**
```
Task Type                           | Agent + MCP Server
-----------------------------------|--------------------------
Excel/CSV parsing                   | python-expert + Context7
Pipeline/architecture               | system-architect + Sequential
Multi-module implementation         | general-purpose + Serena
Performance optimization            | performance-engineer + Sequential
Test strategy & coverage            | quality-engineer + Serena
Code refactoring                    | refactoring-expert + Serena
Documentation & traceability        | technical-writer + Serena
Security review                     | security-engineer + Sequential
Requirements analysis               | requirements-analyst + Context7
```

**MANDATORY**: Before responding to any request, explicitly state which agents and MCP servers will be utilized and why.

### Phase Detection Triggers
- **Permissions Check**: Inspect `global settings.json` to know where/when to request perms.
- **Planning Phase**: “plan/design/requirements” signals.
- **Execution Phase**: “implement/code/build/create” signals.
- **Review Phase**: “review/check/validate/test” signals.

---

## Pipeline Step Validation Requirements

**Purpose**: Ensure data integrity and immediate failure detection when pipeline steps fail to produce required outputs.

### Step Completion Validation
The `completion_check()` function enforces strict file existence requirements:

**Step 1 (Analysis)**: 
- **Required**: `analysis_results.json` must exist
- **Failure**: Pipeline stops immediately if file missing

**Step 2 (CSV Export)**:
- **Required**: `parent_*/data.csv` files must exist (when CSV export enabled)
- **Validation**: Checks each parent folder for corresponding CSV file
- **Failure**: Pipeline stops if any expected CSV file is missing

**Step 3 (Compression)**:
- **Required**: `parent_*/step2_compressed.json` files must exist (when compression enabled)
- **Validation**: Verifies compressed output for each parent directory
- **Failure**: Pipeline stops if any expected compressed file is missing

### Data Integrity Enforcement
- **No Partial Results**: Pipeline fails fast rather than continuing with incomplete data
- **File Existence Only**: Validation checks presence, not content validity
- **Configuration Aware**: Only validates files that should exist based on current settings
- **Immediate Termination**: Any missing file causes complete pipeline abort

---

## 1) Planning Phase

**Agent Utilization**
- `requirements-analyst`, `system-architect`
- MCP: **Sequential** (architecture), **Context7** (library selection)

**Required Outputs**
- **Overview**: 2–4 sentences summarizing the goal.
- **Function List**: Each function with a 1–3 sentence purpose.
- **Acceptance Criteria (AC)**: 5–10+ criteria across:
  - **AC-F** Functional (Gherkin Given/When/Then)
  - **AC-E** Edge/Error cases with typed errors, no mutation, logging
  - **AC-N** Non-functional (quantified perf/observability/i18n/accessibility)
  - **AC-D** Data integrity (idempotency/transactional safety)
- **Test Plan**: Unit scopes mapped to AC IDs; targets ≥80% lines, ≥70% branches; test data strategy.

**Enterprise Sanity Check**
- “Is this overbuilt for pre-customer? Remove needless fallbacks/versioning/testing bloat.”

**Example AC Template**
```
AC-F1: Given valid input
       When the pipeline runs
       Then structured output is produced according to the schema

AC-E1: Given malformed input
       When processing starts
       Then FileFormatError is raised, no mutation occurs, and a warn log is emitted

AC-N1: Given 100 items
       When processed in batch
       Then p95 latency ≤ 50ms per batch

AC-D1: Given the same input processed twice
       When executed
       Then the output is identical (idempotent)
```

---

## 2) Execution Phase

**Agent Utilization**
- `python-expert`, `general-purpose`, `performance-engineer`, `quality-engineer`
- MCP: **Context7** (docs), **Serena** (memory/symbol ops)

**Required Process (tight TDD loop per AC)**
1. Write failing unit test(s) for the next AC.
2. Implement minimal code to pass.
3. Lint & compile.
4. Run tests.
5. Refactor (clarity/DRY). Re-run tests.
6. Commit with message linking AC IDs and Test IDs.

**Unit Test Requirements (per function)**
- Happy path (maps to AC-F*)
- Boundaries (min/max, empty, unicode, large payloads)
- Errors & exceptions (AC-E*)
- Idempotency/retry semantics (AC-D1)
- Performance micro-checks (AC-N1)
- Analytics/log events (e.g., AC-N* for observability)
- i18n/locale behaviors (AC-N*)

**Test Skeleton**
```python
describe "module_under_test":
  it "T-F1: satisfies AC-F1 - processes input as specified"
  it "T-E1: raises ExpectedError and leaves state unchanged"
  it "T-D1: is idempotent on duplicate execution"
  it "T-N*: emits telemetry event with expected payload"
```

**Post Development Run**
- In the final response, list **all** files edited so they can be inspected.

**Data Quality Rules**
- **NO MOCK DATA (in production)**: Do not use placeholders/fake/hardcoded values.
- **DYNAMIC DATA ONLY**: Derive values from actual inputs/templates/configs.
- **REAL VALIDATION RULES**: Extract constraints from real schemas/specs.
- **NO HARDCODED CONSTANTS**: Make values configurable or derive from inputs.
- **DATA INTEGRITY**: Maintain genuine data flows; avoid simulated responses.
- **ASK FOR MOCKS** (only if unavoidable in dev): Explicitly request exact mock specs.

**Development Rules**
- Think hard; write elegant, minimal code.
- Implement only what's needed **now**.

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

## 3) Review Phase

**Agent Utilization**
- `refactoring-expert`, `security-engineer`, `quality-engineer`, `performance-engineer`, `technical-writer`

**Required Verification**
- All ACs present with 1:1 test mapping (traceability table).
- Coverage targets met (≥80% lines, ≥70% branches).
- No flaky tests; meaningful assertions.
- No premature abstractions, dead code, or needless versioning.
- Observability events & error taxonomy consistent.
- Code is clean, readable, and DRY.

**PR Traceability Table (mandatory)**
```
AC ID    File(s) touched              Test ID(s)    Status
AC-F1    pkg/core/module.py           T-F1          Pass
AC-E1    pkg/core/module.py           T-E1          Pass
AC-D1    pkg/core/module.py           T-D1          Pass
AC-N*    pkg/core/telemetry.py        T-N*          Pass
```

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
