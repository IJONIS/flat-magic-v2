 Start Working with Global Agent

Quick command to begin working with a specific global agent.

## Usage
`/start {agent-name} "{task description}"`

## Agent: $ARGUMENTS

## Process
1. Read current context: `@context/current-state.md`
2. Create handoff file: `context/handoff-{agent}.md` with task details
3. Update project state with new assignment
4. Provide exact command for user to run

## Available Global Agents
- requirements-analyst, system-architect, backend-architect
- frontend-architect, python-expert, security-engineer
- performance-engineer, quality-engineer, refactoring-expert  
- root-cause-analyst, technical-writer

## Output Format
✅ Setup complete for @{agent}
Next Command:
claude @{agent}
Task Created: context/handoff-{agent}.md
Status Updated: context/current-state.md
The agent will automatically load context and begin work.

Execute this agent startup process.