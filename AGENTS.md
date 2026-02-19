# AGENTS.md

## Execution and cost-control rules

- Run long-running commands with TTY enabled and monitor streaming logs continuously.
- Determine completion only after both conditions are met: logs indicate completion, and expected output files exist.
- Do not rerun any API-costly command unless the user explicitly requests the rerun.
