# AGENTS.md

## Execution and cost-control rules

- Run long-running commands with TTY enabled and monitor streaming logs continuously.
- Determine completion only after both conditions are met: logs indicate completion, and expected output files exist.
- Do not rerun any API-costly command unless the user explicitly requests the rerun.
- Python commands must be executed via `uv` from repo root:
  - `cd /Users/yuuta/Research/asset_placer_eval_standalone && uv run python ...`
  - Do not use plain `python` / `python3` directly.

## Sub-agent batch execution contract (important)

- When running batch jobs with sub-agents, each agent must execute from repo root explicitly:
  - `cd /Users/yuuta/Research/asset_placer_eval_standalone && <command>`
- For Python tasks in sub-agents, always use:
  - `cd /Users/yuuta/Research/asset_placer_eval_standalone && uv run python ...`
- Always use the project venv interpreter explicitly for pipeline commands:
  - `/Users/yuuta/Research/asset_placer_eval_standalone/.venv/bin/python ...`
  - Do not rely on implicit interpreter resolution.
- Before API-costly batch runs, run a cheap preflight import check once:
  - `/Users/yuuta/Research/asset_placer_eval_standalone/.venv/bin/python -c "import openai, dotenv; print('ok')"`
- For JSON batch runner (`experiments/src/run_pipeline_from_json.py`), ensure `defaults.python_exec` points to `.venv/bin/python`.
- If a run fails before any API call due to environment/module error, fixing environment and retrying is allowed.
- If a run reaches API call stage, do not rerun unless the user explicitly requests it.
