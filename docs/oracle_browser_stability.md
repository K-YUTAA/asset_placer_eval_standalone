# Oracle Browser Stability Guide

## Goal

Make Oracle browser-mode runs more reliable for this repository and avoid:

- `socket hang up`
- Browser prompt not being injected
- Stale profile lock issues

Also enforce policy:

- `--engine api` is prohibited.

## Recommended command

Use the repo wrapper script:

```bash
cd /Users/yuuta/Research/asset_placer_eval_standalone
scripts/oracle_browser_stable.sh \
  --slug current-code-review-lite \
  -p "Review this code..." \
  --file "README.md" \
  --file "experiments/src/generate_layout_json.py" \
  --file "experiments/src/step2_rule_based.py"
```

The wrapper:

1. Uses a wrapper-level profile lock (`oracle-wrapper.lock`) to serialize runs.
2. Cleans stale Oracle browser profile state before browser launch (only when needed).
3. Ensures DevTools endpoint is reachable (`127.0.0.1:9222`) before Oracle run.
4. Optionally closes existing ChatGPT tabs and opens a fresh one via DevTools.
5. Performs optional cookie DB precheck for ChatGPT session token.
6. Blocks forbidden API-related arguments.
7. Runs Oracle in browser mode with GPT-5.2 Pro and manual-login/reuse lock flags.

## Stable operating flow

1. Keep prompt/file set small.
2. Run `--dry-run summary --files-report` first.
3. Start browser run once.
4. If run detaches, reattach with:

```bash
npx -y @steipete/oracle status --hours 6
npx -y @steipete/oracle session <slug-or-id> --render
```

5. Do not switch to API mode.

## Notes

- Large bundles can degrade reliability. Prefer targeted files.
- If DevTools does not come up, inspect:
  - `/tmp/oracle-browser-stable.log`
- Tunable env vars:
  - `ORACLE_DEVTOOLS_WAIT_SECONDS` (default `30`)
  - `ORACLE_BROWSER_LOG` (default `/tmp/oracle-browser-stable.log`)
  - `ORACLE_OPEN_CHATGPT_TAB` (`1`/`0`, default `1`)
  - `ORACLE_CLOSE_EXISTING_CHATGPT_TABS` (`1`/`0`, default `1`)
  - `ORACLE_PROFILE_LOCK_TIMEOUT_SECONDS` (default `300`)
  - `ORACLE_CHECK_SESSION_COOKIE` (`1`/`0`, default `1`)
  - `ORACLE_FAIL_IF_NO_SESSION_COOKIE` (`1`/`0`, default `0`)
  - `ORACLE_FORCE_MANUAL_LOGIN` (`1`/`0`, default `1`)
  - `ORACLE_BROWSER_REUSE_WAIT` (default `15s`)
  - `ORACLE_BROWSER_PROFILE_LOCK_TIMEOUT` (default `5m`)
