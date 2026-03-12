#!/usr/bin/env bash
set -euo pipefail

# Oracle browser-mode runner with basic stability guards.
# Usage:
#   scripts/oracle_browser_stable.sh --slug my-review -p "..." --file "src/**"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -eq 0 ]]; then
  cat <<'EOF'
Usage:
  scripts/oracle_browser_stable.sh <oracle-args...>

Example:
  scripts/oracle_browser_stable.sh \
    --slug current-code-review-lite \
    -p "Review this code..." \
    --file "experiments/src/generate_layout_json.py"

Notes:
  - Always uses Oracle browser mode (never API mode).
  - Serializes runs with a wrapper-level profile lock.
  - Starts browser with extension-noise minimized.
  - Resets ChatGPT tabs and opens a fresh one before run.
  - Performs optional cookie/session precheck.
EOF
  exit 0
fi

ORACLE_PROFILE="${ORACLE_PROFILE:-$HOME/.oracle/browser-profile}"
ORACLE_DEBUG_PORT="${ORACLE_DEBUG_PORT:-9222}"
ORACLE_BROWSER_BIN="${ORACLE_BROWSER_BIN:-/Applications/Brave Browser.app/Contents/MacOS/Brave Browser}"
ORACLE_START_URL="${ORACLE_START_URL:-https://chatgpt.com/}"
ORACLE_DEVTOOLS_WAIT_SECONDS="${ORACLE_DEVTOOLS_WAIT_SECONDS:-30}"
ORACLE_BROWSER_LOG="${ORACLE_BROWSER_LOG:-/tmp/oracle-browser-stable.log}"
ORACLE_OPEN_CHATGPT_TAB="${ORACLE_OPEN_CHATGPT_TAB:-1}"
ORACLE_CLOSE_EXISTING_CHATGPT_TABS="${ORACLE_CLOSE_EXISTING_CHATGPT_TABS:-1}"
ORACLE_PROFILE_LOCK_TIMEOUT_SECONDS="${ORACLE_PROFILE_LOCK_TIMEOUT_SECONDS:-300}"
ORACLE_PROFILE_LOCK_POLL_SECONDS="${ORACLE_PROFILE_LOCK_POLL_SECONDS:-1}"
ORACLE_CHECK_SESSION_COOKIE="${ORACLE_CHECK_SESSION_COOKIE:-1}"
ORACLE_FAIL_IF_NO_SESSION_COOKIE="${ORACLE_FAIL_IF_NO_SESSION_COOKIE:-0}"
ORACLE_COOKIE_DB="${ORACLE_COOKIE_DB:-}"
ORACLE_FORCE_MANUAL_LOGIN="${ORACLE_FORCE_MANUAL_LOGIN:-1}"
ORACLE_BROWSER_REUSE_WAIT="${ORACLE_BROWSER_REUSE_WAIT:-15s}"
ORACLE_BROWSER_PROFILE_LOCK_TIMEOUT="${ORACLE_BROWSER_PROFILE_LOCK_TIMEOUT:-5m}"
ORACLE_DISABLE_EXTENSIONS="${ORACLE_DISABLE_EXTENSIONS:-1}"

DEVTOOLS_VERSION_URL="http://127.0.0.1:${ORACLE_DEBUG_PORT}/json/version"
DEVTOOLS_LIST_URL="http://127.0.0.1:${ORACLE_DEBUG_PORT}/json/list"
WRAPPER_LOCK_DIR="${ORACLE_PROFILE}/oracle-wrapper.lock"

for arg in "$@"; do
  case "$arg" in
    --engine|--engine=*|api|--api-key|--api-url)
      echo "error: forbidden argument detected: $arg" >&2
      echo "hint: this wrapper is browser-only and API mode is prohibited by policy." >&2
      exit 2
      ;;
  esac
done

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required command not found: $cmd" >&2
    exit 1
  fi
}

warn() {
  echo "warn: $*" >&2
}

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required but not found in PATH" >&2
  exit 1
fi

require_command npx

if [[ ! -x "$ORACLE_BROWSER_BIN" ]]; then
  echo "error: browser binary not found or not executable: $ORACLE_BROWSER_BIN" >&2
  exit 1
fi

release_wrapper_lock() {
  if [[ -f "${WRAPPER_LOCK_DIR}/pid" ]]; then
    local owner_pid
    owner_pid="$(cat "${WRAPPER_LOCK_DIR}/pid" 2>/dev/null || true)"
    if [[ "$owner_pid" == "$$" ]]; then
      rm -rf "$WRAPPER_LOCK_DIR" >/dev/null 2>&1 || true
    fi
  fi
}

acquire_wrapper_lock() {
  local started_at now elapsed owner_pid
  started_at="$(date +%s)"
  while ! mkdir "$WRAPPER_LOCK_DIR" >/dev/null 2>&1; do
    owner_pid="$(cat "${WRAPPER_LOCK_DIR}/pid" 2>/dev/null || true)"
    if [[ -n "$owner_pid" ]] && ! kill -0 "$owner_pid" >/dev/null 2>&1; then
      rm -rf "$WRAPPER_LOCK_DIR" >/dev/null 2>&1 || true
      continue
    fi
    now="$(date +%s)"
    elapsed=$((now - started_at))
    if (( elapsed >= ORACLE_PROFILE_LOCK_TIMEOUT_SECONDS )); then
      echo "error: timed out waiting for wrapper profile lock (${WRAPPER_LOCK_DIR})" >&2
      exit 1
    fi
    sleep "$ORACLE_PROFILE_LOCK_POLL_SECONDS"
  done
  printf '%s\n' "$$" > "${WRAPPER_LOCK_DIR}/pid"
  trap release_wrapper_lock EXIT INT TERM
}

devtools_reachable() {
  curl -fsS "$DEVTOOLS_VERSION_URL" >/dev/null 2>&1
}

cleanup_stale_profile_state() {
  mkdir -p "$ORACLE_PROFILE/Default"
  rm -f "$ORACLE_PROFILE/chrome.pid" \
        "$ORACLE_PROFILE/DevToolsActivePort" \
        "$ORACLE_PROFILE/Default/LOCK" \
        "$ORACLE_PROFILE/SingletonLock" \
        "$ORACLE_PROFILE/SingletonSocket" \
        "$ORACLE_PROFILE/SingletonCookie"
}

launch_browser_if_needed() {
  if devtools_reachable; then
    return
  fi

  cleanup_stale_profile_state
  local launch_flags=(
    "--user-data-dir=$ORACLE_PROFILE"
    "--remote-debugging-port=$ORACLE_DEBUG_PORT"
    "--no-first-run"
    "--disable-background-networking"
  )
  if [[ "$ORACLE_DISABLE_EXTENSIONS" == "1" ]]; then
    launch_flags+=(
      "--disable-extensions"
      "--disable-component-extensions-with-background-pages"
    )
  fi

  "$ORACLE_BROWSER_BIN" "${launch_flags[@]}" "$ORACLE_START_URL" >"$ORACLE_BROWSER_LOG" 2>&1 &

  local i
  for i in $(seq 1 "$ORACLE_DEVTOOLS_WAIT_SECONDS"); do
    if devtools_reachable; then
      return
    fi
    sleep 1
  done

  echo "error: DevTools endpoint is not reachable at $DEVTOOLS_VERSION_URL" >&2
  echo "hint: check browser launch log: $ORACLE_BROWSER_LOG" >&2
  exit 1
}

close_existing_chatgpt_tabs() {
  if [[ "$ORACLE_CLOSE_EXISTING_CHATGPT_TABS" != "1" ]]; then
    return
  fi
  if ! command -v jq >/dev/null 2>&1; then
    warn "jq not found; skipping ChatGPT tab cleanup."
    return
  fi
  local tabs_json tab_id
  tabs_json="$(curl -fsS "$DEVTOOLS_LIST_URL" 2>/dev/null || true)"
  if [[ -z "$tabs_json" ]]; then
    return
  fi
  while IFS= read -r tab_id; do
    [[ -n "$tab_id" ]] || continue
    curl -fsS "http://127.0.0.1:${ORACLE_DEBUG_PORT}/json/close/${tab_id}" >/dev/null 2>&1 || true
  done < <(printf '%s' "$tabs_json" | jq -r '.[] | select(.url | test("chatgpt\\.com"; "i")) | .id')
}

open_fresh_chatgpt_tab() {
  if [[ "$ORACLE_OPEN_CHATGPT_TAB" != "1" ]]; then
    return
  fi
  local encoded_url
  if command -v jq >/dev/null 2>&1; then
    encoded_url="$(printf '%s' "$ORACLE_START_URL" | jq -sRr @uri)"
  else
    encoded_url="$ORACLE_START_URL"
  fi
  curl -fsS "http://127.0.0.1:${ORACLE_DEBUG_PORT}/json/new?${encoded_url}" >/dev/null 2>&1 || true
}

resolve_cookie_db_path() {
  local candidates path
  candidates=(
    "$ORACLE_COOKIE_DB"
    "$ORACLE_PROFILE/Default/Network/Cookies"
    "$ORACLE_PROFILE/Default/Cookies"
    "$HOME/Library/Application Support/BraveSoftware/Brave-Browser/Default/Network/Cookies"
    "$HOME/Library/Application Support/BraveSoftware/Brave-Browser/Default/Cookies"
    "$HOME/Library/Application Support/Google/Chrome/Default/Network/Cookies"
    "$HOME/Library/Application Support/Google/Chrome/Default/Cookies"
  )
  for path in "${candidates[@]}"; do
    [[ -n "$path" ]] || continue
    [[ -f "$path" ]] || continue
    printf '%s\n' "$path"
    return 0
  done
  return 1
}

precheck_session_cookie() {
  if [[ "$ORACLE_CHECK_SESSION_COOKIE" != "1" ]]; then
    return
  fi
  if ! command -v sqlite3 >/dev/null 2>&1; then
    warn "sqlite3 not found; skipping ChatGPT session cookie precheck."
    return
  fi
  local cookie_db tmp_db token_count
  cookie_db="$(resolve_cookie_db_path || true)"
  if [[ -z "$cookie_db" ]]; then
    warn "cookie database not found; cannot precheck login session."
    if [[ "$ORACLE_FAIL_IF_NO_SESSION_COOKIE" == "1" ]]; then
      echo "error: no cookie database found and ORACLE_FAIL_IF_NO_SESSION_COOKIE=1" >&2
      exit 1
    fi
    return
  fi
  tmp_db="$(mktemp "/tmp/oracle-cookie-check.XXXXXX.sqlite")"
  cp "$cookie_db" "$tmp_db" 2>/dev/null || true
  token_count="$(sqlite3 "$tmp_db" "SELECT COUNT(*) FROM cookies WHERE host_key LIKE '%chatgpt.com%' AND (name='__Secure-next-auth.session-token' OR name LIKE '__Secure-next-auth.session-token.%');" 2>/dev/null || true)"
  rm -f "$tmp_db" >/dev/null 2>&1 || true

  if [[ -z "$token_count" ]]; then
    warn "session precheck query failed for cookie DB: $cookie_db"
    return
  fi
  if [[ "$token_count" == "0" ]]; then
    warn "no ChatGPT session cookie token found in cookie DB: $cookie_db"
    if [[ "$ORACLE_FAIL_IF_NO_SESSION_COOKIE" == "1" ]]; then
      echo "error: session cookie precheck failed and ORACLE_FAIL_IF_NO_SESSION_COOKIE=1" >&2
      exit 1
    fi
  fi
}

acquire_wrapper_lock
launch_browser_if_needed
close_existing_chatgpt_tabs
open_fresh_chatgpt_tab
precheck_session_cookie

oracle_cmd=(
  npx -y @steipete/oracle
  --engine browser
  --model gpt-5.2-pro
  --browser-model-strategy current
  --browser-port "$ORACLE_DEBUG_PORT"
  --browser-chrome-path "$ORACLE_BROWSER_BIN"
  --chatgpt-url "$ORACLE_START_URL"
)

if [[ "$ORACLE_FORCE_MANUAL_LOGIN" == "1" ]]; then
  oracle_cmd+=(
    --browser-manual-login
    --browser-reuse-wait "$ORACLE_BROWSER_REUSE_WAIT"
    --browser-profile-lock-timeout "$ORACLE_BROWSER_PROFILE_LOCK_TIMEOUT"
  )
fi

oracle_cmd+=("$@")
"${oracle_cmd[@]}"
