#!/usr/bin/env bash
# Adapter only: requires a separately configured/authenticated/budgeted agent CLI.
set -uo pipefail
if (( $# < 3 )) || [[ ! $1 =~ ^[1-9][0-9]*$ ]] || [[ $2 != -- ]]; then
  printf 'Usage: %s TIMEOUT_SECONDS -- EXECUTABLE [ARGUMENT ...]\n' "$0" >&2
  exit 64
fi
agent_timeout=$1
shift 2
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for required in flock timeout; do
  if ! command -v "$required" >/dev/null 2>&1; then
    printf 'Required command missing: %s\n' "$required" >&2
    exit 69
  fi
done
# Resolve Git's common directory: every entry point must use this same lock.
git_common=$(git rev-parse --path-format=absolute --git-common-dir) || exit 69
lock_file="$git_common/genai-agent.lock"
mkdir -p artifacts/agent-runs
run_dir=$(mktemp -d "artifacts/agent-runs/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  git rev-parse HEAD
  printf 'timeout_seconds=%s\n' "$agent_timeout"
  # Do not record argv: callers may inadvertently put credentials in arguments.
} > "$run_dir/context.txt" 2>&1
flock --exclusive --nonblock --conflict-exit-code 75 "$lock_file" \
  timeout --signal=TERM --kill-after=10s "$agent_timeout" "$@" \
  2>&1 | tee "$run_dir/output.log"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if (( statuses[1] != 0 && status == 0 )); then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$run_dir/exit-status.txt"
printf 'Agent adapter result: %s (exit %s)\n' "$run_dir" "$status"
exit "$status"
