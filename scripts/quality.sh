#!/usr/bin/env bash
# Read-only source checks; generated results go under ignored artifacts/quality.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -n "${GENAI_PYTHON:-}" ]]; then
  python_bin="$GENAI_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  python_bin="$PWD/.venv/bin/python"
else
  python_bin=python
fi
mkdir -p artifacts/quality
run_dir=$(mktemp -d "artifacts/quality/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
run_status=0
finish() {
  local exit_status=$?
  if (( exit_status != 0 )); then run_status=$exit_status; fi
  printf '%s\n' "$run_status" > "$run_dir/exit-status.txt"
  printf 'Quality results: %s (exit %s)\n' "$run_dir" "$run_status"
}
trap finish EXIT
{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  git rev-parse HEAD
  "$python_bin" --version
} > "$run_dir/context.txt" 2>&1
run_check() {
  local name=$1
  shift
  printf 'Running %s\n' "$name"
  timeout --signal=TERM --kill-after=10s 300s "$@" 2>&1 | tee "$run_dir/$name.log"
  local statuses=("${PIPESTATUS[@]}")
  local status=${statuses[0]}
  if (( statuses[1] != 0 && status == 0 )); then status=${statuses[1]}; fi
  printf '%s\n' "$status" > "$run_dir/$name.exit-status.txt"
  if (( status != 0 )); then run_status=1; fi
}
run_check docs "$python_bin" scripts/check_docs.py
run_check tests "$python_bin" -m pytest -q --junitxml="$run_dir/pytest.xml"
run_check smoke-flow "$python_bin" -m genai_study.experiment --method flow --steps 20 --samples 64 --output "$run_dir/flow"
run_check smoke-ddpm "$python_bin" -m genai_study.experiment --method ddpm --steps 20 --samples 64 --output "$run_dir/ddpm"
run_check smoke-video "$python_bin" -m genai_study.video_diagnostics --output "$run_dir/video"
exit "$run_status"
