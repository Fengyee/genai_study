# Maintainer instructions

Read README.md, docs/learning-path.md, docs/notation.md, records/PROGRESS.md and records/BACKLOG.md before changing content. Preserve user edits. This cloud task is already isolated: use the existing checkout; create no worktree unless explicitly requested.

Work in reviewable rounds: identify one reader problem, collect primary evidence, implement, validate, update records, commit on a feature branch, submit through a PR. Never force push. Do not represent an unmerged workflow as deployed or a planned chapter as complete.

Use Chinese explanations with English technical names at first occurrence. State capabilities, results and applicability directly. Avoid repetitive defensive contrasts such as "X, not Y"; express necessary assumptions and limitations as specific facts, and keep mathematical counterexamples when they serve the lesson. Every chapter needs prerequisites, learning goals, scope, exercises with hints/answers, and primary references. Distinguish measured results, mathematical claims, and hypotheses. Use docs/notation.md conventions; changes to shared notation require cross-chapter review.

Keep examples cheap and CPU runnable. For algorithm changes use analytic properties, independent derivations and meaningful numerical tests. Run `bash scripts/quality.sh` for shared code/dependency/convention changes. For documentation changes run `.venv/bin/python scripts/check_docs.py` plus relevant algorithm tests. Do not weaken assertions to clear checks. Record the exact command, outcome and limits; no fake benchmark numbers.

Do not execute paid model API calls, introduce secrets into files, or claim production image/video capability from toy distributions. Report credentials by presence only. Dependencies must retain TLS, hashes and artifact verification.

Continue highest-priority independent backlog work when an external operation is blocked. Save unfinished work and concrete recovery steps in records/PROGRESS.md. Automated writers must acquire the workspace lock documented in docs/automation.md. Scheduling, pushing a branch, creating a PR, and publishing an environment are separate states to report honestly.
