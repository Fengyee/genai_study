"""Record one foreground command and its timeout on Linux, using only stdlib.

``wall_seconds`` starts immediately before Popen and ends after child cleanup.
It includes command startup, imports, execution and termination grace, while
excluding this recorder's startup and initial metadata collection. SIGKILL,
host shutdown or a full filesystem can leave the last record as ``running``.
The command inherits the environment, receives closed stdin, and writes both
output streams to command.log. No shell interprets its arguments.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


RUNNER_VERSION = 1
TERMINATION_GRACE_SECONDS = 1.0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_identity(cwd):
    """Collect revision and changed paths; unavailable Git is non-fatal."""
    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(cwd), *arguments], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=3,
        ).stdout

    try:
        root = git("rev-parse", "--show-toplevel").strip()
        commit = git("rev-parse", "HEAD").strip()
        entries = iter(git("status", "--porcelain=v1", "-z", "--untracked-files=all").split("\0"))
        changed = []
        for entry in entries:
            if not entry:
                continue
            item = {"status": entry[:2], "path": entry[3:]}
            if "R" in entry[:2] or "C" in entry[:2]:
                item["original_path"] = next(entries)
            changed.append(item)
        return {"root": root, "head": commit, "changed_paths": changed}
    except (OSError, subprocess.SubprocessError, StopIteration, UnicodeError):
        return None


def write_record(path, record):
    """Replace JSON atomically, retaining the preceding complete record."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def signal_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def stop_group(process):
    """Terminate the whole command group, including TERM-resistant children."""
    signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()  # Reap the direct child even when descendants remain.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    signal_group(process, signal.SIGKILL)
    process.wait()


class RunInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


def record_command(command, output, timeout):
    """Run argv once; callers must supply a fresh output directory."""
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    path = output / "run.json"
    log_path = output / "command.log"
    cwd = Path.cwd()
    record = {
        "schema_version": 1,
        "status": "running",
        "argv": command,
        "cwd": str(cwd),
        "started_utc": utc_now(),
        "timeout_seconds": timeout,
        "termination_grace_seconds": TERMINATION_GRACE_SECONDS,
        "runner": {
            "version": RUNNER_VERSION,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
        },
        "git": git_identity(cwd),
        "stdin": "devnull",
        "log_file": "command.log",
        "timed_out": False,
        "child_returncode": None,
        "exit_code": None,
    }
    write_record(path, record)
    print(f"Run record: {path}", flush=True)
    process = None
    previous_handlers = {}

    def interrupted(signum, frame):
        raise RunInterrupted(signum)

    with log_path.open("wb") as log:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
                record["status"] = "succeeded" if returncode == 0 else "failed"
                record["exit_code"] = returncode if returncode >= 0 else 128 - returncode
            except subprocess.TimeoutExpired:
                record.update(status="timed_out", timed_out=True, exit_code=124)
                stop_group(process)
        except RunInterrupted as interruption:
            # Ignore further terminal signals while cleaning up this command.
            for signum in previous_handlers:
                signal.signal(signum, signal.SIG_IGN)
            record.update(status="interrupted", signal=interruption.signum,
                          exit_code=128 + interruption.signum)
            if process is not None:
                stop_group(process)
        except OSError as error:
            record.update(status="launch_failed", error=str(error),
                          exit_code=127 if isinstance(error, FileNotFoundError) else 126)
        finally:
            if process is not None:
                if process.poll() is None:
                    stop_group(process)
                record["child_returncode"] = process.returncode
            record["wall_seconds"] = time.monotonic() - started
            record["finished_utc"] = utc_now()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    record["log_size_bytes"] = log_path.stat().st_size
    with log_path.open("rb") as log:
        record["log_sha256"] = hashlib.file_digest(log, "sha256").hexdigest()
    write_record(path, record)
    print(f"{record['status']}: exit {record['exit_code']}, "
          f"wall {record['wall_seconds']:.3f}s; log: {log_path}", flush=True)
    return record["exit_code"]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, required=True,
                        help="positive wall-time limit for the child command, in seconds")
    parser.add_argument("--output", type=Path, required=True,
                        help="new record directory, separate from the command's result directory")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="command argv after --; use python -u for live Python logs")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("this recorder requires POSIX process groups (validated on Linux)")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not command[0]:
        parser.error("provide a command after --")
    try:
        return record_command(command, args.output, args.timeout)
    except FileExistsError:
        parser.error(f"record path already exists: {args.output}; use a new --output")
    except OSError as error:
        parser.exit(1, f"could not save run record: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
