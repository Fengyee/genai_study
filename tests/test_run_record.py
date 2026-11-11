"""Exercise process outcomes and durable records without model dependencies."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def launch(output, command, timeout=5, **kwargs):
    return subprocess.run(
        [sys.executable, "-m", "genai_study.run_record", "--timeout", str(timeout),
         "--output", str(output), "--", *command],
        capture_output=True, text=True, timeout=15, **kwargs,
    )


def read_record(output):
    record = json.loads((output / "run.json").read_text())
    log = (output / "command.log").read_bytes()
    assert record["log_sha256"] == hashlib.sha256(log).hexdigest()
    assert record["log_size_bytes"] == len(log)
    assert record["wall_seconds"] >= 0
    assert record["finished_utc"] >= record["started_utc"]
    return record, log


def test_success_captures_both_streams_and_literal_argv(tmp_path):
    literal = "spaces; $(printf shell-must-not-run) `echo literal`"
    command = [sys.executable, "-u", "-c",
               "import sys; print(sys.argv[1]); print('stderr', file=sys.stderr)", literal]
    result = launch(tmp_path / "record", command)
    assert result.returncode == 0, result.stderr
    record, log = read_record(tmp_path / "record")
    assert record["status"] == "succeeded"
    assert record["child_returncode"] == record["exit_code"] == 0
    assert record["argv"] == command
    assert record["cwd"] == str(Path.cwd())
    assert not record["timed_out"]
    assert log.decode() == literal + "\nstderr\n"
    assert len(record["runner"]["source_sha256"]) == 64
    assert record["git"]["head"]


def test_nonzero_command_records_original_failure(tmp_path):
    result = launch(tmp_path / "record", [sys.executable, "-c", "raise SystemExit(2)"])
    assert result.returncode == 2
    record, _ = read_record(tmp_path / "record")
    assert record["status"] == "failed"
    assert record["child_returncode"] == record["exit_code"] == 2
    assert not record["timed_out"]


def test_launch_failure_and_absent_git_are_recorded(tmp_path):
    result = launch(tmp_path / "record", [str(tmp_path / "absent")], cwd=tmp_path,
                    env={**os.environ, "PATH": str(tmp_path)})
    assert result.returncode == 127
    record, log = read_record(tmp_path / "record")
    assert record["status"] == "launch_failed"
    assert record["child_returncode"] is None
    assert record["git"] is None
    assert "No such file" in record["error"]
    assert log == b""


def test_existing_record_is_preserved_and_command_never_starts(tmp_path):
    output = tmp_path / "record"
    output.mkdir()
    (output / "run.json").write_bytes(b"existing experiment")
    marker = tmp_path / "side-effect"
    result = launch(output, [sys.executable, "-c",
                            "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()", str(marker)])
    assert result.returncode == 2
    assert not marker.exists()
    assert (output / "run.json").read_bytes() == b"existing experiment"
    assert not (output / "command.log").exists()


def process_is_running(pid):
    try:
        status = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return status != "Z"


def test_timeout_kills_term_resistant_descendant_and_preserves_output(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    descendant = ("import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                  f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)")
    command = [sys.executable, "-u", "-c",
               "import subprocess,sys,time; print('before timeout',flush=True); "
               f"subprocess.Popen([sys.executable,'-c',{descendant!r}]); time.sleep(60)"]
    result = launch(tmp_path / "record", command, timeout=0.5)
    assert result.returncode == 124
    record, log = read_record(tmp_path / "record")
    assert record["status"] == "timed_out" and record["timed_out"]
    assert record["child_returncode"] == -signal.SIGTERM
    assert 0.5 <= record["wall_seconds"] < 5
    assert b"before timeout" in log
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2
    while process_is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not process_is_running(pid)


@pytest.mark.parametrize("interrupt_signal", [signal.SIGINT, signal.SIGTERM])
def test_interruption_records_status_and_cleans_up(tmp_path, interrupt_signal):
    output = tmp_path / "record"
    child_pid = tmp_path / "child.pid"
    runner = subprocess.Popen(
        [sys.executable, "-m", "genai_study.run_record", "--timeout", "10", "--output", str(output),
         "--", sys.executable, "-u", "-c",
         "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)",
         str(child_pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_pid.exists()
        runner.send_signal(interrupt_signal)
        stdout, stderr = runner.communicate(timeout=5)
        assert runner.returncode == 128 + interrupt_signal, (stdout, stderr)
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()
    record, _ = read_record(output)
    assert record["status"] == "interrupted"
    assert record["signal"] == interrupt_signal
    assert record["child_returncode"] == -signal.SIGTERM
    assert not process_is_running(int(child_pid.read_text()))


@pytest.mark.parametrize("timeout,command", [("0", ["unused"]), ("-1", ["unused"]),
                                           ("nan", ["unused"]), ("inf", ["unused"]), ("1", [])])
def test_invalid_arguments_fail_before_creating_record(tmp_path, timeout, command):
    output = tmp_path / "record"
    result = launch(output, command, timeout=timeout)
    assert result.returncode == 2
    assert not output.exists()
