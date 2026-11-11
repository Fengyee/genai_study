#!/usr/bin/env python3
"""Execute the committed eight-request SD1.4 attributes pilot, once and serially."""
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

import numpy as np
from PIL import Image


SOURCE_PATHS = {
    "examples/pretrained-sd14/manifest.json", "requirements-pretrained-lock.txt",
    "src/genai_study/pretrained_image.py", "src/genai_study/run_record.py",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"Missing or invalid {label}")
    return value


def committed_plan(path, root):
    relative = path.resolve().relative_to(root).as_posix()
    result = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=root,
                            capture_output=True, timeout=10, check=True)
    require(result.stdout == path.read_bytes(), "Plan differs from committed HEAD content")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, timeout=10, check=True).stdout.strip()


def check_sources(plan, root):
    identities = plan["source_identities"]
    require(set(identities) == SOURCE_PATHS, "Plan must fix the four source/lock identities")
    for name, identity in identities.items():
        require(sha256(root / name) == identity["sha256"], f"Source identity changed: {name}")


def check_plan_paths(plan, root, execution_path):
    requests = plan["requests"]
    require(len(requests) == 8, "This pilot requires exactly eight planned requests")
    require(len({r["request_id"] for r in requests}) == 8, "Duplicate request IDs")
    require(len({(r["prompt_id"], r["seed"]) for r in requests}) == 8,
            "Duplicate prompt/seed requests")
    paths = []
    for order, request in enumerate(requests, 1):
        require(request["order"] == order, "Request order differs from list order")
        for key in ("output", "record_output"):
            path = root / request[key]
            require(not os.path.lexists(path), f"Planned directory already exists: {path}")
            resolved = path.resolve()
            require(resolved.is_relative_to(root), "Output path escapes the repository")
            require(not execution_path.is_relative_to(resolved),
                    "Execution record must be outside each request directory")
            paths.append(resolved)
        require(not os.path.lexists(execution_path.parent / f"{request['request_id']}.recorder.log"),
                "Recorder stdout log already exists")
        # Confirm exact outer and inner arguments without running either command.
        from genai_study.run_record import make_parser as record_parser
        from genai_study.pretrained_image import make_parser as image_parser
        argv = request["argv"]
        require(argv[:3] == [".venv/bin/python", "-m", "genai_study.run_record"],
                "Unexpected outer recorder command")
        outer = record_parser().parse_args(argv[3:])
        require(outer.command[:1] == ["--"], "Missing command separator")
        child = outer.command[1:]
        require(child[:4] == [".venv-pretrained/bin/python", "-u", "-m",
                             "genai_study.pretrained_image"], "Unexpected inference command")
        args = image_parser().parse_args(child[4:])
        require(outer.timeout == plan["execution_budget"]["per_request_timeout_seconds"]
                and str(outer.output) == request["record_output"], "Recorder config differs from plan")
        require(args.command == "run" and str(args.output) == request["output"]
                and str(args.model_dir) == plan["model_directory"]
                and str(args.manifest) == "examples/pretrained-sd14/manifest.json"
                and args.prompt == plan["prompts"][request["prompt_id"]]["text"]
                and args.seed == request["seed"], "Inference command differs from request")
        for name in ("steps", "guidance_scale", "height", "width", "threads"):
            require(getattr(args, name) == plan["fixed"][name], f"Unexpected CLI {name}")
    require(len(set(paths)) == 16, "Output/recorder directories must be distinct")
    require(not any(a != b and a.is_relative_to(b) for a in paths for b in paths),
            "Request directories must not contain each other")


def normalized_scheduler(value):
    result = json.loads(json.dumps(value))
    defaults = result.get("config", {}).get("_use_default_values")
    if defaults is not None:
        require(isinstance(defaults, list), "Invalid scheduler default-name list")
        result["config"]["_use_default_values"] = sorted(defaults)
    return result


def validate_success(plan, request, root, returncode, noise_by_seed):
    record_dir, output = root / request["record_output"], root / request["output"]
    record = read_json(record_dir / "run.json")
    argv = request["argv"]
    child = argv[argv.index("--") + 1:]
    require(returncode == 0 and record["status"] == "succeeded"
            and record["exit_code"] == record["child_returncode"] == 0
            and record["timed_out"] is False, "Recorder did not complete successfully")
    require(record["argv"] == child, "Recorded child argv differs from plan")
    require(record["timeout_seconds"] == plan["execution_budget"]["per_request_timeout_seconds"],
            "Recorded timeout differs from plan")
    require(record["runner"]["source_sha256"]
            == plan["source_identities"]["src/genai_study/run_record.py"]["sha256"],
            "Recorded runner identity differs from plan")
    require(record["log_file"] == "command.log", "Unexpected recorder log path")
    log = record_dir / "command.log"
    require(log.stat().st_size == record["log_size_bytes"] and sha256(log) == record["log_sha256"],
            "Recorder log identity mismatch")
    wall = positive_number(record.get("wall_seconds"), "run wall_seconds")
    metrics = read_json(output / "metrics.json")
    expected_config = {key: plan["fixed"][key] for key in (
        "steps", "guidance_scale", "height", "width", "threads", "device", "dtype",
        "eta", "negative_prompt", "batch_size")}
    expected_config.update(prompt=plan["prompts"][request["prompt_id"]]["text"],
                           seed=request["seed"], deterministic_algorithms=True)
    require(metrics["config"] == expected_config, "Generated config differs from plan")
    versions = {**plan["dependency_preflight"]["versions"],
                "python": plan["dependency_preflight"]["python"]}
    require(metrics["versions"] == versions, "Generated software versions differ from plan")
    model = metrics["model"]
    require(model["model_id"] == plan["model_id"] and model["revision"] == plan["revision"]
            and model["manifest_path"] == "examples/pretrained-sd14/manifest.json"
            and model["manifest_sha256"] == plan["source_identities"][model["manifest_path"]]["sha256"],
            "Generated model identity differs from plan")
    for key in ("verified_files", "verified_bytes"):
        require(model[key] == plan["verified_cached_assets"][key], "Model verification scope changed")
    expected_scheduler = {key: plan["expected_scheduler"][key]
                          for key in ("class", "config", "timesteps", "init_noise_sigma")}
    require(normalized_scheduler(metrics["scheduler"]) == normalized_scheduler(expected_scheduler),
            "Actual scheduler differs from plan")
    evaluations = metrics["network_evaluations"]
    require(evaluations["unet_forward_calls"] == 20
            and evaluations["unet_batch_sizes"] == [2] * 20
            and evaluations["unet_sample_branch_evaluations"] == 40,
            "Actual UNet evaluations differ from the twenty-step CFG protocol")
    representation = metrics["representation"]
    require(representation["prompt_tokens_with_special_tokens"]
            == plan["prompts"][request["prompt_id"]]["tokens_with_special_tokens"]
            and representation["tokenizer_max_length"] == 77
            and representation["prompt_truncated"] is False, "Tokenizer contract mismatch")
    require(metrics["outputs"]["image"] == "image.png"
            and metrics["outputs"]["initial_noise"] == "initial_noise.npy", "Unexpected output names")
    image = output / "image.png"
    with Image.open(image) as decoded:
        require(decoded.format == "PNG" and decoded.size == (512, 512)
                and decoded.mode == "RGB", "PNG format/shape/mode mismatch")
        decoded.load()
    require(sha256(image) == metrics["outputs"]["image_sha256"], "PNG SHA256 mismatch")
    noise = np.load(output / "initial_noise.npy", allow_pickle=False)
    expected_noise = plan["expected_initial_noise"][str(request["seed"])]
    require(noise.dtype == np.dtype("float32") and noise.flags.c_contiguous
            and list(noise.shape) == expected_noise["shape"] and np.isfinite(noise).all(),
            "Initial-noise array contract mismatch")
    raw_sha = hashlib.sha256(noise.tobytes(order="C")).hexdigest()
    require(raw_sha == expected_noise["sha256"] == representation["initial_noise_sha256"]
            and representation["initial_noise_shape"] == expected_noise["shape"],
            "Initial-noise raw SHA/shape mismatch")
    for seed, saved in noise_by_seed.items():
        require(np.array_equal(noise, saved) == (seed == request["seed"]),
                "Initial-noise pairing differs from plan")
    noise_by_seed[request["seed"]] = noise.copy()
    safety = metrics["safety"]
    require(safety["checker_enabled"] is True and type(safety["nsfw_content_detected"]) is bool,
            "Missing or invalid safety-check result")
    rss = positive_number(metrics.get("memory", {}).get("process_peak_rss_kib"),
                          "model process_peak_rss_kib")
    return {"run_record_status": record["status"], "wall_seconds": wall,
            "process_peak_rss_kib": rss, "filtered": safety["nsfw_content_detected"],
            "image_sha256": metrics["outputs"]["image_sha256"],
            "initial_noise_sha256": raw_sha, "validation": "passed"}


def terminate_recorder(process):
    """The verified recorder handles TERM and cleans its separate child process group."""
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
        return True
    except subprocess.TimeoutExpired:
        # Do not SIGKILL the recorder and abandon its independent child group.
        return False


def execute(plan_path, execution_path):
    root = Path.cwd().resolve()
    plan_path, execution_path = Path(plan_path).resolve(), Path(execution_path).resolve()
    require(not os.path.lexists(execution_path)
            and not os.path.lexists(execution_path.with_name(execution_path.name + ".tmp")),
            "Execution record or its temporary file already exists")
    plan = read_json(plan_path)
    check_plan_paths(plan, root, execution_path)
    execution_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": 1, "campaign": plan["campaign"], "driver_pid": os.getpid(),
             "driver_path": str(Path(__file__).resolve()), "driver_sha256": sha256(__file__),
             "plan_path": str(plan_path), "plan_sha256": sha256(plan_path),
             "created_utc": utc_now(), "status": "preflight", "stop_reason": None,
             "active_request": None, "requests": [
                 {"order": r["order"], "request_id": r["request_id"], "argv": r["argv"],
                  "prompt_id": r["prompt_id"], "seed": r["seed"],
                  "output": r["output"], "record_output": r["record_output"],
                  "status": "not_attempted", "pid": None, "started_utc": None,
                  "finished_utc": None, "returncode": None} for r in plan["requests"]]}
    atomic_json(execution_path, state)
    cancellation = {"signal": None}
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(
            signum, lambda received, frame: cancellation.update(signal=received))
    started = None
    active = None
    current = None
    exit_code = 1
    try:
        state["plan_commit"] = committed_plan(plan_path, root)
        check_plan_paths(plan, root, execution_path)
        check_sources(plan, root)
        budget = positive_number(plan["execution_budget"]["campaign_wall_budget_seconds"], "campaign budget")
        minimum_remaining = positive_number(
            plan["execution_budget"]["start_next_only_with_remaining_seconds_at_least"], "start reserve")
        started = time.monotonic()
        deadline = started + budget
        # Leave time for the recorder's one-second child cleanup and final records.
        cleanup_reserve = min(5.0, budget / 10)
        state.update(status="running", started_utc=utc_now(), budget_seconds=budget)
        atomic_json(execution_path, state)
        noise_by_seed = {}
        for request, current in zip(plan["requests"], state["requests"]):
            if cancellation["signal"] is not None:
                state["stop_reason"] = f"signal_{cancellation['signal']}"
                break
            if deadline - time.monotonic() < minimum_remaining:
                state["stop_reason"] = "remaining_campaign_budget_below_start_reserve"
                break
            check_sources(plan, root)
            require(sha256(plan_path) == state["plan_sha256"], "Plan changed during campaign")
            for key in ("output", "record_output"):
                require(not os.path.lexists(root / request[key]), "Request path appeared before launch")
            log_path = execution_path.parent / f"{request['request_id']}.recorder.log"
            current.update(status="starting", started_utc=utc_now(), recorder_stdout_log=str(log_path))
            state["active_request"] = request["request_id"]
            atomic_json(execution_path, state)
            print(f"Starting {request['order']}/8: {request['request_id']}", flush=True)
            interrupted = None
            with log_path.open("xb") as log:
                active = subprocess.Popen(request["argv"], cwd=root, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT)
                current.update(status="running", pid=active.pid)
                atomic_json(execution_path, state)
                while active.poll() is None:
                    if cancellation["signal"] is not None:
                        interrupted = f"signal_{cancellation['signal']}"
                    elif time.monotonic() >= deadline - cleanup_reserve:
                        interrupted = "campaign_deadline_cleanup_reserve"
                    if interrupted:
                        current["cleanup_confirmed"] = terminate_recorder(active)
                        break
                    time.sleep(0.1)
            current.update(returncode=active.poll(), finished_utc=utc_now())
            if interrupted:
                current["status"] = "interrupted"
                state["stop_reason"] = interrupted
                if not current["cleanup_confirmed"]:
                    state["stop_reason"] += ":recorder_cleanup_unconfirmed"
                atomic_json(execution_path, state)
                break
            active = None
            record_path = root / request["record_output"] / "run.json"
            if record_path.is_file():
                recorded = read_json(record_path)
                current["run_record_status"] = recorded.get("status")
                current["run_record_exit_code"] = recorded.get("exit_code")
            check_sources(plan, root)
            validation = validate_success(plan, request, root, current["returncode"], noise_by_seed)
            current.update(validation)
            current["status"] = "filtered" if validation["filtered"] else "succeeded"
            state["active_request"] = None
            if request["order"] == 1:
                gate = plan["first_request_gate"]["successful_resource_gate"]
                passed = (validation["wall_seconds"] <= gate["maximum_run_record_wall_seconds"]
                          and validation["process_peak_rss_kib"] <= gate["maximum_model_process_peak_rss_kib"])
                state["first_request_gate"] = {"passed": passed, **validation}
                if not passed:
                    state["stop_reason"] = "first_request_resource_gate"
            atomic_json(execution_path, state)
            print(f"{request['request_id']}: {current['status']}", flush=True)
            if state["stop_reason"]:
                break
        if state["stop_reason"] is None:
            state["status"], exit_code = "completed", 0
        else:
            state["status"] = "stopped"
            exit_code = 128 + cancellation["signal"] if cancellation["signal"] else 1
    except (Exception, SystemExit) as error:
        if active is not None and active.poll() is None:
            cleanup = terminate_recorder(active)
            if current is not None:
                current["cleanup_confirmed"] = cleanup
        if current is not None and current["status"] in ("starting", "running"):
            recorded_status = current.get("run_record_status")
            failure_status = (recorded_status if recorded_status in
                              ("failed", "launch_failed", "timed_out", "interrupted")
                              else "technical_failure")
            current.update(status=failure_status, finished_utc=utc_now(),
                           returncode=active.poll() if active is not None else current["returncode"],
                           error=f"{type(error).__name__}: {error}")
        state.update(status="stopped", stop_reason=f"technical_or_shared_failure:{type(error).__name__}: {error}")
    finally:
        state["finished_utc"] = utc_now()
        state["campaign_wall_seconds"] = time.monotonic() - started if started is not None else None
        for request in state["requests"]:
            if request["status"] == "not_attempted":
                request["not_attempted_reason"] = state["stop_reason"]
        if active is None or active.poll() is not None:
            state["active_request"] = None
        atomic_json(execution_path, state)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    args = parser.parse_args()
    try:
        return execute(args.plan, args.execution)
    except (OSError, ValueError) as error:
        parser.exit(2, f"pilot preflight failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
