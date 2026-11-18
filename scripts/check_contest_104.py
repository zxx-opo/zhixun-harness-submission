"""Exercise the unmodified SDK 1.0.4 entry against a local host and real PyTRiO."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import tempfile
from time import monotonic

import numpy as np

from contest_engine.model import CLASSES, EXTENSION, NATIVE, TASKS, load, sha256_file
from harness.data import atomic_json, read_json
from harness.storage import exclusive
from scripts.check_contest_submission import ARGV, PROTOCOL, REF, ROOT, fail_on_timeout, manifest, send, stable, validation_metrics
from scripts.contest_live_gateway import LiveGateway, MODEL


IMAGE = "zhixun-harness:contest-1.0.4"
OUT = ROOT / "reports/contest-submission-1.0.4"
LIMIT = 64 * 1024 * 1024
CHUNK = 32 * 1024
WALL = 240
PROMPT = ("Local reproduction of the host task contract. Use only the permitted training projection and "
          "actual measurements. Continue the mounted parent. Select one valid artifact and cite this round's model call.")


def docker(image, name, mounts=(), env=None):
    cmd = ["docker", "run", "--rm", "-i", "--network=none", "--read-only", "--cpus=2", "--memory=4g",
           "--memory-swap=4g", "--pids-limit=256", "--tmpfs", "/tmp:rw,noexec,size=64m,mode=1777"]
    for source, dest, ro in mounts:
        args = f"type=bind,src={source.resolve()},dst={dest}"
        cmd.extend(("--mount", args + (",readonly" if ro else "")))
    for key, value in (env or {}).items():
        cmd.extend(("-e", f"{key}={value}"))
    return [*cmd, image, name]


class Uploads:
    def __init__(self, work):
        self.work = work
        self.records = {}
        self.bytes = {"candidate": 0, "state": 0}
        self.chunks = 0

    def begin(self, msg):
        identity = str(len(self.records))
        kind, files = msg["kind"], msg["files"]
        if kind not in self.bytes or not 1 <= len(files) <= 2:
            raise ValueError("unexpected upload declaration")
        expected = {}
        for entry in files:
            path, size = entry["path"], entry["size_bytes"]
            valid = (NATIVE, EXTENSION) if kind == "candidate" else ("memory.json",)
            if path not in valid or path in expected or type(size) is not int or not 0 < size <= LIMIT:
                raise ValueError("unexpected upload file")
            expected[path] = entry
        total = sum(entry["size_bytes"] for entry in files)
        if total > LIMIT or self.bytes[kind] + total > LIMIT * (2 if kind == "candidate" else 1):
            raise ValueError("upload exceeds contest artifact limit")
        self.bytes[kind] += total
        folder = self.work / ("upload-" + identity)
        folder.mkdir(mode=0o755)
        self.records[identity] = {"kind": kind, "files": expected, "folder": folder,
                                  "complete": False, "used": False}
        return {"type": "upload_begin_response", "upload_id": identity}

    def chunk(self, msg):
        entry = self.records[msg["upload_id"]]
        file = entry["files"][msg["path"]]
        path = entry["folder"] / file["path"]
        current = path.stat().st_size if path.is_file() else 0
        data = base64.b64decode(msg["data_base64"], validate=True)
        if entry["complete"] or current != msg["offset"] or not 0 < len(data) <= CHUNK or current + len(data) > file["size_bytes"]:
            raise ValueError("invalid upload chunk offset, size or sequence")
        with path.open("ab") as stream:
            stream.write(data)
        self.chunks += 1
        return {"type": "upload_chunk_response", "offset": current + len(data)}

    def finish(self, msg):
        identity = msg["upload_id"]
        entry = self.records[identity]
        if entry["complete"]:
            raise ValueError("upload was already finished")
        for file in entry["files"].values():
            path = entry["folder"] / file["path"]
            if path.stat().st_size != file["size_bytes"] or sha256_file(path) != file["sha256"]:
                raise ValueError("upload differs from its declaration")
        entry["complete"] = True
        return {"type": "upload_finish_response", "upload_id": identity}

    def consume(self, msg, kind):
        entry = self.records[msg["upload_id"]]
        if entry["kind"] != kind or not entry["complete"] or entry["used"]:
            raise ValueError("incomplete or reused upload")
        entry["used"] = True
        return entry["folder"], set(entry["files"])


class PublicScorer:
    def __init__(self, image, dataset, validation):
        self.image, self.dataset, self.task = image, dataset, TASKS[dataset]
        self.labels, lines = [], []
        self.cache = {}
        self.forward_passes = 0
        with validation.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                lines.append(stable({"task": self.task, "inputs": row["inputs"]}) + b"\n")
                self.labels.append(CLASSES.index(row["answer"]) if dataset == "A" else row["answer"])
        self.labels, self.first, self.more = np.asarray(self.labels), lines[0], b"".join(lines[1:])

    def measure(self, folder, artifact, *, force=False):
        key = tuple(item["sha256"] for item in artifact["files"])
        if key in self.cache and not force:
            return self.cache[key], True
        context = {"protocol": "contest.predict-context.v1", "task": self.task, "dataset_key": self.dataset,
                   "artifact": artifact["artifact"], "artifact_id": artifact["artifact_id"]}
        env = {"CONTEST_PREDICTION_CONTEXT": stable(context).decode(), "CONTEST_ARTIFACT_PATH": f"/model/{NATIVE}",
               "CONTEST_TASK": self.task, "CONTEST_DATASET_KEY": self.dataset}
        command = docker(self.image, "predict", [(folder, "/model", True)], env)
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            try:
                proc.stdin.write(self.first)
                proc.stdin.flush()
                if not select.select([proc.stdout], [], [], 40)[0]:
                    raise TimeoutError("prediction did not reply before EOF")
                first = proc.stdout.readline()
                rest, stderr = proc.communicate(self.more, timeout=100)
                if not first or proc.returncode:
                    raise RuntimeError("prediction process failed: " + stderr.decode(errors="replace")[-1000:])
            finally:
                if proc.poll() is None:
                    proc.kill()
        outputs = [json.loads(line)["outputs"] for line in (first + rest).splitlines()]
        if len(outputs) != len(self.labels):
            raise ValueError("prediction row count mismatch")
        if self.dataset == "A":
            scores = np.asarray([[{item["label"]: item["score"] for item in row}[label] for label in CLASSES] for row in outputs])
        else:
            scores = np.asarray([row["value"] for row in outputs])
        result = validation_metrics(self.dataset, self.labels, scores)
        if self.dataset == "A":
            predictions = scores.argmax(axis=1)
            result["acetaldehyde_tp"] = int(np.sum((self.labels == 3) & (predictions == 3)))
            result["acetaldehyde_n"] = int(np.sum(self.labels == 3))
        self.cache[key] = result
        self.forward_passes += 1
        return result, False


def probe_measurement(folder, artifact, parent_folder, parent, msg, dataset):
    model = load(folder / NATIVE, folder / EXTENSION if (folder / EXTENSION).exists() else None, dataset).native
    previous = load(parent_folder / NATIVE, dataset=dataset).native
    for field in ("target_mean", "target_scale", "paths", "seed", "learning_rate"):
        if model.meta[field] != previous.meta[field]:
            raise ValueError("candidate altered inherited metadata: " + field)
    np.testing.assert_array_equal(model.mean, previous.mean)
    np.testing.assert_array_equal(model.scale, previous.scale)
    steps = msg["configuration"].get("recipe", {}).get("steps", 0)
    if steps:
        if model.meta["steps"] - previous.meta["steps"] != steps:
            raise ValueError("candidate Adam step ancestry mismatch")
    elif artifact["files"][0]["sha256"] != parent["files"][0]["sha256"]:
        raise ValueError("frozen native weights or Adam state changed")


def fixture_reply(frame, msg):
    facts = json.loads(msg["messages"][-1]["content"])
    return {"call_id": f"fixture-{frame['dataset_key']}-{frame['round_index']}",
            "request": {"facts": facts}, "response": {"choices": [{"message": {"content": "{invalid fixture"}}]}}


def run_round(image, frame, mounts, work, scorer, bound_parent, history, gateway):
    dataset, tick = frame["dataset_key"], frame["round_index"]
    parent_folder, parent = bound_parent
    uploads = Uploads(work)
    artifacts, measurements, trace = {}, [], []
    receipt = state = state_sha = None
    sequence = 0
    scratch = work / "workspace"
    scratch.mkdir(mode=0o777)
    scratch.chmod(0o777)
    command = docker(image, "train", [*mounts, (scratch, "/workspace", False)])
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr, subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, encoding="utf-8"
    ) as proc:
        try:
            signal.alarm(WALL)
            send(proc, frame)
            for line in proc.stdout:
                msg = json.loads(line)
                kind = msg["type"]
                trace.append(kind)
                if kind == "round_failed":
                    raise RuntimeError(msg["error"])
                if kind == "round_complete":
                    record = msg["record"]
                    if (not receipt or not state or record["model_call_ids"] != [receipt["call_id"]]
                            or record["history_ids"] != [item["id"] for item in history]):
                        raise ValueError("model, state or history evidence is missing")
                    selected = record["selected_artifact_id"]
                    config = record["configuration"]
                    if config["measurement_ids"] != [item["artifact_id"] for item in measurements]:
                        raise ValueError("measured and cited artifacts differ")
                    if selected == parent["artifact_id"]:
                        selection = bound_parent, measurements[0]["metrics"] if measurements else None
                    elif len(measurements) == 2 and selected == measurements[1]["artifact_id"] and config["eligible_child"]:
                        selection = artifacts[selected], measurements[1]["metrics"]
                    else:
                        raise ValueError("final selection did not resolve to parent or eligible child")
                    if gateway is None and (selected != parent["artifact_id"] or config["proposal"] is not None):
                        raise ValueError("invalid reply did not retain the parent")
                    send(proc, {"type": "round_ack", "accepted": True})
                    break
                if msg["id"] != sequence:
                    raise ValueError("out-of-order SDK frame")
                sequence += 1
                if kind == "history_request":
                    if sequence != 1:
                        raise ValueError("history was not the first RPC")
                    reply = {"type": "history_response", "records": history}
                elif kind == "upload_begin_request":
                    reply = uploads.begin(msg)
                elif kind == "upload_chunk_request":
                    reply = uploads.chunk(msg)
                elif kind == "upload_finish_request":
                    reply = uploads.finish(msg)
                elif kind == "candidate_request":
                    if (receipt or len(measurements) >= 2 or msg["parent_artifact_id"] != parent["artifact_id"]
                            or msg["parent_checkpoint_sha256"] != parent["files"][0]["sha256"]):
                        raise ValueError("candidate has wrong parent or order")
                    folder, files = uploads.consume(msg, "candidate")
                    spec = msg["artifact"]
                    if (spec["predict_argv"] != ARGV or spec["code_paths"] or set(spec["checkpoint_paths"]) != files
                            or spec["checkpoint_paths"][0] != NATIVE):
                        raise ValueError("candidate manifest does not match checkpoint files or predict command")
                    identity = f"local-{dataset}-{tick}-m{len(measurements) + 1}"
                    artifact = manifest(folder, identity, spec["checkpoint_paths"])
                    probe_measurement(folder, artifact, parent_folder, parent, msg, dataset)
                    if not measurements and (msg["configuration"].get("role") != "paired_parent" or artifact["files"] != parent["files"]):
                        raise ValueError("paired parent does not match the mounted artifact")
                    metrics, cached = scorer.measure(folder, artifact)
                    artifacts[identity] = (folder, artifact)
                    measurements.append({"artifact_id": identity, "metrics": metrics, "cached_identical_artifact": cached,
                                         "artifact": artifact, "declaration": msg["configuration"]})
                    reply = {"type": "candidate_response", "artifact_id": identity, "summary": {"metrics": metrics}}
                elif kind == "model_request":
                    if receipt or len(measurements) not in (0, 2):
                        raise ValueError("model request did not follow a complete paired measurement")
                    facts = json.loads(msg["messages"][-1]["content"])
                    metric = "macro_f1" if dataset == "A" else "rmse"
                    if measurements and (facts["paired_parent"][metric] != measurements[0]["metrics"][metric]
                                         or facts["child"][metric] != measurements[1]["metrics"][metric]):
                        raise ValueError("model request omitted the exact paired metric")
                    receipt = gateway.model(frame, msg) if gateway else fixture_reply(frame, msg)
                    reply = {"type": "model_response", "call_id": receipt["call_id"], "response": receipt["response"]}
                elif kind == "state_request":
                    if not receipt or state or msg["parent_state_sha256"] != (frame["state_sha256"] or ""):
                        raise ValueError("state save lost model evidence or state ancestry")
                    state, _ = uploads.consume(msg, "state")
                    memory = read_json(state / "memory.json")
                    if memory["dataset"] != dataset or memory["round"] != tick:
                        raise ValueError("saved state has incorrect round identity")
                    state_sha = sha256_file(state / "memory.json")
                    reply = {"type": "state_response", "state_sha256": state_sha}
                else:
                    raise ValueError("unexpected SDK frame " + kind)
                send(proc, {**reply, "id": msg["id"]})
            else:
                raise RuntimeError("participant exited before round_complete")
            proc.stdin.close()
            if proc.wait(timeout=15) or proc.stdout.read():
                raise RuntimeError("participant did not stop after round_ack")
        except BaseException:
            stderr.seek(0)
            extra = stderr.read()[-1600:]
            if extra:
                print(extra, flush=True)
            if proc.poll() is None:
                proc.kill()
            raise
        finally:
            signal.alarm(0)
    evidence = {**record, "id": f"history-{dataset}-{tick}", "trace": trace, "measurements": measurements,
                "new_state_sha256": state_sha, "parent_state_sha256": frame["state_sha256"], "upload_chunks": uploads.chunks,
                "bytes_uploaded": uploads.bytes, "qwen": receipt}
    return evidence, selection, state


def task_run(args, dataset, run_dir, gateway):
    official = REF / dataset
    public = official / "public"
    report = run_dir / dataset
    report.mkdir(parents=True, exist_ok=True)
    start = monotonic()
    hashes = {"parent": sha256_file(official / NATIVE), "train": sha256_file(public / "train.jsonl"),
              "validation": sha256_file(public / "validation.jsonl")}
    scorer = PublicScorer(args.image, dataset, public / "validation.jsonl")
    parent_folder = report / "baseline"
    parent_folder.mkdir(exist_ok=True)
    shutil.copyfile(official / NATIVE, parent_folder / NATIVE)
    parent = manifest(parent_folder, f"baseline-{dataset}", [NATIVE])
    baseline, _ = scorer.measure(parent_folder, parent)
    metric = "macro_f1" if dataset == "A" else "rmse"
    score = baseline
    state_dir = state_sha = None
    history, trace, inventory = [], [], []
    for tick in range(1, args.rounds + 1):
        begin = monotonic()
        step = report / f"{tick:02d}"
        receipt_path = step / "round.json"
        frame = {"protocol": PROTOCOL, "prompt": PROMPT, "prompt_version": "industrial-workspace-flexible-v3",
                 "run_id": args.run_id, "dataset_key": dataset, "round_index": tick, "stage_id": "public-replay",
                 "task_contract": {"task": TASKS[dataset], "metric": metric, "public_projection": {
                     "train_path": "/data/train.jsonl", "validation_path": "/data/validation.jsonl"}},
                 "model_name": MODEL, "max_output_tokens": 224, "wall_seconds": WALL,
                 "parent_checkpoint_sha256": parent["files"][0]["sha256"], "parent_artifact": parent,
                 "resource_limits": {"artifact_upload_protocol": "chunked-v1", "artifact_bytes": LIMIT,
                                     "state_bytes": LIMIT, "upload_chunk_bytes": CHUNK,
                                     "candidate_upload_bytes_per_round": 2 * LIMIT, "state_upload_bytes_per_round": LIMIT},
                 "artifacts": inventory, "state_path": "/state" if state_dir else None, "state_sha256": state_sha}
        mounts = [(public, "/data", True), (parent_folder, "/previous", True)]
        if state_dir:
            mounts.append((state_dir, "/state", True))
        step.mkdir(exist_ok=True)
        received = step / "selected"
        next_state = step / "state"
        if receipt_path.is_file():
            record = read_json(receipt_path)
            if record["configuration"]["parent_artifact_id"] != parent["artifact_id"] or record["parent_state_sha256"] != state_sha:
                raise ValueError("restored run does not match the mounted round lineage")
        else:
            with tempfile.TemporaryDirectory(prefix="contest-104-round-") as temporary:
                disk = Path(temporary)
                disk.chmod(0o755)
                record, ((folder, chosen), measured), updated_state = run_round(
                    args.image, frame, mounts, disk, scorer, (parent_folder, parent), history, gateway)
                received.mkdir(exist_ok=True)
                for spec in chosen["files"]:
                    shutil.copyfile(folder / spec["path"], received / spec["path"])
                next_state.mkdir(exist_ok=True)
                shutil.copyfile(updated_state / "memory.json", next_state / "memory.json")
                record.update(selected=chosen, selected_metrics=measured or score, wall_seconds=monotonic() - begin)
                atomic_json(receipt_path, record)
        if (manifest(received, record["selected_artifact_id"], record["selected"]["artifact"]["checkpoint_paths"]) != record["selected"]
                or sha256_file(next_state / "memory.json") != record["new_state_sha256"]):
            raise ValueError("committed selected artifact or state failed verification")
        parent_folder, parent, score = received, record["selected"], record["selected_metrics"]
        scorer.cache[tuple(item["sha256"] for item in parent["files"])] = score
        state_dir, state_sha = next_state, record["new_state_sha256"]
        history.append({k: record[k] for k in ("id", "configuration", "reason", "seed", "selected_artifact_id")})
        inventory.extend(item["artifact"] for item in record["measurements"])
        trace.append(record)
        print(f"{dataset} {tick}/{args.rounds} {record['configuration']['selection']} "
              f"{metric}={score[metric]:.6f} paired={len(record['measurements']) == 2}", flush=True)
    verified, _ = scorer.measure(parent_folder, parent, force=True)
    if verified != score:
        raise ValueError("selected artifact differs after reload")
    return {"baseline": baseline, "final": verified, "final_artifact": parent, "rounds": trace, "hashes": hashes,
            "validation_rows": len(scorer.labels), "accepted": sum(r["configuration"]["selection"] == "accepted" for r in trace),
            "candidate_measurements": sum(len(r["measurements"]) for r in trace),
            "real_local_candidate_fits": sum(len(r["measurements"]) == 2 for r in trace),
            "new_full_prediction_passes": scorer.forward_passes,
            "elapsed_seconds": monotonic() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "fixture"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tasks", default="ABC")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--image", default=IMAGE)
    args = parser.parse_args()
    if (not re.fullmatch(r"contest-104-[a-z0-9-]{1,48}", args.run_id) or not 1 <= args.rounds <= 10
            or not args.tasks or len(set(args.tasks)) != len(args.tasks) or not set(args.tasks) <= set(TASKS)):
        parser.error("invalid run identity, tasks or rounds")
    signal.signal(signal.SIGALRM, fail_on_timeout)
    inspect = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image], text=True))[0]
    desc = json.loads(subprocess.check_output(docker(args.image, "describe"), text=True))
    if (desc["protocol"] != PROTOCOL or desc["model_access"] != "host_gateway_rpc" or desc["sdk_version"] != "1.0.4"
            or set(desc["tasks"]) != {"classification", "regression"} or (inspect["Os"], inspect["Architecture"]) != ("linux", "amd64")):
        raise ValueError("image has the wrong platform or SDK declaration")
    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"image_id": inspect["Id"], "mode": args.mode, "run_id": args.run_id, "tasks": args.tasks, "rounds": args.rounds,
                "local_host_sha256": sha256_file(Path(__file__)), "model": MODEL, "memory_gib": 4, "cpus": 2}
    with exclusive(run_dir / ".run.lock"):
        binding = run_dir / "run.json"
        if binding.exists() and read_json(binding) != identity:
            raise ValueError("run settings or image changed; use a new recorded run identity")
        atomic_json(binding, identity)
        gateway = LiveGateway(run_dir / "requests", ROOT / "runs/v12-trainer", args.run_id) if args.mode == "live" else None
        runs = {dataset: task_run(args, dataset, run_dir, gateway) for dataset in args.tasks}
        live_calls = [round_ for run in runs.values() for round_ in run["rounds"] if gateway]
        if len({row["model_call_ids"][0] for row in live_calls}) != len(live_calls):
            raise ValueError("provider IDs are not distinct across rounds")
        result = {**identity, "passed": True, "describe": desc, "runs": runs, "network": "none",
                  "scope": "local public host with real Qwen" if gateway else "local protocol fixture",
                  "image_bytes": inspect["Size"], "paid_request_count": len(live_calls),
                  "paid_charged_upper_cny": sum(row["qwen"]["charged_upper_cny"] for row in live_calls),
                  "official_scoring_ready": False}
        if gateway:
            ledger = gateway.budget.read()
            result["project_ledger_upper_cny"] = sum(row["charged_upper_cny"] for row in ledger["entries"])
        atomic_json(OUT / (args.run_id + ".json"), result)
        print(json.dumps({k: result[k] for k in ("run_id", "passed", "paid_request_count", "paid_charged_upper_cny")}), flush=True)


if __name__ == "__main__":
    main()
