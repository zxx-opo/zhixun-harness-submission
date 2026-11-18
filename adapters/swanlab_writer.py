"""SwanLab run mapping and local-first, restartable upload of public experiment facts."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from harness.data import ROOT, atomic_json, digest, read_json
from harness.storage import Ledger, encode, exclusive


SDK_VERSION = "0.10.5"


def destination():
    names = ("SWANLAB_WORKSPACE", "SWANLAB_PROJECT")
    values = tuple(os.environ.get(name, "").strip() for name in names)
    if any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", value) for value in values):
        raise ValueError("configure SWANLAB_WORKSPACE and SWANLAB_PROJECT explicitly")
    return values


def sdk(online):
    import swanlab
    if version("swanlab") != SDK_VERSION:
        raise RuntimeError(f"SwanLab is locked to {SDK_VERSION}")
    if online:
        api_key = os.environ.get("SWANLAB_API_KEY", "").strip()
        if not api_key:
            raise ValueError("SWANLAB_API_KEY is required for online upload")
        if not swanlab.login(api_key=api_key, host=os.environ.get("SWANLAB_HOST"), relogin=True, save=False, timeout=15):
            raise RuntimeError("SwanLab did not verify the supplied key")
    return swanlab


def stable_run_id(target, name):
    return sha256(encode((*target, name)).encode()).hexdigest()[:28]


def open_run(swanlab, target, directory, name, metadata, online):
    workspace, project = target
    run_id = stable_run_id(target, name)
    run = swanlab.init(workspace=workspace, project=project, id=run_id, name=name,
                       group=metadata.get("session_id"), job_type=metadata.get("kind", "candidate"),
                       config=metadata, resume="allow" if online else "never", log_dir=str(directory),
                       mode="online" if online else "offline", on_online_error="raise")
    return run, run_id


def remote_query(command, path, *options):
    # Read by the documented CLI, using environment login in the short-lived child.
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTRIO_")}
    completed = subprocess.run([sys.executable, "-m", "swanlab", "api", "run", command, path, *options],
                               env=env, text=True, encoding="utf-8",
                               errors="replace", capture_output=True, timeout=35)
    if completed.returncode:
        return None
    try:
        contents = json.loads(completed.stdout)
    except ValueError:
        return None
    if isinstance(contents, dict) and "ok" in contents:
        return contents.get("data") if contents["ok"] is True else None
    return contents


def remote_info(target, run_id):
    path = "/".join((*target, run_id))
    details = remote_query("info", path)
    return {"query": path, "remote_found": True, "state": details.get("state"),
            "finished_at": details.get("finished_at")} if isinstance(details, dict) else None


def run_objects(payload):
    if isinstance(payload, list):
        for entry in payload:
            yield from run_objects(entry)
    elif isinstance(payload, dict):
        if any(isinstance(payload.get(key), str) for key in ("run_id", "runId", "cuid", "id", "experimentId")):
            yield payload
        for value in payload.values():
            if isinstance(value, (list, dict)):
                yield from run_objects(value)


def check(directory, target):
    directory.mkdir(parents=True, exist_ok=True)
    workspace, project = target
    swanlab = sdk(True)
    name = f"zhixun-v11-sdk-probe-{sha256(str(directory.resolve()).encode()).hexdigest()[:10]}"
    run, run_id = open_run(swanlab, target, directory / "probe-logs", name,
                           {"experiment_id": name, "kind": "connectivity_probe", "sdk_version": SDK_VERSION}, True)
    run.define_metric("probe/connected", x_axis="axis/event_seq")
    run.log({"axis/event_seq": 1, "probe/connected": 1}, step=1)
    url = run.url
    run.finish()
    confirmation = remote_info(target, run_id)
    facts = {"status": "remote_verified" if confirmation and confirmation["remote_found"] else "sdk_finished_remote_unverified",
             "workspace": workspace, "project": project, "sdk_version": SDK_VERSION,
             "id": run_id, "url": url, "verification": confirmation}
    atomic_json(directory / "swanlab-probe.json", facts)
    print(encode(facts), flush=True)


def artifact_manifest(stages):
    entries = []
    for stage in stages:
        if stage["status"] != "complete":
            continue
        plan = stage["plan"]
        for path in sorted(Path(stage["path"]).iterdir()):
            if path.is_file():
                try:
                    relative = path.relative_to(ROOT).as_posix()
                except ValueError:
                    relative = "/".join(path.parts[-3:])
                entries.append({"rung": stage["rung"], "path": relative, "sha256": digest(path),
                                "bytes": path.stat().st_size, "resume_parent": plan["resume_checkpoint_hash"]})
    return entries


def points(trial, stages, calls=(), rollbacks=()):
    yield {"trace/plan": encode({key: trial[key] for key in
           ("id", "action_config", "parent", "selection", "applicability", "policy", "resources_before")})}
    for stage in stages:
        cost = stage.get("cost", {})
        if "wall_seconds" in cost:
            yield {"resource/train_seconds": cost["wall_seconds"],
                   **({"resource/peak_rss_bytes": cost["peak_rss_bytes_sampled"]} if cost.get("peak_rss_bytes_sampled") else {})}
        if stage["status"] != "complete":
            yield {"trace/stage_failure": encode({"rung": stage["rung"], "failure": stage.get("failure"),
                                                 "reserved_train_credits": stage["reserved_passes"]})}
            continue
        receipt = stage["receipt"]
        for epoch in receipt["history"]:
            yield {"axis/optimizer_step": epoch["optimizer_step"], "axis/epoch": epoch["epoch"],
                   "train/loss": epoch["loss"], "train/learning_rate": epoch["learning_rate"]}
        if receipt["solver"]:
            yield {"solver/fit_index": 1, "solver/train_standardized_mse": receipt["solver"]["train_standardized_mse"],
                   "trace/solver": encode(receipt["solver"])}
        yield {"resource/train_samples": receipt["samples_observed"], "resource/optimizer_updates": receipt["optimizer_updates"],
               "resource/added_train_passes": receipt["training_passes"]}
    unit = trial["unit"]
    for index, item in enumerate(trial["observations"], 1):
        if "metrics" not in item:
            yield {"trace/evaluation_error": encode(item)}
            continue
        result = item["metrics"]
        names = ["macro_f1", "accuracy"] if unit == "class" else ["rmse", "mae"]
        values = {f"eval/{trial['task']}/{metric}" + ("" if unit == "class" else f"_{unit}"): result[metric]
                  for metric in names}
        yield {"axis/evaluation_index": index, **values, "trace/evaluation": encode({
            "eval_key": item["evaluation"], "units": unit, "groups": result["groups"],
            "target_passes": item["target_passes"], "checkpoint": item["digest"]})}
    for call in calls:
        usage = call.get("usage") or {}
        yield {"llm/request_count": 1, "llm/seconds": call.get("wall_seconds", 0.0),
               "llm/failed_requests": int(call["status"] != "accepted"),
               **{f"llm/{key}": usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens") if key in usage},
               "trace/llm": encode(call)}
    yield {"attempt/completed": int(bool(trial["observations"]) and any("metrics" in o for o in trial["observations"])),
           "attempt/improved": int(trial["eligible"]), "decision/accepted": int(trial["decision"] == "promoted"),
           "trace/decision": encode({"status": trial["decision"], "reasons": trial["reasons"],
                                     "child": trial.get("child"), "gain": trial.get("gain")}),
           "trace/artifact_manifest": encode(artifact_manifest(stages))}
    for rollback in rollbacks:
        yield {"decision/rolled_back": 1, "trace/rollback": encode(rollback)}


def upload_trial(swanlab, target, directory, trial, stages, session, online, calls, rollbacks):
    name = trial["id"]
    config = {"kind": "candidate", "experiment_id": name, "session_id": session, "task": trial["task"],
              "unit": trial["unit"], "action_config": trial["action_config"], "parent_checkpoint": trial["parent"],
              "applicability": trial["applicability"], "policy": trial["policy"], "sdk_version": SDK_VERSION}
    run, run_id = open_run(swanlab, target, directory, name, config, online)
    try:
        for pattern, axis in (("train/*", "axis/optimizer_step"), ("eval/*", "axis/evaluation_index"),
                              ("resource/*", "axis/event_seq"), ("llm/*", "axis/event_seq")):
            run.define_metric(pattern, x_axis=axis, overwrite=True)
        for i, values in enumerate(points(trial, stages, calls, rollbacks), 1):
            mapped = {key: swanlab.Text(value) if key.startswith("trace/") else value for key, value in values.items()}
            run.log({"axis/event_seq": i, **mapped}, step=i)
        url = run.url if online else None
    except Exception:
        run.finish(state="crashed", error="tracking serialization failed", async_log_timeout=40)
        raise
    if trial["decision"] == "failed":
        run.finish(state="crashed", error="industrial trial failed; see trace/decision", async_log_timeout=40)
    else:
        run.finish(async_log_timeout=40)
    return {"id": run_id, "url": url, "status": "sdk_finished" if online else "offline_pending"}


def upload_session(swanlab, root, target, name, study, online):
    run_id = stable_run_id(target, name)
    run, run_id = open_run(swanlab, target, root / "swanlab" / run_id, name,
                           {"experiment_id": name, "session_id": study["session_id"],
                            "kind": "session", "protocol": study["scope"], "sdk_version": SDK_VERSION}, online)
    step = 0
    e0 = read_json(root / "E0.json")
    for task, facts in sorted(e0["tasks"].items()):
        step += 1
        metric = "macro_f1" if task == "A" else ("rmse_minutes" if task == "B" else "rmse_cycles")
        score = facts["baseline_validation"]["macro_f1" if task == "A" else "rmse"]
        run.log({f"E0/{task}/{metric}": score, f"E0/{task}/optimizer_updates_each_path": facts["adam_steps_each_path"],
                 f"E0/{task}/resume_equal": int(facts["resume_bytes_equal"]),
                 f"E0/{task}/verify_compute_count": facts["reference_validation_metrics_compute_count"],
                 "trace/E0": swanlab.Text(encode({"task": task, "seed": facts["seed"], "train_sha": facts["train_sha256"],
                                                 "validation_sha": facts["validation_sha256"],
                                                 "archives_verified": facts["original_archive_verified"]}))}, step=step)
    for task, arms in sorted(study["experiment_arms"].items()):
        step += 1
        scoped = {arm: {field: output[field] for field in (
            "training_passes", "optimizer_updates", "metric_compute_count", "metric_release_count", "started_trials")
        } for arm, output in arms.items()}
        run.log({"trace/arm_accounting": swanlab.Text(encode({"task": task, "arms": scoped}))}, step=step)
    for call in study.get("qwen", {}).get("calls", []):
        if call["status"] != "started":
            step += 1
            run.log({"trace/trainer_accounting": swanlab.Text(encode(call))}, step=step)
    step += 1
    run.log({"session/completed": 1, "session/confirmation_scans": study["confirmation_access_count"]}, step=step)
    url = run.url if online else None
    run.finish(async_log_timeout=40)
    return {"id": run_id, "url": url, "status": "sdk_finished" if online else "offline_pending", "kind": "session"}


def sync(root, target, online, maximum=None):
    workspace, project = target
    path = root / "swanlab-index.json"
    with exclusive(root / ".swanlab-writer.lock"):
        index = read_json(path) if path.exists() else {"workspace": workspace, "project": project,
                                                      "sdk_version": SDK_VERSION, "runs": {}}
        if (index["workspace"], index["project"]) != (workspace, project):
            raise ValueError("tracking index already binds a different SwanLab destination")
        swanlab = sdk(online)
        uploaded = 0
        pending = []
        shared = Ledger(root / "shared")
        try:
            calls = shared.calls()
        finally:
            shared.close()
        for db_path in sorted((root / "arms").glob("*/*/ledger.sqlite3")):
            ledger = Ledger(db_path.parent)
            try:
                arm = ledger.setting("config")
                for trial in ledger.history():
                    name = trial["id"]
                    if trial["decision"] in {"running", "reused"}:
                        continue
                    previous = index["runs"].get(name, {})
                    if previous.get("status") in {"sdk_finished", "remote_verified"}:
                        continue
                    if not online and previous.get("status") == "offline_pending":
                        continue
                    if maximum is not None and uploaded >= maximum:
                        pending.append(name)
                        continue
                    stage_rows = ledger.db.execute("SELECT data FROM rungs WHERE trial_id=? ORDER BY rung", (name,))
                    stages = [json.loads(row[0]) for row in stage_rows]
                    trace = [call for call in calls if call.get("experiment") == name and call["status"] != "started"]
                    rollbacks = [event["data"] for event in ledger.events("rolled_back")
                                 if event["data"].get("from") == trial.get("child") and trial.get("child")]
                    # Record before network I/O. Stable ID and deterministic steps are reused on recovery.
                    run_id = stable_run_id(target, name)
                    if (online and previous.get("mode") != "offline"
                            and previous.get("status") in {"upload_in_progress", "reconciliation_required"}):
                        found = remote_info(target, run_id)
                        if found:
                            completed = remote_query("summary", found["query"], "--keys", "attempt/completed")
                            if found["state"] in {"FINISHED", "CRASHED", "SUCCESS"} and completed:
                                index["runs"][name] = {"id": run_id, "status": "remote_verified", "verification": found,
                                                       "url": f"https://swanlab.cn/@{workspace}/{project}/runs/{run_id}"}
                            else:
                                pending.append(name)
                                index["runs"][name].update(status="reconciliation_required", verification=found)
                            atomic_json(path, index)
                            continue
                        # Failure to read the remote can leave delivery ambiguous. Preserve the original ID.
                        pending.append(name)
                        continue
                    index["runs"][name] = {"id": run_id, "status": "upload_in_progress", "mode": "online" if online else "offline"}
                    atomic_json(path, index)
                    try:
                        index["runs"][name] = upload_trial(swanlab, target, root / "swanlab" / run_id, trial, stages,
                                                          arm["session_id"], online, trace, rollbacks)
                    except Exception as exc:
                        index["runs"][name].update(status="reconciliation_required", error=type(exc).__name__)
                        pending.append(name)
                        atomic_json(path, index)
                        break
                    print(f"SwanLab {name}: {index['runs'][name]['status']}", flush=True)
                    atomic_json(path, index)
                    uploaded += 1
            finally:
                ledger.close()
        study_file = root / "study.json"
        if maximum is None and study_file.is_file():
            study = read_json(study_file)
            if study.get("status") in {"public_matrix_complete", "qwen_recheck_complete"} and not pending:
                name = f"{study['session_id']}-evidence"
                prior = index["runs"].get(name, {})
                if not prior or (online and prior["status"] == "offline_pending"):
                    index["runs"][name] = {"id": stable_run_id(target, name), "status": "upload_in_progress", "kind": "session"}
                    atomic_json(path, index)
                    try:
                        index["runs"][name] = upload_session(swanlab, root, target, name, study, online)
                    except Exception as exc:
                        index["runs"][name].update(status="reconciliation_required", error=type(exc).__name__)
                        pending.append(name)
                elif prior["status"] in {"upload_in_progress", "reconciliation_required"}:
                    pending.append(name)
        if online:
            cloud = remote_query("list", "/".join(target), "--all")
            expected = {value["id"] for value in index["runs"].values()}
            observed = {}
            duplicates = {}
            for run in run_objects(cloud):
                for key in ("run_id", "runId", "cuid", "id", "experimentId"):
                    run_id = run.get(key)
                    if isinstance(run_id, str) and run_id in expected:
                        duplicates[run_id] = duplicates.get(run_id, 0) + 1
                        observed[run_id] = run
                        break
            index["cloud_duplicate_run_ids"] = sorted(key for key, count in duplicates.items() if count > 1)
            index["cloud_list_verified_count"] = len(observed)
            for value in index["runs"].values():
                if value["id"] in observed and value["status"] in {"sdk_finished", "remote_verified"}:
                    value.update(status="remote_verified", verification={"remote_found": True,
                                 "state": observed[value["id"]].get("state")})
            if not observed:
                sample = next((value for value in index["runs"].values() if value["status"] == "sdk_finished"), None)
                if sample:
                    proof = remote_info(target, sample["id"])
                    if proof and proof["remote_found"]:
                        sample.update(status="remote_verified", verification=proof)
        index["reported_pending"] = sorted(set(pending))
        atomic_json(path, index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "sync"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--maximum", type=int)
    args = parser.parse_args()
    target = destination()
    # The 0.10.5 settings parser treats SWANLAB_PROJECT as a JSON object.
    # Our launcher intentionally accepts a project slug and passes it explicitly.
    os.environ.pop("SWANLAB_PROJECT", None)
    os.environ.pop("SWANLAB_WORKSPACE", None)
    if args.command == "check":
        check(args.directory.resolve(), target)
    else:
        result = sync(args.directory.resolve(), target, not args.offline, args.maximum)
        counts = {state: sum(row["status"] == state for row in result["runs"].values())
                  for state in sorted({row["status"] for row in result["runs"].values()})}
        print(encode({"runs": counts, "workspace": result["workspace"], "project": result["project"],
                      "pending_known": len(result["reported_pending"])}))


if __name__ == "__main__":
    main()
