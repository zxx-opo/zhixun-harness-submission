"""Reconcile real calls and publish the exact contest image acceptance to SwanLab."""
import argparse
import json
import math
import os
from pathlib import Path
from time import sleep

from adapters.swanlab_writer import destination, open_run, remote_query, sdk, stable_run_id
from contest_engine.model import sha256_file
from harness.data import atomic_json, read_json
from harness.storage import exclusive


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/contest-submission-1.0.4"
ACCEPTANCE = REPORT / "contest-104-release.json"
LOCAL = ROOT / "runs/contest-104-release"
VERIFY = REPORT / "acceptance-verification.json"
REMOTE = REPORT / "swanlab-verification.json"
TRACKING = LOCAL / "swanlab"


def reconcile():
    audit = read_json(ACCEPTANCE)
    fixture = read_json(REPORT / "contest-104-fixture.json")
    ledger = read_json(ROOT / "runs/v12-trainer/cost-ledger.json")
    all_rows = [row for run in audit["runs"].values() for row in run["rounds"]]
    if (audit["mode"] != "live" or set(audit["runs"]) != set("ABC") or len(all_rows) != 30
            or any(len(run["rounds"]) != 10 for run in audit["runs"].values())
            or fixture["image_id"] != audit["image_id"] or not fixture["passed"]):
        raise ValueError("live and fixture verification must bind to the exact same image")
    costs = [row for row in ledger["entries"] if row["id"].startswith(audit["run_id"] + "-")]
    successes = {row["qwen"]["id"] for row in all_rows}
    fresh_costs = sum(row["charged_upper_cny"] for row in costs)
    if (len(successes) != 30 or successes != {row["id"] for row in costs if row["status"] == "completed"}
            or not math.isclose(sum(row["charged_upper_cny"] for row in ledger["entries"]), audit["project_ledger_upper_cny"], abs_tol=1e-8)):
        raise ValueError("gateway calls and project ledger do not reconcile")
    model = audit["model"]
    for row in all_rows:
        qwen = row["qwen"]
        if qwen["request"]["model"] != model or not (qwen["provider"].get("id") or qwen["provider"].get("request_id")):
            raise ValueError("missing authentic model identity in the receipt")
    prior_failures = [row for row in costs if row["status"] != "completed"]
    result = {"passed": True, "image_id": audit["image_id"], "run_id": audit["run_id"], "model": model,
              "acceptance_sha256": sha256_file(ACCEPTANCE), "rounds": len(all_rows), "host": "local public projection",
              "provider_successes": len(successes), "provider_attempts": len(costs), "failed_attempts": prior_failures,
              "parse_valid": sum(row["configuration"]["proposal"] is not None for row in all_rows),
              "paired_rounds": sum(len(row["measurements"]) == 2 for row in all_rows),
              "history_experiments": sum(len(row["measurements"]) == 2 and bool(row["history_ids"]) for row in all_rows),
              "accepted": sum(row["configuration"]["selection"] == "accepted" for row in all_rows),
              "measurement_count": sum(len(row["measurements"]) for row in all_rows),
              "largest_candidate_bytes": max(sum(file["size_bytes"] for file in item["artifact"]["files"])
                                             for row in all_rows for item in row["measurements"]),
              "upload_chunks": sum(row["upload_chunks"] for row in all_rows),
              "prompt_tokens": sum(row["qwen"]["response"]["usage"].get("prompt_tokens", 0) for row in all_rows),
              "completion_tokens": sum(row["qwen"]["response"]["usage"].get("completion_tokens", 0) for row in all_rows),
              "missing_usage_count": sum(not row["qwen"]["response"]["usage"] for row in all_rows),
              "new_paid_upper_cny": fresh_costs, "cumulative_paid_upper_cny": audit["project_ledger_upper_cny"],
              "approved_cny": ledger["cap"], "ledger_basis": ledger["billing_basis"], "official_scoring_ready": False,
              "offline_fixture": {"run_id": fixture["run_id"], "scored_candidates": fixture["runs"]["A"]["candidate_measurements"],
                                  "kept_parent": fixture["runs"]["A"]["accepted"] == 0}}
    if result["history_experiments"] < 3:
        raise ValueError("fewer than three recorded history-experiment-decision steps")
    atomic_json(VERIFY, result)
    return audit, result


def points(audit, verify):
    for task, run in audit["runs"].items():
        for row in run["rounds"]:
            config = row["configuration"]
            stats = {f"score/{task}/current": row["selected_metrics"]["macro_f1" if task == "A" else "rmse"],
                     "round/paired_measurements": len(row["measurements"]), "round/accepted": int(config["selection"] == "accepted"),
                     "round/next_round": config["round"], "round/cpu_seconds": config["cpu_seconds"],
                     "round/wall_seconds": row["wall_seconds"], "round/upload_chunks": row["upload_chunks"],
                     "trace/decision": {"task": task, "qwen": row["qwen"], "history_ids": row["history_ids"],
                                        "configuration": config, "selected": row["selected"]}}
            if config["positive_is_better_gain"] is not None:
                stats[f"score/{task}/paired_gain"] = config["positive_is_better_gain"]
            yield stats
    yield {**{f"final/{task}/{key}": value for task, run in audit["runs"].items() for key, value in run["final"].items()},
           "summary/rounds": verify["rounds"], "summary/qwen_successes": verify["provider_successes"],
           "summary/qwen_attempts": verify["provider_attempts"], "summary/accepted": verify["accepted"],
           "summary/measurements": verify["measurement_count"], "summary/upper_cny": verify["new_paid_upper_cny"],
           "summary/finished": 1, "trace/verification": verify}


def readback(target, run_id, totals):
    query = "/".join((*target, run_id))
    for wait in (0, 3, 7):
        sleep(wait)
        summary = remote_query("summary", query)
        if not isinstance(summary, dict) or summary.get("summary/finished", {}).get("value") != 1:
            continue
        missing = []
        for key, expected in totals.items():
            if key.startswith("trace/"):
                continue
            actual = summary.get(key, {}).get("value")
            if not isinstance(actual, (float, int)) or not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9):
                missing.append(key)
        info = remote_query("info", query)
        if not missing and isinstance(info, dict):
            return {"passed": True, "id": run_id, "workspace": target[0], "project": target[1], "query": query,
                    "state": info.get("state"), "url": f"https://swanlab.cn/@{target[0]}/{target[1]}/runs/{run_id}",
                    "scalar_checks": len([key for key in totals if not key.startswith("trace/")])}
    raise RuntimeError("SwanLab evidence was not read back with matching final scalars")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    audit, verify = reconcile()
    tracking_points = list(points(audit, verify))
    atomic_json(TRACKING / "payload.json", tracking_points)
    if not args.online:
        print(json.dumps(verify, ensure_ascii=False))
        return
    target = destination()
    if target != ("zxx12123", "zhixun-engine-dev"):
        raise ValueError("SwanLab destination is different from the project's authorized workspace")
    # Pass the resolved destination explicitly instead of letting the SDK parse it.
    os.environ.pop("SWANLAB_PROJECT", None)
    os.environ.pop("SWANLAB_WORKSPACE", None)
    name = "contest-sdk-1.0.4-real-qwen-release"
    run_id = stable_run_id(target, name)
    with exclusive(TRACKING / ".upload.lock"):
        status = TRACKING / "upload.json"
        if not status.exists():
            swanlab = sdk(True)
            config = {"experiment_id": audit["run_id"], "session_id": "contest-1.0.4", "kind": "acceptance",
                      "scope": audit["scope"], "image_id": audit["image_id"], "model": audit["model"],
                      "acceptance_sha256": verify["acceptance_sha256"], "official_scoring_ready": False}
            atomic_json(status, {"id": run_id, "state": "started", "acceptance_sha256": verify["acceptance_sha256"]})
            run, _ = open_run(swanlab, target, TRACKING / "log", name, config, True)
            for step, point in enumerate(tracking_points, 1):
                run.log({key: swanlab.Text(json.dumps(value, ensure_ascii=False)) if key.startswith("trace/") else value
                         for key, value in point.items()}, step=step)
            run.finish(async_log_timeout=40)
        else:
            entry = read_json(status)
            if entry["id"] != run_id or entry["acceptance_sha256"] != verify["acceptance_sha256"]:
                raise ValueError("existing SwanLab run is bound to another acceptance")
        outcome = readback(target, run_id, tracking_points[-1])
        outcome["acceptance_sha256"] = verify["acceptance_sha256"]
        atomic_json(REMOTE, outcome)
        print(json.dumps(outcome, ensure_ascii=False))


if __name__ == "__main__":
    main()
