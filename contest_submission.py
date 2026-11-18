"""Competition SDK entry: one bounded experiment and one gateway decision per round."""
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from functools import lru_cache
from time import monotonic, process_time

import numpy as np

from contest_engine.model import (
    ACTIONS, EXTENSION, NATIVE, TASKS as DATASETS, input_vector, load, save, sha256_file,
    source_rows, train_action,
)


TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
METRICS = {"A": "macro_f1", "B": "rmse", "C": "rmse"}
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def parent_files(rnd):
    paths = rnd.parent.descriptor["checkpoint_paths"]
    if not paths or Path(paths[0]).suffix != ".npz" or len(paths) > 2:
        raise ValueError("unsupported mounted checkpoint descriptor")
    if len(paths) == 2 and paths[1] != EXTENSION:
        raise ValueError("unsupported auxiliary checkpoint")
    return [rnd.parent.path(path) for path in paths]


def state_for(directory, dataset, round_index):
    memory = directory / "memory.json"
    if memory.exists():
        previous = json.loads(memory.read_text(encoding="utf-8"))
        if previous.get("dataset") != dataset or previous.get("round", round_index) >= round_index:
            raise ValueError("mounted state does not precede this dataset and round")
        return previous
    return {"attempts": [], "next_action": None}


def remaining_actions(actions, state, parent_key, train_hash):
    seen = {e["action"] for e in state["attempts"]
            if e["parent"] == parent_key and e["train"] == train_hash}
    return [action for action in actions if action not in seen]


def select_action(choices, state):
    preferred = state.get("next_action")
    return preferred if preferred in choices else next(iter(choices), None)


def history_facts(records):
    result = []
    for record in records[-4:]:
        config = record.get("configuration")
        if not isinstance(config, dict):
            config = {}
        row = {"id": record["id"]}
        for key in ("round", "action", "selection", "positive_is_better_gain"):
            value = config.get(key)
            if type(value) in (int, float) and math.isfinite(value):
                row[key] = value
            elif isinstance(value, str) and len(value) <= 64:
                row[key] = value
        result.append(row)
    return result


def official_metric(receipt, metric):
    summary = receipt.get("summary")
    metrics = summary.get("metrics") if isinstance(summary, dict) else None
    if not isinstance(metrics, dict):
        return {}
    result = {}
    for name in (metric, "accuracy", "mae", "nasa_score"):
        value = metrics.get(name)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            if name not in ("macro_f1", "accuracy") or value <= 1:
                result[name] = value
    return result


def submit(rnd, folder, hypothesis, change, config, seed):
    checkpoints = [NATIVE, EXTENSION] if (folder / EXTENSION).is_file() else [NATIVE]
    if sum((folder / path).stat().st_size for path in checkpoints) > MAX_ARTIFACT_BYTES:
        raise ValueError("checkpoint artifact exceeds the 64 MiB contest limit")
    return rnd.submit_candidate(folder, hypothesis=hypothesis, change_summary=change,
                                configuration=config, seed=seed, checkpoint_paths=checkpoints,
                                code_paths=[], predict_argv=PREDICT_ARGV)


def decision(rnd, evidence, actions):
    messages = [
        {"role": "system", "content": "Choose the industrial action using only the given measured facts. "
         "The engine computes improvement direction and blocks an ineligible child. "
         "Output only a JSON object with selection, next_action, and reason. "
         "Choose child only when eligible_child is true and the measured gain justifies it. "
         "For next_action use an option from next_if_child or next_if_parent matching your selection. "
         "If that list is empty, use hold. Give a brief reason of at most 100 characters."},
        {"role": "user", "content": json_text(evidence)},
    ]
    receipt = rnd.model(messages, max_tokens=min(rnd.max_output_tokens, 224), temperature=0, enable_thinking=False)
    usage = receipt.response.get("usage")
    tokens = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
              if isinstance(usage, dict) and type(usage.get(key)) is int and usage[key] >= 0}
    try:
        content = receipt.text.strip()
        if content.startswith("```json\n") and content.endswith("\n```"):
            content = content[8:-4].strip()
        parsed = json.loads(content)
        if (not isinstance(parsed, dict) or set(parsed) != {"selection", "next_action", "reason"}
                or parsed["selection"] not in ("parent", "child") or parsed["next_action"] not in (*actions, "hold")
                or not isinstance(parsed["reason"], str) or not 1 <= len(parsed["reason"].strip()) <= 100):
            raise ValueError("invalid decision fields")
    except (ValueError, TypeError):
        return None, tokens
    return parsed, tokens


def train(rnd):
    start, cpu_start = monotonic(), process_time()
    dataset = rnd.dataset_key
    if rnd.task.get("task") != DATASETS[dataset]:
        raise ValueError("frozen task contract conflicts with dataset")
    metric = METRICS[dataset]
    paths = parent_files(rnd)
    parent = load(*paths, dataset=dataset)
    seed = parent.native.meta["seed"]
    actions = ACTIONS[dataset]
    previous = rnd.history()
    train_hash = sha256_file(rnd.data.train_path)
    native_hash = sha256_file(paths[0])
    parent_key = [native_hash, sha256_file(paths[1]) if len(paths) == 2 else None]

    with tempfile.TemporaryDirectory(prefix="zhixun-", dir="/workspace") as location:
        work = Path(location)
        state_dir = rnd.state_workspace(work / "state")
        state = state_for(state_dir, dataset, rnd.round_index)
        choices = remaining_actions(actions, state, parent_key, train_hash)
        action = select_action(choices, state)
        phase = {"dataset": dataset, "round": rnd.round_index, "stage": rnd.stage_id,
                 "parent_artifact_id": rnd.parent.artifact_id, "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
                 "train_sha256": train_hash, "parent_files_sha256": parent_key, "action": action}
        if type(rnd.wall_seconds) in (int, float) and rnd.wall_seconds <= 8:
            action = None
            phase["action"] = None
        anchor_metrics, child_metrics, recipe = {}, {}, {}
        candidate = None
        error = "no_untried_action" if action is None else None
        uploads = []
        hypothesis = "Bounded source-only update may improve the current stage against the inherited model."
        change = "No new model was selected."
        if action is not None:
            anchor = work / "anchor"
            anchor.mkdir()
            for path, name in zip(paths, (NATIVE, EXTENSION)):
                shutil.copyfile(path, anchor / name)
            try:
                x, y = source_rows(rnd.data.training_rows(), parent.native)
                child, recipe = train_action(parent, action, x, y)
                candidate_dir = work / "candidate"
                candidate_dir.mkdir()
                save(child, candidate_dir, frozen_native=None if recipe.get("steps") else paths[0])
                check = load(candidate_dir / NATIVE, candidate_dir / EXTENSION if child.extension is not None else None,
                             dataset=dataset)
                np.testing.assert_allclose(check.predict(x[:16]), child.predict(x[:16]), rtol=1e-12, atol=1e-12)
                phase["train_rows"] = len(y)
                phase["changed_native_sha256"] = sha256_file(candidate_dir / NATIVE)
                change = f"Tested {action} from the mounted parent on this round's authorized training projection."
            except (ValueError, ArithmeticError, AssertionError) as exc:
                error = f"fit_{type(exc).__name__}"
                print(f"{dataset} round {rnd.round_index} {action}: {type(exc).__name__}", file=sys.stderr)
            if error is None:
                reference = submit(rnd, anchor, "Measure the identical mounted parent in this phase.",
                                   "Byte-for-byte copy of the mounted checkpoint and its extension.",
                                   {**phase, "role": "paired_parent"}, seed)
                uploads.append(reference["artifact_id"])
                anchor_metrics = official_metric(reference, metric)
                candidate = submit(rnd, candidate_dir, hypothesis, change, {**phase, "recipe": recipe}, seed)
                uploads.append(candidate["artifact_id"])
                child_metrics = official_metric(candidate, metric)
        gain = None
        if metric in anchor_metrics and metric in child_metrics:
            old, new = anchor_metrics[metric], child_metrics[metric]
            gain = new - old if dataset == "A" else old - new
        eligible = error is None and gain is not None and gain > max(1e-10, anchor_metrics[metric] * 1e-8)
        facts = {"dataset": dataset, "round": rnd.round_index, "metric": metric,
                 "direction": "maximize" if dataset == "A" else "minimize", "action": action,
                 "paired_parent": anchor_metrics, "child": child_metrics, "positive_is_better_gain": gain,
                 "eligible_child": eligible, "failure": error, "candidate_actions_for_next_round": list(actions),
                 "next_if_parent": [item for item in choices if item != action],
                 "next_if_child": list(actions) if eligible else [],
                 "history_checkpoint_count": len(previous), "recent_host_history": history_facts(previous),
                 "recent_attempts": state["attempts"][-4:]}
        proposal, tokens = decision(rnd, facts, actions)
        accept = eligible and proposal is not None and proposal["selection"] == "child"
        selected = candidate["artifact_id"] if accept else rnd.parent.artifact_id
        decision_status = "accepted" if accept else "kept_parent"
        config = {**phase, "recipe": recipe, "model_name": rnd.model_name,
                  "eligible_child": eligible, "measurement_ids": uploads, "paired_parent": anchor_metrics,
                  "measured_child": child_metrics, "positive_is_better_gain": gain, "error": error,
                  "selection": decision_status, "proposal": proposal, "gateway_token_usage": tokens,
                  "selected_artifact_id": selected}
        if action is not None:
            state["attempts"].append({"parent": parent_key, "train": train_hash, "action": action,
                                      "gain": gain, "decision": decision_status, "failure": error})
        config["cpu_seconds"] = process_time() - cpu_start
        config["participant_wall_seconds"] = monotonic() - start
        record = {"dataset": dataset, "round": rnd.round_index, "next_action": (proposal or {}).get("next_action"),
                  "attempts": state["attempts"][-20:], "last": config}
        (state_dir / "memory.json").write_text(json_text(record), encoding="utf-8")
        rnd.save_state(state_dir)
        reason = (f"{decision_status}; model call {rnd.call_ids[-1]}; valid decision={proposal is not None}; "
                  f"paired {metric} gain={gain}; positive means improvement; eligible={eligible}; failure={error}.")
        rnd.finish(selected_artifact_id=selected, hypothesis=hypothesis, change_summary=change,
                   reason=reason, configuration=config, seed=seed,
                   history_ids=[item["id"] for item in previous], model_call_ids=list(rnd.call_ids))


@lru_cache(maxsize=1)
def predictor(path, extension, dataset):
    return load(path, extension, dataset=dataset)


def predict(sample, model_dir):
    root = Path(model_dir).resolve()
    path = Path(os.environ.get("CONTEST_ARTIFACT_PATH", root / NATIVE))
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("selected checkpoint must be within /model")
    extension = root / EXTENSION
    if extension.is_symlink():
        raise ValueError("extension must be a regular mounted file")
    if sample.context is not None:
        declared = sample.context["artifact"].get("checkpoint_paths", [])
        if path.relative_to(root).as_posix() not in declared or (EXTENSION in declared) != extension.is_file():
            raise ValueError("checkpoint files differ from the selected artifact descriptor")
    model = predictor(path, extension if extension.is_file() else None, sample.dataset_key)
    if sample.task != DATASETS[model.dataset]:
        raise ValueError("prediction type differs from the checkpoint")
    x = np.asarray([input_vector(sample.inputs, model.native)], dtype=np.float64)
    result = model.predict(x)[0]
    if sample.task == "regression":
        return {"value": float(result)}
    scores = result.tolist()
    best = int(np.argmax(scores))
    if scores.count(scores[best]) > 1:
        scores[best] = math.nextafter(scores[best], 1.)
    return [{"label": label, "score": score} for label, score in zip(model.native.meta["classes"], scores)]
