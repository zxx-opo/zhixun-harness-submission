"""One slim-pi decision through the SDK transport, then the same candidate handshake."""
import os
import subprocess
from pathlib import Path

PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]


def predict(sample, model_dir):
    del model_dir
    if sample.task == "regression":
        return {"value": 0}
    return [{"label": "normal", "score": 1}]


def train(rnd):
    pi = os.environ.get("CONTEST_PI")
    if not pi:
        raise RuntimeError("CONTEST_PI is required")
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    work.mkdir(parents=True, exist_ok=True)
    session = work / "pi-session"
    session.mkdir(exist_ok=True)
    with rnd.transport() as transport:
        env = os.environ.copy()
        env.update({
            "PI_BASE_URL": transport.base_url,
            "PI_API_KEY": transport.api_key,
            "PI_MODEL": rnd.model_name,
            "PI_MAX_TOKENS": "64",
            "PI_SESSION_DIR": str(session),
            "HOME": str(work),
        })
        completed = subprocess.run(
            ["node", pi, "-p", "不要使用工具，不要读写文件。只回复：协议检查通过。"],
            cwd=str(work), env=env, timeout=90, check=False, text=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if completed.returncode != 0 or not rnd.call_ids:
            raise RuntimeError("slim pi did not complete a model call: %s" % completed.returncode)
    candidate = work / "candidate"
    state = work / "state"
    candidate.mkdir(exist_ok=False)
    state.mkdir(exist_ok=False)
    (candidate / "model.bin").write_bytes(b"contest-harness-pi-example\n")
    (state / "note.json").write_text("{}\n", encoding="utf-8")
    rnd.save_state(state)
    submitted = rnd.submit_candidate(
        candidate,
        hypothesis="由精简 pi 通过 SDK 传输读取平台提示后做一次确认。",
        change_summary="保存示例候选，不替换官方训练代码。",
        configuration={"model_call_ids": list(rnd.call_ids)},
        seed=1,
        checkpoint_paths=["model.bin"],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis="由精简 pi 通过 SDK 传输读取平台提示后做一次确认。",
        change_summary="保存示例候选，不替换官方训练代码。",
        reason="精简 pi 已通过 SDK 传输完成模型调用。",
        configuration={"model_call_ids": list(rnd.call_ids)},
        seed=1,
        history_ids=list(rnd.history_ids),
    )
