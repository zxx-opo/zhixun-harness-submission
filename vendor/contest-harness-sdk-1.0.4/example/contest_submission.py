"""Protocol example. It proves the SDK round, not a leaderboard model."""
import os
from pathlib import Path

from contest_harness import Round

PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]


def train(rnd: Round):
    preview = []
    for index, row in enumerate(rnd.data.training_rows()):
        preview.append(row["id"])
        if index == 1:
            break
    history = rnd.history()
    receipt = rnd.model(
        [{"role": "user", "content": "本轮公开训练样本预览 %d 条，请用一个词确认已收到任务提示。" % len(preview)}],
        max_tokens=min(32, rnd.max_output_tokens),
        temperature=0,
    )
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    candidate = work / "candidate"
    state = work / "state"
    candidate.mkdir(parents=True, exist_ok=False)
    state.mkdir(parents=True, exist_ok=False)
    (candidate / "model.bin").write_bytes(b"contest-harness-sdk-example\n")
    (state / "note.json").write_text('{"round":%d}\n' % rnd.round_index, encoding="utf-8")
    rnd.save_state(state)
    submitted = rnd.submit_candidate(
        candidate,
        hypothesis="用平台固定提示和公开 JSONL 做一次协议检查。",
        change_summary="保存一个示例候选文件，不改变官方训练算法。",
        configuration={"preview_ids": preview, "model_call_id": receipt.call_id},
        seed=1,
        checkpoint_paths=["model.bin"],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis="用平台固定提示和公开 JSONL 做一次协议检查。",
        change_summary="保存一个示例候选文件，不改变官方训练算法。",
        reason="已通过 SDK 完成一次模型调用，调用号 %s。" % receipt.call_id,
        configuration={"preview_ids": preview, "model_call_id": receipt.call_id},
        seed=1,
        history_ids=[item["id"] for item in history],
    )


def predict(sample, model_dir):
    del model_dir
    if sample.task == "regression":
        return {"value": 0}
    return [{"label": "normal", "score": 1}]
