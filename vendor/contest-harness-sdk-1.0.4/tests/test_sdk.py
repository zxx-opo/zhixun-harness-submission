"""The SDK must speak the current workspace v2 frames, not a private dialect."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.support import Host  # noqa: E402

PROMPT = "从官方基线与公开训练、验证数据出发，在授权预算内自主提出假设。"


def start_frame(train_path, validation_path):
    digest = "sha256:" + "ab" * 32
    parent = {
        "artifact_id": "baseline-a",
        "artifact_sha256": digest,
        "artifact": {
            "checkpoint_paths": ["checkpoint.npz"],
            "code_paths": [],
            "predict_argv": ["python3", "/app/harness.py", "predict"],
        },
        "files": [{"path": "checkpoint.npz", "sha256": digest, "size_bytes": 3}],
    }
    return {
        "protocol": "contest.harness.workspace.v2",
        "run_id": "run-test",
        "dataset_key": "A",
        "round_index": 1,
        "stage_id": "public-development",
        "prompt": PROMPT,
        "prompt_version": "industrial-workspace-flexible-v3",
        "rounds_per_dataset": 10,
        "parent_checkpoint_sha256": digest,
        "task_contract": {
            "dataset_key": "A",
            "task": "classification",
            "metric": "macro_f1",
            "metric_direction": "higher",
            "public_projection": {
                "train_path": str(train_path),
                "validation_path": str(validation_path),
                "row_fields": ["id", "inputs", "answer"],
                "source_metadata": False,
            },
        },
        "model_name": "base:test-model",
        "parent_artifact": parent,
        "artifacts": [parent],
        "state_sha256": "",
        "state_path": None,
        "max_output_tokens": 128,
        "resource_limits": {"artifact_upload_protocol": "chunked-v1", "artifact_bytes": 512 * 1024 * 1024, "state_bytes": 512 * 1024 * 1024, "candidate_upload_bytes_per_round": 1024 * 1024 * 1024, "state_upload_bytes_per_round": 512 * 1024 * 1024},
        "current_validation": {},
        "wall_seconds": 60,
    }


def stub_model(messages, options):
    assert options["max_tokens"] == 32
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(PROMPT + "\n")
    assert messages[1]["role"] == "user"
    assert PROMPT not in messages[1]["content"]
    return {
        "request_id": "call-1",
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "收到"}}],
    }


class SdkProtocolTest(unittest.TestCase):
    def env(self, work, submission="contest_submission"):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "example")
        env["CONTEST_SUBMISSION"] = submission
        env["CONTEST_WORK"] = str(work)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def test_describe_reports_workspace_v2(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "template" / "harness.py"), "describe"],
            cwd=ROOT / "example", env=self.env(ROOT), text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        described = json.loads(completed.stdout)
        self.assertEqual(described["protocol"], "contest.harness.workspace.v2")
        self.assertEqual(described["model_access"], "host_gateway_rpc")
        self.assertEqual(described["tasks"], ["classification", "regression"])
        self.assertEqual(described["sdk"], "contest_harness")

    def test_train_reads_prompt_data_and_leaves_prompt_attachment_to_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            train.write_text('{"id":"s1","inputs":{"x":1},"answer":"normal"}\n{"id":"s2","inputs":{"x":2},"answer":"fault"}\n', encoding="utf-8")
            validation.write_text('{"id":"v1","inputs":{"x":3},"answer":"normal"}\n', encoding="utf-8")
            work = root / "work"
            work.mkdir()
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "template" / "harness.py"), "train"],
                cwd=ROOT / "example", env=self.env(work), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
            host = Host(start_frame(train, validation), stub_model, history=[{"id": "round-0", "round_index": 0}])
            thread = host.serve(process)
            stderr = process.stderr.read()
            code = process.wait(timeout=20)
            thread.join(timeout=5)
            self.assertEqual(code, 0, stderr)
            self.assertIsNone(host.failure, host.failure)
            self.assertEqual(host.raw_messages, [[{"role": "user", "content": "本轮公开训练样本预览 2 条，请用一个词确认已收到任务提示。"}]])
            self.assertTrue(host.bound_messages[0][0]["content"].startswith(PROMPT))
            self.assertEqual(host.record["selected_artifact_id"], "model-test-1")
            self.assertEqual(host.record["model_call_ids"], ["call-1"])
            self.assertEqual(host.record["history_ids"], ["round-0"])
            self.assertEqual(host.record["configuration"]["preview_ids"], ["s1", "s2"])

    def test_predict_replies_before_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = self.env(Path(tmp))
            env["CONTEST_ARTIFACT_PATH"] = tmp
            process = subprocess.run(
                [sys.executable, str(ROOT / "template" / "harness.py"), "predict"],
                input='{"task":"classification","inputs":{"x":1}}\n{"task":"regression","inputs":{"x":2}}\n',
                env=env, text=True, capture_output=True, check=False,
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        lines = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(lines[0], {"outputs": [{"label": "normal", "score": 1}]})
        self.assertEqual(lines[1], {"outputs": {"value": 0}})

    def test_prediction_context_leaves_stdin_on_the_first_sample(self):
        from contest_harness.predict import serve_predictions
        from io import StringIO
        context = {"protocol": "contest.predict-context.v1", "dataset_key": "B", "task": "regression",
                   "artifact": {"checkpoint_paths": ["model.bin"]}, "artifact_id": "model-test-1"}
        seen = {}

        def predict(sample, model_dir):
            del model_dir
            seen["sample"] = sample
            return {"value": 1}

        previous = os.environ.get("CONTEST_PREDICTION_CONTEXT")
        os.environ["CONTEST_PREDICTION_CONTEXT"] = json.dumps(context)
        os.environ["CONTEST_DATASET_KEY"] = "B"
        os.environ["CONTEST_TASK"] = "regression"
        try:
            output = StringIO()
            serve_predictions(predict, reader=StringIO('{"task":"regression","inputs":{"x":1}}\n'), writer=output)
        finally:
            if previous is None:
                os.environ.pop("CONTEST_PREDICTION_CONTEXT", None)
                os.environ.pop("CONTEST_DATASET_KEY", None)
                os.environ.pop("CONTEST_TASK", None)
            else:
                os.environ["CONTEST_PREDICTION_CONTEXT"] = previous
        self.assertEqual(json.loads(output.getvalue()), {"outputs": {"value": 1}})
        self.assertEqual(seen["sample"].dataset_key, "B")
        self.assertEqual(seen["sample"].inputs, {"x": 1})

    def test_predict_rejects_ambiguous_and_negative_outputs(self):
        from contest_harness.predict import normalize_output
        from contest_harness.errors import HarnessError
        with self.assertRaises(HarnessError):
            normalize_output("classification", [{"label": "a", "score": 0.5}, {"label": "b", "score": 0.5}])
        with self.assertRaises(HarnessError):
            normalize_output("regression", {"value": -1})

    def test_streamed_upload_checks_chunks_and_mutation(self):
        import base64
        from contest_harness.round import Round
        from contest_harness.errors import HarnessError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'weights'
            original = b'a' * (1024 * 1024 + 7)
            path.write_bytes(original)
            rnd = Round(start_frame(root, root), None, None)
            calls = []
            def rpc(request_type, expect, **fields):
                calls.append((request_type, fields))
                if request_type == 'upload_begin_request':
                    self.assertEqual(fields['files'][0]['sha256'], 'sha256:' + hashlib.sha256(original).hexdigest())
                    return {'upload_id': 'u'}
                if request_type == 'upload_chunk_request':
                    data = base64.b64decode(fields['data_base64'])
                    self.assertLessEqual(len(data), 1024 * 1024)
                    return {'offset': fields['offset'] + len(data)}
                return {'upload_id': 'u'}
            rnd._rpc = rpc
            self.assertEqual(rnd._upload(root, 'candidate'), 'u')
            self.assertEqual([fields['offset'] for kind, fields in calls if kind == 'upload_chunk_request'], [0, 1024 * 1024])
            def mutate(request_type, expect, **fields):
                if request_type == 'upload_begin_request':
                    path.write_bytes(b'b' * len(original))
                return rpc(request_type, expect, **fields)
            rnd._rpc = mutate
            with self.assertRaises(HarnessError) as caught:
                rnd._upload(root, 'candidate')
            self.assertEqual(caught.exception.code, 'UPLOAD_DIGEST_MISMATCH')
            rnd.limits = {}
            with self.assertRaises(HarnessError) as caught:
                rnd._upload(root, 'candidate')
            self.assertEqual(caught.exception.code, 'CHUNKED_UPLOAD_UNSUPPORTED')

    def test_parent_file_digest_is_checked(self):
        from contest_harness.round import Round
        from contest_harness.errors import HarnessError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = b"abc"
            (root / "checkpoint.npz").write_bytes(payload)
            digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            frame = start_frame(root / "missing-train.jsonl", root / "missing-validation.jsonl")
            frame["parent_checkpoint_sha256"] = digest
            frame["parent_artifact"]["files"][0]["sha256"] = digest
            frame["parent_artifact"]["files"][0]["size_bytes"] = len(payload)
            rnd = Round(frame, None, None)
            self.assertEqual(rnd.parent.path("checkpoint.npz", root).read_bytes(), payload)
            (root / "checkpoint.npz").write_bytes(b"nope")
            with self.assertRaises(HarnessError) as caught:
                rnd.parent.path("checkpoint.npz", root)
            self.assertEqual(caught.exception.code, "PARENT_DIGEST_MISMATCH")

    def test_transport_forwards_one_nonstreaming_call(self):
        import urllib.request
        from contest_harness.round import Round
        calls = []

        def model(messages, options):
            calls.append((messages, options))
            return {"request_id": "call-transport", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        frame = start_frame(Path("unused-train"), Path("unused-validation"))
        reader_host, writer_guest = os.pipe()
        reader_guest, writer_host = os.pipe()
        guest_in = os.fdopen(reader_guest, "r", encoding="utf-8")
        guest_out = os.fdopen(writer_guest, "w", encoding="utf-8")
        host_in = os.fdopen(reader_host, "r", encoding="utf-8")
        host_out = os.fdopen(writer_host, "w", encoding="utf-8")
        rnd = Round(frame, guest_in, guest_out)
        host = Host(frame, model, send_start=False)
        class PipeProcess:
            def __init__(self):
                self.stdin = host_out
                self.stdout = host_in
            def kill(self):
                return None
        thread = host.serve(PipeProcess())
        try:
            with rnd.transport() as transport:
                body = json.dumps({
                    "model": frame["model_name"],
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 16,
                }).encode()
                request = urllib.request.Request(transport.base_url + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    payload = json.load(response)
            self.assertEqual(payload["choices"][0]["message"]["content"], "ok")
            self.assertEqual(rnd.call_ids, ["call-transport"])
            self.assertEqual(calls[0][0][1]["content"], "ping")
        finally:
            thread.join(timeout=5)
            guest_in.close()
            guest_out.close()
            host_in.close()
            host_out.close()

    def test_unfinished_round_is_a_participant_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "forget_submission.py").write_text(textwrap.dedent("""
                def train(rnd):
                    rnd.history()
            """), encoding="utf-8")
            spec_env = self.env(root, submission="forget_submission")
            spec_env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(root)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            train.write_text('{"id":"s1","inputs":{},"answer":"a"}\n', encoding="utf-8")
            validation.write_text('{"id":"v1","inputs":{},"answer":"a"}\n', encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "template" / "harness.py"), "train"],
                cwd=root, env=spec_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            host = Host(start_frame(train, validation), stub_model)
            thread = host.serve(process)
            stderr = process.stderr.read()
            code = process.wait(timeout=20)
            thread.join(timeout=5)
        self.assertIn("ROUND_NOT_FINISHED", stderr)
        self.assertNotEqual(code, 0)
        self.assertIsInstance(host.failure, RuntimeError)


if __name__ == "__main__":
    unittest.main()
