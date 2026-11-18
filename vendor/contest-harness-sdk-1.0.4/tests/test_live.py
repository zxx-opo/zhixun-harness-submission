"""Live checks. They call PyTRIO only when PYTRIO_API_KEY is set, and slim pi only when CONTEST_PI points at pi.mjs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.support import Host  # noqa: E402
from tests.test_sdk import PROMPT, start_frame  # noqa: E402


def provider_model(base_url, key):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/models?limit=100&offset=0",
        headers={"Authorization": "Bearer " + key, "X-API-Key": key},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    names = [item.get("model_id") for item in payload.get("items", []) if isinstance(item, dict)]
    selected = next((name for name in names if isinstance(name, str) and "4B" in name), None)
    if selected is None:
        selected = next((name for name in names if isinstance(name, str)), None)
    if not selected:
        raise RuntimeError("no model available to this key")
    return selected


def call_provider(base_url, key, model, messages):
    body = json.dumps({
        "model": model, "messages": messages, "max_tokens": 32, "temperature": 0, "stream": False,
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/openai/v1/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not payload.get("choices"):
        raise RuntimeError("provider returned no choices")
    payload["request_id"] = "pytrio-" + str(payload.get("id", "call"))
    return payload


class LiveSdkTest(unittest.TestCase):
    def test_real_model_call_sees_official_prompt_once(self):
        key = os.environ.get("PYTRIO_API_KEY", "")
        if not key:
            self.skipTest("PYTRIO_API_KEY is not set")
        base = os.environ.get("PYTRIO_BASE_URL", "https://pytrio.com")
        model_name = provider_model(base, key)
        seen = {}

        def model(messages, _options):
            seen["messages"] = messages
            return call_provider(base, key, model_name, messages)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            train.write_text('{"id":"s1","inputs":{"x":1},"answer":"normal"}\n', encoding="utf-8")
            validation.write_text('{"id":"v1","inputs":{"x":1},"answer":"normal"}\n', encoding="utf-8")
            frame = start_frame(train, validation)
            frame["model_name"] = model_name
            frame["prompt"] = PROMPT
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "example")
            env["CONTEST_SUBMISSION"] = "contest_submission"
            env["CONTEST_WORK"] = str(root / "work")
            (root / "work").mkdir()
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "template" / "harness.py"), "train"],
                cwd=ROOT / "example", env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
            host = Host(frame, model, history=[])
            thread = host.serve(process)
            stderr = process.stderr.read()
            code = process.wait(timeout=90)
            thread.join(timeout=5)
        self.assertEqual(code, 0, stderr)
        self.assertIsNone(host.failure, host.failure)
        content = seen["messages"][0]["content"]
        self.assertTrue(content.startswith(PROMPT), content[:80])
        self.assertEqual(content.count(PROMPT), 1)
        self.assertEqual(seen["messages"][1]["role"], "user")
        self.assertNotIn(PROMPT, seen["messages"][1]["content"])
        self.assertEqual(host.record["model_call_ids"], [host.record["model_call_ids"][0]])

    def test_slim_pi_uses_sdk_transport(self):
        key = os.environ.get("PYTRIO_API_KEY", "")
        pi = os.environ.get("CONTEST_PI", "")
        if not key or not pi:
            self.skipTest("PYTRIO_API_KEY and CONTEST_PI are required")
        base = os.environ.get("PYTRIO_BASE_URL", "https://pytrio.com")
        model_name = provider_model(base, key)

        def model(messages, _options):
            return call_provider(base, key, model_name, messages)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            train.write_text('{"id":"s1","inputs":{"x":1},"answer":"normal"}\n', encoding="utf-8")
            validation.write_text('{"id":"v1","inputs":{"x":1},"answer":"normal"}\n', encoding="utf-8")
            frame = start_frame(train, validation)
            frame["model_name"] = model_name
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "example")
            env["CONTEST_SUBMISSION"] = "pi_submission"
            env["CONTEST_WORK"] = str(root / "work")
            env["CONTEST_PI"] = pi
            (root / "work").mkdir()
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "template" / "harness.py"), "train"],
                cwd=ROOT / "example", env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
            host = Host(frame, model, history=[])
            thread = host.serve(process)
            stderr = process.stderr.read()
            code = process.wait(timeout=180)
            thread.join(timeout=5)
        self.assertEqual(code, 0, stderr)
        self.assertIsNone(host.failure, host.failure)
        self.assertTrue(host.bound_messages)
        self.assertTrue(host.bound_messages[0][0]["content"].startswith(PROMPT))
        self.assertTrue(host.record["model_call_ids"])


if __name__ == "__main__":
    unittest.main()
