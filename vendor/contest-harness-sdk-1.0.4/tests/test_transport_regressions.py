import json
import unittest
import urllib.request
from types import SimpleNamespace

from contest_harness.transport import model_transport


class FakeRound:
    model_name = "test-model"
    max_output_tokens = 128

    def __init__(self, response):
        self.response = response
        self.calls = []
        self.call_ids = []

    def model(self, messages, **options):
        self.calls.append((messages, options))
        return SimpleNamespace(response=self.response)


class TransportRegressionTests(unittest.TestCase):
    def request(self, response, **options):
        rnd = FakeRound(response)
        messages = [{"role": "user", "content": "hello"}]
        request_body = {"model": rnd.model_name, "messages": messages, **options}
        with model_transport(rnd) as transport:
            request = urllib.request.Request(
                transport.base_url + "/chat/completions",
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as result:
                content_type = result.headers.get_content_type()
                body = result.read().decode("utf-8")
        self.assertEqual(rnd.calls, [(messages, {"temperature": 0})])
        return content_type, body

    def test_stream_preserves_choices_tools_envelope_and_requested_usage(self):
        response = {
            "id": "completion-1", "model": "test-model", "created": 123,
            "object": "chat.completion",
            "choices": [
                {"index": 3, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "reasoning_content": "thinking",
                    "tool_calls": [
                        {"id": "tool-a", "type": "function", "function": {"name": "first", "arguments": "{}"}},
                        {"id": "tool-b", "type": "function", "function": {"name": "second", "arguments": "{\"x\":1}"}},
                    ],
                }},
                {"index": 7, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "answer", "refusal": "none",
                }},
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
        }
        content_type, body = self.request(
            response, stream=True, stream_options={"include_usage": True}, temperature=0,
        )
        self.assertEqual(content_type, "text/event-stream")
        frames = body.strip().split("\n\n")
        self.assertEqual(frames[-1], "data: [DONE]")
        events = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-1]]
        self.assertEqual(len(events), 3)
        for event in events:
            for key in ("id", "model", "created", "object"):
                self.assertEqual(event[key], response[key])
        self.assertEqual([choice["index"] for choice in events[0]["choices"]], [3, 7])
        first, second = events[0]["choices"]
        self.assertEqual(first["delta"]["reasoning_content"], "thinking")
        self.assertIsNone(first["delta"]["content"])
        self.assertEqual(second["delta"], response["choices"][1]["message"])
        self.assertEqual(first["delta"]["tool_calls"], [
            {**call, "index": index}
            for index, call in enumerate(response["choices"][0]["message"]["tool_calls"])
        ])
        self.assertEqual(events[1]["choices"], [
            {"index": 3, "delta": {}, "finish_reason": "tool_calls"},
            {"index": 7, "delta": {}, "finish_reason": "stop"},
        ])
        self.assertEqual(events[2]["choices"], [])
        self.assertEqual(events[2]["usage"], response["usage"])

    def test_stream_without_usage_and_nonstream_each_call_model_once(self):
        response = {
            "id": "single", "model": "test-model", "created": 42, "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 2},
        }
        for options in ({}, {"stream_options": {"include_usage": False}}):
            with self.subTest(options=options):
                content_type, body = self.request(response, stream=True, temperature=0, **options)
                self.assertEqual(content_type, "text/event-stream")
                frames = body.strip().split("\n\n")
                self.assertEqual(frames[-1], "data: [DONE]")
                events = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-1]]
                self.assertEqual(len(events), 2)
                self.assertTrue(all("usage" not in event for event in events))
                self.assertEqual(events[0]["choices"][0]["delta"], response["choices"][0]["message"])
                self.assertEqual(events[1]["choices"][0]["finish_reason"], "stop")
        content_type, body = self.request(response, stream=False, temperature=0)
        self.assertEqual(content_type, "application/json")
        self.assertEqual(json.loads(body), response)

    def test_missing_finish_reason_still_finishes_each_choice(self):
        choices = [
            {"index": 0, "message": {"content": "done"}},
            {"index": 1, "message": {"tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]}, "finish_reason": None},
            {"index": 2, "message": {"content": "truncated"}, "finish_reason": "length"},
            {"index": 3, "message": {"content": "done"}, "finish_reason": "stop"},
        ]
        _, body = self.request({"choices": choices}, stream=True, temperature=0)
        frames = body.strip().split("\n\n")
        self.assertEqual(frames[-1], "data: [DONE]")
        events = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-1]]
        self.assertTrue(all(choice["finish_reason"] is None for choice in events[0]["choices"]))
        self.assertEqual([choice["finish_reason"] for choice in events[-1]["choices"]], ["stop", "tool_calls", "length", "stop"])
