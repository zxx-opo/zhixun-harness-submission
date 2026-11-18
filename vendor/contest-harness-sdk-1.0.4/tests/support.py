"""In-process stand-in for the workspace v2 host."""
import json
import threading


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def bind_task_prompt(messages, prompt):
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        first = dict(messages[0])
        content = first.get("content")
        if isinstance(content, str):
            first["content"] = prompt + "\n\n" + content
        else:
            raise RuntimeError("unexpected system content")
        return [first, *messages[1:]]
    return [{"role": "system", "content": prompt}, *messages]


class Host:
    def __init__(self, start, model, history=(), send_start=True):
        self.start = start
        self.model = model
        self.history = list(history)
        self.send_start = send_start
        self.raw_messages = []
        self.bound_messages = []
        self.record = None
        self.failure = None

    def serve(self, process):
        def run():
            try:
                self._serve(process)
            except Exception as exc:
                self.failure = exc
                try:
                    process.kill()
                except OSError:
                    pass
        thread = threading.Thread(target=run)
        thread.start()
        return thread

    def _serve(self, process):
        if self.send_start:
            process.stdin.write(json.dumps(self.start, ensure_ascii=False) + "\n")
            process.stdin.flush()
        uploads = {}
        sequence = 0
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("participant closed stdout")
            frame = json.loads(line)
            kind = frame.get("type")
            if kind == "round_failed":
                raise RuntimeError("participant failed: " + str(frame.get("error")))
            if kind == "round_complete":
                self.record = frame["record"]
                process.stdin.write(json.dumps({"type": "round_ack", "accepted": True}) + "\n")
                process.stdin.flush()
                process.stdin.close()
                return
            if frame.get("id") != sequence:
                raise RuntimeError("rpc sequence mismatch")
            sequence += 1
            if kind == "history_request":
                reply = {"type": "history_response", "id": frame["id"], "records": self.history}
            elif kind == "model_request":
                messages = frame.get("messages")
                self.raw_messages.append(messages)
                metadata = {key: self.start[key] for key in ("run_id", "dataset_key", "round_index", "stage_id", "prompt_version", "task_contract") if key in self.start}
                bound = bind_task_prompt(messages, self.start["prompt"] + "\n" + encoded(metadata).decode())
                self.bound_messages.append(bound)
                options = {key: value for key, value in frame.items() if key not in ("type", "id", "messages")}
                response = self.model(bound, options)
                call_id = response.pop("request_id")
                reply = {"type": "model_response", "id": frame["id"], "call_id": call_id, "response": response}
            elif kind == "upload_begin_request":
                upload_id = str(len(uploads))
                uploads[upload_id] = {item['path']: bytearray() for item in frame['files']}
                reply = {"type": "upload_begin_response", "id": frame["id"], "upload_id": upload_id}
            elif kind == "upload_chunk_request":
                import base64
                data = uploads[frame['upload_id']][frame['path']]
                if frame['offset'] != len(data):
                    raise RuntimeError('upload offset mismatch')
                data.extend(base64.b64decode(frame['data_base64'], validate=True))
                reply = {"type": "upload_chunk_response", "id": frame["id"], "offset": len(data)}
            elif kind == "upload_finish_request":
                reply = {"type": "upload_finish_response", "id": frame["id"], "upload_id": frame['upload_id']}
            elif kind == "state_request":
                reply = {"type": "state_response", "id": frame["id"], "state_sha256": "sha256:" + "11" * 32}
            elif kind == "candidate_request":
                if frame.get("parent_checkpoint_sha256") != self.start["parent_checkpoint_sha256"]:
                    raise RuntimeError("parent binding changed")
                reply = {"type": "candidate_response", "id": frame["id"], "artifact_id": "model-test-1", "summary": {"metrics": {}}}
            else:
                raise RuntimeError("unexpected frame " + str(kind))
            process.stdin.write(json.dumps(reply, ensure_ascii=False) + "\n")
            process.stdin.flush()
