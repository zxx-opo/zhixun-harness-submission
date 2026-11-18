"""Structured trainer boundary, PyTRiO chat transport and deterministic test transport."""

import json
import os
import time
from hashlib import sha256
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


_HTTPS = build_opener(NoRedirects())
_MAX_RESPONSE_BYTES = 256 * 1024
DEFAULT_BASE_URL = "https://pytrio.com/api/openai/v1"
DEFAULT_MODEL_ID = "Qwen/Qwen3.8-27B"
PROMPT_VERSION = "zhixun-proposal.1"
_PROPOSAL_INSTRUCTION = (
    "You are the experiment planner for an industrial model Harness. Choose exactly one recipe "
    "from available_recipes. Return one bare JSON object with exactly these fields: "
    'schema_version="proposal.v1", task_id=task, action="continue_train", parent_sha256=parent, '
    "recipe, evidence_refs (one or more exact IDs copied from the supplied evidence_refs list), "
    "and rationale_summary (one short sentence). Use the metric, observed results and remaining budget. "
    "The recipe is the only action you may choose. Copy identifiers exactly."
)


class TrainerServiceError(OSError):
    def __init__(self, kind, status=None, request_id=None, retry_after=None):
        super().__init__(f"PyTRiO HTTP {status}" if status is not None else f"PyTRiO {kind}")
        self.kind = kind
        self.status = status
        self.request_id = request_id
        self.retry_after = retry_after


class PyTrioTransport:
    def __init__(self, base_url, model_id, api_key):
        url = urlsplit(base_url.rstrip("/"))
        if (url.scheme != "https" or not url.hostname or url.username is not None or url.password is not None
                or url.query or url.fragment or any(char.isspace() for char in base_url)):
            raise ValueError("PYTRIO_OPENAI_BASE_URL must be an HTTPS base URL without credentials, query or fragment")
        if url.path.endswith("/chat/completions"):
            raise ValueError("PYTRIO_OPENAI_BASE_URL must end at the API base path, before /chat/completions")
        if not model_id or len(model_id) > 256 or any(char.isspace() for char in model_id):
            raise ValueError("PYTRIO_MODEL_ID must be the exact model ID from the contest provider")
        if not api_key or any(not 33 <= ord(char) <= 126 for char in api_key):
            raise ValueError("PYTRIO_API_KEY is missing or malformed")
        self.base_url = url.geturl()
        self.model_id = model_id
        self._api_key = api_key
        self.thinking_mode = os.environ.get("PYTRIO_THINKING_MODE", "prefix").strip().lower()
        if self.thinking_mode not in {"prefix", "template", "low", "default"}:
            raise ValueError("PYTRIO_THINKING_MODE must be prefix, template, low or default")

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get("PYTRIO_OPENAI_BASE_URL", DEFAULT_BASE_URL).strip(),
                   os.environ.get("PYTRIO_MODEL_ID", DEFAULT_MODEL_ID).strip(),
                   os.environ.get("PYTRIO_API_KEY", "").strip())

    def __call__(self, context, timeout_seconds, max_tokens, *, instruction=_PROPOSAL_INSTRUCTION):
        context_text = json.dumps(context, ensure_ascii=False, allow_nan=False)
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": context_text},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False,
        }
        api = "chat/completions"
        if self.thinking_mode == "prefix":
            # The PyTRiO chat endpoint rejects chat_template_kwargs. Its documented text
            # endpoint can accept the pretrained Qwen chat template with a closed thinking block.
            if any(marker in instruction + context_text for marker in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")):
                raise ValueError("unexpected control token in the authorized summary")
            api = "completions"
            payload.pop("messages")
            payload["prompt"] = (f"<|im_start|>system\n{instruction}<|im_end|>\n"
                                 f"<|im_start|>user\n{context_text}<|im_end|>\n"
                                 "<|im_start|>assistant\n<think>\n\n</think>\n\n")
            payload["stop"] = ["<|im_end|>"]
        elif self.thinking_mode == "template":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.thinking_mode == "low":
            payload["reasoning_effort"] = "low"
        prefix_fingerprint = payload.get("prompt") or (instruction + "\n" + context_text)
        request = Request(f"{self.base_url}/{api}", data=json.dumps(payload).encode("utf-8"),
                          headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                          method="POST")
        try:
            with _HTTPS.open(request, timeout=timeout_seconds) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                request_id = getattr(response, "headers", {}).get("X-Request-ID")
        except HTTPError as exc:
            try:
                retry_after = exc.headers.get("Retry-After")
                fragment = exc.read(4096).lower() if exc.fp is not None else b""
            finally:
                exc.close()
            if exc.code == 402 or any(word in fragment for word in (b"insufficient_quota", b"insufficient balance", b"quota exhausted")):
                kind = "quota"
            elif exc.code in (401, 403):
                kind = "auth"
            elif exc.code == 429 or exc.code >= 500:
                kind = "temporary"
            else:
                kind = "request_rejected"
            raise TrainerServiceError(kind, exc.code, exc.headers.get("X-Request-ID"), retry_after) from None
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TrainerServiceError("timeout") from None
            raise TrainerServiceError("unreachable") from None
        except TimeoutError:
            raise TrainerServiceError("timeout") from None
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("PyTRiO response exceeded size limit")
        try:
            response = json.loads(raw)
            choice = response["choices"][0]
            message = {"content": choice["text"]} if api == "completions" else choice["message"]
            if not isinstance(message, dict):
                raise ValueError("expected a chat message object")
            content = message.get("content")
        except (ValueError, KeyError, IndexError, TypeError):
            raise ValueError("invalid PyTRiO chat response") from None
        reported_usage = response.get("usage")
        usage = {}
        if isinstance(reported_usage, dict):
            usage = {name: reported_usage[name] for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                     if type(reported_usage.get(name)) is int and reported_usage[name] >= 0}
        metadata = {}
        for field in ("model", "id"):
            value = response.get(field)
            if isinstance(value, str) and 0 < len(value) <= 256:
                metadata[field] = value
        if isinstance(request_id, str) and len(request_id) <= 256:
            metadata["request_id"] = request_id
        # The provider may embed an account/key prefix in its completion ID.
        for field, value in metadata.items():
            metadata[field] = self.redact(value)
        metadata.update(endpoint=api, finish_reason=choice.get("finish_reason"),
                        content_length=len(content) if isinstance(content, str) else 0,
                        provider_reasoning_field=bool(message.get("reasoning_content") or message.get("reasoning")))
        return {"content": content, "usage": usage or None, "provider": metadata,
                "prompt_sha256": sha256(prefix_fingerprint.encode()).hexdigest()}

    def redact(self, value):
        value = value.replace(self._api_key, "[redacted]")
        prefix = self._api_key.removeprefix("trio_")[:12]
        return value.replace(prefix, "[redacted]") if len(prefix) >= 12 else value


def mock_transport(context, timeout_seconds, max_tokens):
    return {"content": json.dumps({
        "schema_version": "proposal.v1", "task_id": context["task"], "action": "continue_train",
        "parent_sha256": context["parent"], "recipe": context["available_recipes"][0],
        "evidence_refs": context["evidence_refs"][:2],
        "rationale_summary": "Local mock selected the first recipe from the ranked options.",
    }), "usage": None}


def validate(result, context):
    if not isinstance(result, dict) or not isinstance(result.get("content"), str) or len(result["content"]) > 4096:
        raise ValueError("invalid trainer response")
    value = json.loads(result["content"])
    required = {"schema_version", "task_id", "action", "parent_sha256", "recipe", "evidence_refs", "rationale_summary"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("invalid proposal fields")
    if (value["schema_version"], value["task_id"], value["action"], value["parent_sha256"]) != (
        "proposal.v1", context["task"], "continue_train", context["parent"]
    ):
        raise ValueError("proposal targets a stale or forbidden action")
    if value["recipe"] not in context["available_recipes"]:
        raise ValueError("proposal recipe is unavailable")
    refs = value["evidence_refs"]
    if not isinstance(refs, list) or not refs or any(ref not in context["evidence_refs"] for ref in refs):
        raise ValueError("proposal cites unknown evidence")
    rationale = value["rationale_summary"]
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 500:
        raise ValueError("invalid rationale summary")
    return value


def probe(timeout=40, max_tokens=1000):
    transport = PyTrioTransport.from_environment()
    context = {"task": "A", "parent": "probe-checkpoint", "available_recipes": ["small"],
               "evidence_refs": ["synthetic.probe"]}
    started = time.monotonic()
    facts = {"scope": "synthetic_probe", "requested_model": transport.model_id, "request_count": 1,
             "thinking_mode": transport.thinking_mode}
    try:
        response = transport(context, timeout, max_tokens)
        facts.update(provider=response["provider"], usage=response["usage"], prompt_sha256=response["prompt_sha256"])
        proposal = validate(response, context)
        facts.update(status="proposal_valid", recipe=proposal["recipe"])
    except TrainerServiceError as exc:
        facts.update(status="failed", error_kind=exc.kind, http_status=exc.status,
                     request_id=transport.redact(exc.request_id) if exc.request_id else None,
                     retry_after=exc.retry_after)
    except (OSError, ValueError, TypeError) as exc:
        facts.update(status="failed", error_kind=type(exc).__name__, detail=str(exc))
    facts["wall_seconds"] = time.monotonic() - started
    return facts
