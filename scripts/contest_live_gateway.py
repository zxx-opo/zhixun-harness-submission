"""PyTRiO bridge used only by the local acceptance host."""
import json
import os
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import monotonic

from adapters.trainer import DEFAULT_BASE_URL, DEFAULT_MODEL_ID, PyTrioTransport
from harness.data import atomic_json, read_json
from scripts.paid_budget_v12 import PaidBudget
from scripts.sdk_response_v12 import completion_text


PRICES = {"prefill": 8.41, "sample": 25.29}
PREFILL_LIMIT = 8192
PROVIDER_TIMEOUT = 175
MODEL = DEFAULT_MODEL_ID


class LiveGateway:
    def __init__(self, directory, ledger_directory, run_id):
        self.directory = directory
        self.transport = PyTrioTransport(DEFAULT_BASE_URL, MODEL, os.environ.get("PYTRIO_API_KEY", "").strip())
        self.transport.thinking_mode = "prefix"
        self.budget = PaidBudget(ledger_directory, 30)
        self.run_id = run_id
        self.mode = os.environ.get("CONTEST_104_PYTRIO_TRANSPORT", "openai")
        if self.mode not in ("openai", "sdk"):
            raise ValueError("unknown PyTRiO transport")
        self.sampler = None
        self.directory.mkdir(parents=True, exist_ok=True)

    def sample_sdk(self, context, max_tokens, instruction):
        import pytrio as trio
        from transformers import AutoTokenizer

        if version("pytrio") != "0.2.9":
            raise ValueError("PyTRiO SDK lock mismatch")
        if self.sampler is None:
            default = Path(__file__).resolve().parents[1] / "runs/v12-preflight/tokenizer"
            self.tokenizer = AutoTokenizer.from_pretrained(os.environ.get("CONTEST_104_TOKENIZER", str(default)),
                                                           local_files_only=True)
            service = trio.ServiceClient(api_key=os.environ["PYTRIO_API_KEY"])
            self.sampler = service.create_sampling_client(base_model=MODEL)
            if self.sampler.model_id != MODEL:
                raise ValueError("PyTRiO resolved a different model")
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": instruction},
             {"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = self.tokenizer.encode(formatted, add_special_tokens=False)
        if not 0 < len(tokens) <= PREFILL_LIMIT:
            raise ValueError("tokenized summary exceeds local limit")
        # The locked SDK's public SampleResponse drops the Actor request ID.
        future = self.sampler._event_loop.run_sync(self.sampler._submit_sample(
            prompt=trio.ModelInput.from_ints(tokens), num_samples=1,
            sampling_params=trio.SamplingParams(max_tokens=max_tokens, temperature=0, seed=42),
            include_prompt_logprobs=False, topk_prompt_logprobs=0, return_token_logprobs=False))
        actor = future.result(timeout=140)
        output = self.sampler._parse_sample_response(actor)
        result = output.model_dump(mode="json")
        if not actor.request_id or len(result["sequences"]) != 1 or output.input_tokens != len(tokens):
            raise ValueError("provider sampling receipt did not match the tokenized request")
        return {"content": completion_text(self.tokenizer, result["sequences"][0]),
                "usage": {"prompt_tokens": output.input_tokens, "completion_tokens": output.output_tokens,
                          "total_tokens": output.input_tokens + output.output_tokens},
                "provider": {"request_id": self.transport.redact(actor.request_id), "model": MODEL,
                             "endpoint": "pytrio-sdk-sample", "sdk_version": "0.2.9",
                             "sampling_run": self.transport.redact(self.sampler.sampling_run_cuid),
                             "finish_reason": result["sequences"][0].get("stop_reason")}}

    def model(self, frame, msg):
        messages = msg["messages"]
        if [item["role"] for item in messages] != ["system", "user"] or not 1 <= msg["max_tokens"] <= 224:
            raise ValueError("unexpected local gateway request")
        meta = {k: frame[k] for k in ("run_id", "dataset_key", "round_index", "stage_id", "prompt_version", "task_contract")}
        instruction = frame["prompt"] + "\n" + json.dumps(meta, sort_keys=True) + "\n\n" + messages[0]["content"]
        context = json.loads(messages[1]["content"])
        if frame["prompt"] in messages[0]["content"] or not isinstance(context, dict):
            raise ValueError("unexpected prompt duplication or evidence")
        request = {"model": MODEL, "instruction": instruction, "facts": context, "max_tokens": msg["max_tokens"]}
        signature = sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
        identity = f"{self.run_id}-{frame['dataset_key']}-{frame['round_index']:02d}"
        target = self.directory / (identity + ".json")
        recovered_from = None
        if target.exists():
            previous = read_json(target)
            if previous["request_sha256"] != signature:
                raise ValueError("previous gateway receipt does not match this round")
            if previous["status"] == "completed":
                return previous
            retry = self.directory / (identity + "-retry1.json")
            if retry.exists():
                outcome = read_json(retry)
                if outcome["status"] != "completed" or outcome["request_sha256"] != signature:
                    raise ValueError("recorded provider retry did not complete")
                return outcome
            transient = (previous.get("error_type") in ("RemoteDisconnected", "TimeoutError", "ConnectionResetError")
                         or previous.get("error_kind") in ("timeout", "temporary", "unreachable")
                         or previous.get("http_status") in (429, 500, 502, 503, 504)
                         or (previous.get("error_type") == "TrainerServiceError" and previous.get("http_status") is None
                             and previous.get("seconds", 0) >= 75))
            if not transient or os.environ.get("CONTEST_104_RETRY_REQUEST") != identity:
                raise ValueError("reconcile the prior provider attempt before any annotated retry")
            recovered_from, identity, target = identity, identity + "-retry1", retry
        if not self.budget.path.exists():
            raise ValueError("the existing authorized project cost ledger is required")
        if len(instruction.encode()) + len(json.dumps(context, ensure_ascii=False).encode()) + 256 > PREFILL_LIMIT:
            raise ValueError("excess gateway input volume")
        held = self.budget.reserve(identity, {"prefill": PREFILL_LIMIT, "sample": msg["max_tokens"]}, PRICES)
        start = monotonic()
        try:
            if self.mode == "sdk":
                response = self.sample_sdk(context, msg["max_tokens"], instruction)
            else:
                response = self.transport(context, PROVIDER_TIMEOUT, msg["max_tokens"], instruction=instruction)
            provider = response["provider"]
            upstream_id = provider.get("request_id") or provider.get("id")
            if not upstream_id or provider.get("model") not in (None, MODEL):
                raise ValueError("missing real provider ID or model mismatch")
            content = response["content"]
            if not isinstance(content, str) or len(content) > 4096:
                raise ValueError("invalid upstream response shape")
            usage = response.get("usage") or {}
            costs = {"prefill": usage["prompt_tokens"], "sample": usage["completion_tokens"]} if all(
                k in usage for k in ("prompt_tokens", "completion_tokens")) else None
            accounting = self.budget.finish(identity, actual=costs, metadata={"model": MODEL, "seconds": monotonic() - start,
                                                                            "provider_request_id": upstream_id})
            entry = next(item for item in accounting["entries"] if item["id"] == identity)
            proof = {"id": identity, "request_sha256": signature, "request": request, "status": "completed",
                     "recovered_from": recovered_from,
                     "provider": provider, "call_id": "local-pytrio-" + sha256(upstream_id.encode()).hexdigest()[:24],
                     "response": {"choices": [{"message": {"content": self.transport.redact(content)},
                                               "finish_reason": provider.get("finish_reason")}], "usage": usage},
                     "seconds": monotonic() - start, "reserved_cny": held, "charged_upper_cny": entry["charged_upper_cny"]}
        except Exception as exc:
            metadata = {"error_type": type(exc).__name__, "error_kind": getattr(exc, "kind", None),
                        "http_status": getattr(exc, "status", None),
                        "seconds": monotonic() - start}
            self.budget.finish(identity, status="failed_or_ambiguous", metadata=metadata)
            atomic_json(target, {"id": identity, "request_sha256": signature, "status": "failed_or_ambiguous", **metadata})
            raise
        atomic_json(target, proof)
        return proof
