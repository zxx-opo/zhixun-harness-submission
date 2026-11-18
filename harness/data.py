"""Registered public datasets and source-bound model loading."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import os

ROOT = Path(__file__).resolve().parent.parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, allow_nan=False, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path):
    with path.open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def code_version():
    paths = sorted((ROOT / "harness").glob("*.py")) + sorted(
        path for path in (ROOT / "adapters").glob("*.py") if not path.name.startswith("swanlab"))
    sources = {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


TASKS = read_json(ROOT / "configs/tasks.json")
BUDGET = read_json(ROOT / "configs/budget.json")


@dataclass(frozen=True)
class Dataset:
    task: str
    role: str
    path: Path
    sha256: str


def group_key(task, meta):
    if task == "A":
        return str(meta["source_batch"])
    if task == "B":
        return meta["bearing"]
    return (meta["source_subset"], meta["unit"])


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class Task:
    def __init__(self, task, split_dir=None):
        self.id = task
        self.spec = TASKS[task]
        self.root = ROOT / "vendor/reference" / task
        package = f"vendor.reference.{task}.baselines"
        self.model = importlib.import_module(f"{package}.model").Model
        self.reference_data = importlib.import_module(f"{package}.data")
        self.predict_one = importlib.import_module(f"{package}.predict").predict
        self.config_path = self.root / "baselines/configs" / f"{task}.json"
        self.profile = read_json(self.config_path)
        self.report = read_json(self.root / "training.json")
        self.info = read_json(self.root / "baseline-info.json")
        self.baseline = self.root / "checkpoint.npz"
        self.split_dir = split_dir
        self.cached = {}
        self.baseline_model = self.model.load(self.baseline)
        self.paths = self.baseline_model.meta["paths"]
        self.classes = self.baseline_model.meta["classes"]
        self.verify_source()

    def verify_source(self):
        m, spec, report = self.baseline_model.meta, self.spec, self.report
        checks = {
            "checkpoint": digest(self.baseline) == self.info["checkpoint_sha256"] == report["checkpoint_sha256"],
            "profile digest": digest(self.config_path) == report["profile_sha256"],
            "profile": m["profile"] == self.profile["profile"] == self.id,
            "task": m["task"] == self.profile["task"] == spec["type"],
            "classes": m["classes"] == self.profile.get("classes", []),
            "dimensions": len(m["paths"]) == spec["features"] == report["input_features"],
            "parameters": m["parameter_count"] == report["parameter_count"] == self.info["parameter_count"],
            "steps": m["steps"] == report["final_steps"] and m["epochs"] == report["final_epochs"],
            "status": report["purpose"] == "development_reference" and not report["official_scoring_ready"],
            "units": self.id != "B" or m.get("target_unit") == self.profile.get("target_unit") == spec["unit"],
            "C feature order": self.id != "C" or [json.loads(p)[0][1] for p in self.paths] == ["sensors"] * 21 + ["settings"] * 3,
        }
        checks.update({name: digest(self.root / "baselines" / name) == expected
                       for name, expected in report["source_sha256"].items()})
        if not all(checks.values()):
            raise ValueError(f"{self.id} source checks failed: {[name for name, passed in checks.items() if not passed]}")

    def dataset(self, role):
        reference_roles = {"train": "train", "validation": "validation"}
        if self.id != "C":
            reference_roles["development"] = "validation"
        if role in reference_roles:
            name = reference_roles[role]
            return Dataset(self.id, role, self.root / "public" / f"{name}.jsonl", self.report[f"{name}_sha256"])
        if self.id != "C" or self.split_dir is None or role not in {"development", "confirmation"}:
            raise ValueError(f"{self.id} has no registered {role} dataset")
        manifest = read_json(self.split_dir / "manifest.json")
        if manifest["source_sha256"] != self.report["validation_sha256"]:
            raise ValueError("C split manifest does not match the public validation source")
        return Dataset(self.id, role, self.split_dir / f"{role}.jsonl", manifest[role]["sha256"])

    def arrays(self, role):
        if role not in self.cached:
            dataset = self.dataset(role)
            if digest(dataset.path) != dataset.sha256:
                raise ValueError(f"registered {self.id}/{role} file changed")
            result = self.reference_data.read_dataset(dataset.path, self.spec["type"], self.classes, self.paths)
            if result["sha256"] != dataset.sha256:
                raise ValueError(f"registered {self.id}/{role} digest changed during read")
            for field in ("x", "y"):
                result[field].flags.writeable = False
            self.cached[role] = result
        return self.cached[role]

    def groups(self, role):
        key = f"groups:{role}"
        if key not in self.cached:
            data = self.arrays(role)
            dataset = self.dataset(role)
            # Source row order is checked before using metadata as an evaluation grouping.
            keys, ids = [], []
            for row in rows(dataset.path):
                ids.append(row["id"])
                keys.append(group_key(self.id, row["meta"]))
            if ids != data["ids"] or digest(dataset.path) != data["sha256"]:
                raise ValueError("group metadata does not match evaluated rows")
            self.cached[key] = keys
        return self.cached[key]

    def worker_cache_path(self, role):
        if role not in {"train", "development"}:
            raise ValueError("the worker cache only permits train and development roles")
        dataset = self.dataset(role)
        return self.split_dir.parent.parent / "arrays" / f"{self.id}-{role}-{dataset.sha256.removeprefix('sha256:')}.npz"

    def stage_worker_cache(self, role):
        import numpy as np
        path = self.worker_cache_path(role)
        data = self.arrays(role)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Rebuild from the verified source on process start, then reuse across candidate workers.
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, x=data["x"], y=data["y"])
            stream.flush()
            os.fsync(stream.fileno())
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return digest(path)

    def worker_arrays(self, role, ticket):
        import io
        import numpy as np
        path = self.worker_cache_path(role)
        dataset = self.dataset(role)
        if digest(dataset.path) != dataset.sha256:
            raise ValueError(f"registered {self.id}/{role} source changed")
        raw = path.read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != ticket:
            raise ValueError(f"{self.id}/{role} worker cache digest mismatch")
        with np.load(io.BytesIO(raw), allow_pickle=False) as cache:
            x, y = cache["x"], cache["y"]
        if x.shape != (len(y), len(self.paths)) or y.ndim != 1 or not len(y):
            raise ValueError("invalid cached dataset dimensions")
        return {"x": x, "y": y, "sha256": dataset.sha256}

    def load(self, path):
        model = self.model.load(path)
        reference = self.baseline_model.meta
        fields = ("profile", "task", "paths", "classes", "hidden", "parameter_count", "seed", "target_unit")
        if any(model.meta.get(key) != reference.get(key) for key in fields):
            raise ValueError(f"checkpoint does not match task {self.id}")
        return model


def prepare_c_split(task):
    """Stratify by FD and keep entire validation engines in one role."""
    source = task.dataset("validation")
    directory = task.split_dir
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        for role in ("development", "confirmation"):
            dataset = task.dataset(role)
            if digest(dataset.path) != dataset.sha256:
                raise ValueError(f"C {role} changed")
        return read_json(manifest_path)
    if digest(source.path) != source.sha256:
        raise ValueError("C validation source changed")
    devices = defaultdict(set)
    for row in rows(source.path):
        subset, unit = group_key("C", row["meta"])
        devices[subset].add(unit)
    confirmation = set()
    for subset, units in devices.items():
        ranked = sorted(units, key=lambda unit: hashlib.sha256(f"C-split-v1:42:{subset}:{unit}".encode()).digest())
        count = max(1, round(len(ranked) * 0.30))
        if count >= len(ranked):
            raise ValueError("each FD must have engines in both roles")
        confirmation.update((subset, unit) for unit in ranked[:count])
    development = {(subset, unit) for subset, units in devices.items() for unit in units} - confirmation
    directory.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    # Preserve exact source bytes and row order. A manifest is published only after both files exist.
    with (directory / "development.jsonl").open("wb") as dev, (directory / "confirmation.jsonl").open("wb") as confirm:
        with source.path.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                selected = group_key("C", row["meta"]) in confirmation
                (confirm if selected else dev).write(line)
                counts["confirmation" if selected else "development"] += 1
    manifest = {"version": "C-split-v1", "seed": 42, "source_sha256": source.sha256}
    for role, keys in (("development", development), ("confirmation", confirmation)):
        manifest[role] = {"engines": sorted(keys), "rows": counts[role], "sha256": digest(directory / f"{role}.jsonl")}
    atomic_json(manifest_path, manifest)
    return manifest
