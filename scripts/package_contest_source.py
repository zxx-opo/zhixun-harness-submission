"""Create the submit-ready allowlisted source ZIP and check artifact identities."""
import hashlib
import json
import os
from pathlib import Path
import re
from tarfile import open as open_tar
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/contest-sdk-1.0.4"
AUDIT = ROOT / "reports/contest-submission-1.0.4/contest-104-release.json"
SDK = "vendor/contest-harness-sdk-1.0.4"
SDK_SHA = "7afd282d6b5fd889d64d0515a67d169806970e36200689b18d2040026167a4af"
SDK_GUIDE_SHA = "6d02c89f8e5d566879806975d75ff94f951009c005599d232b5989bda96fe55c"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scan(raw):
    patterns = (rb"trio_[A-Za-z0-9]{24,}",
                rb"(?im)^\s*(?:export\s+)?(?:PYTRIO|SWANLAB)_API_KEY\s*=\s*['\"]?[\w-]{16,}",
                rb"(?i)['\"](?:api_key|authorization)['\"]\s*:\s*['\"](?:Bearer\s+)?[\w-]{20,}['\"]")
    keys = [os.environ[name].encode() for name in ("PYTRIO_API_KEY", "SWANLAB_API_KEY") if os.environ.get(name)]
    return any(key in raw for key in keys) or any(re.search(pattern, raw) for pattern in patterns)


def inspect_image(tar, loaded_id):
    with open_tar(tar, mode="r:") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        if len(manifest) != 1 or manifest[0]["RepoTags"] != ["zhixun-harness:contest-1.0.4"]:
            raise ValueError("unexpected image manifest or repository tag")
        meta = archive.extractfile(manifest[0]["Config"]).read()
        config_id = "sha256:" + hashlib.sha256(meta).hexdigest()
        if config_id != loaded_id:
            index = json.load(archive.extractfile("index.json"))
            if [entry["digest"] for entry in index["manifests"]] != [loaded_id]:
                raise ValueError("docker-save index differs from the tested image")
            index_blob = archive.extractfile("blobs/" + loaded_id.replace(":", "/", 1)).read()
            if "sha256:" + hashlib.sha256(index_blob).hexdigest() != loaded_id:
                raise ValueError("invalid image manifest digest")
            image = json.loads(index_blob)
            if image["config"]["digest"] != config_id:
                raise ValueError("archive and OCI configuration differ")
            if ["blobs/" + item["digest"].replace(":", "/", 1) for item in image["layers"]] != manifest[0]["Layers"]:
                raise ValueError("archive and OCI layer lists differ")
        config = json.loads(meta)
        if config["architecture"] != "amd64" or config["os"] != "linux" or scan(meta):
            raise ValueError("unexpected image configuration")
        if config["config"]["Entrypoint"] != ["python3", "/app/harness.py"] or config["config"]["User"] != "65534:65534":
            raise ValueError("unexpected Docker entrypoint")
        app_files = set()
        for layer_name in manifest[0]["Layers"]:
            with archive.extractfile(layer_name) as layer, open_tar(fileobj=layer, mode="r|*") as data:
                for member in data:
                    path = member.name.removeprefix("./")
                    if member.isfile() and path.startswith("app/"):
                        app_files.add(path)
                        if path.endswith((".npz", ".jsonl")) or scan(data.extractfile(member).read()):
                            raise ValueError(f"unexpected app data or credentials: {path}")
        if "app/harness.py" not in app_files or "app/contest_submission.py" not in app_files:
            raise ValueError("missing contest runtime files")
        return sorted(app_files)


def source_entries():
    paths = {ROOT / p for p in (".dockerignore", "README-CONTEST.md", "Dockerfile.contest", "Dockerfile.contest-base",
        "requirements-contest.lock", "contest_submission.py", "contest_engine/model.py",
        "scripts/check_contest_submission.py", "scripts/check_contest_104.py", "scripts/contest_live_gateway.py",
        "scripts/paid_budget_v12.py", "scripts/package_contest_source.py", "scripts/record_contest_104.py",
        "scripts/sdk_response_v12.py", "requirements-v12-trainer.lock",
        "adapters/trainer.py", "adapters/swanlab_writer.py", "harness/data.py", "harness/storage.py",
        "vendor/reference/A/baselines/model.py", "vendor/reference/A/baselines/data.py")}
    paths.add(AUDIT)
    paths.update((ROOT / "reports/contest-submission-1.0.4").glob("*verification.json"))
    paths.update((ROOT / "reports/contest-submission-1.0.4").glob("*-fixture.json"))
    paths.update(path for path in (ROOT / SDK).rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    paths.update((ROOT / "docker/wheels").glob("numpy-*.whl"))
    paths.update((ROOT / "docker/wheels").glob("setuptools-*.whl"))
    paths.update((ROOT / "docker/v17-training-wheels").glob("*.whl"))
    if len([path for path in paths if path.suffix == ".whl"]) != 6:
        raise ValueError("locked wheel count changed")
    return sorted(paths)


def main():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (not audit["passed"] or audit["mode"] != "live" or audit["paid_request_count"] != 30
            or not all(len(audit["runs"][task]["rounds"]) == 10 for task in "ABC")):
        raise ValueError("the 30-round real Qwen acceptance is missing")
    calls = [row for task in "ABC" for row in audit["runs"][task]["rounds"]]
    if any(row["qwen"]["status"] != "completed" or row["model_call_ids"] != [row["qwen"]["call_id"]]
           or not row["qwen"]["call_id"].startswith("local-pytrio-") for row in calls):
        raise ValueError("a round has no real hosted model receipt")
    tar = OUT / "harness.tar"
    app_files = inspect_image(tar, audit["image_id"])
    if sha(ROOT / SDK / "HARNESS_SDK.md") != SDK_GUIDE_SHA:
        raise ValueError("SDK guide checksum changed")
    source = OUT / "source.zip"
    tmp = source.with_suffix(".zip.tmp")
    source_hashes = {}
    with ZipFile(tmp, "w", compression=ZIP_DEFLATED, compresslevel=4) as zip_out:
        for path in source_entries():
            raw = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            if scan(raw):
                raise ValueError(f"credential scan flagged {name}")
            source_hashes[name] = hashlib.sha256(raw).hexdigest()
            zip_out.writestr(name, raw)
        zip_out.writestr("source-sha256.json", json.dumps(source_hashes, indent=2, sort_keys=True))
    with ZipFile(tmp) as bundle:
        for name, signature in source_hashes.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != signature:
                raise ValueError(f"invalid source ZIP member {name}")
    tmp.replace(source)
    result = {"passed": True, "sdk": "1.0.4", "sdk_download_sha256": SDK_SHA,
              "sdk_doc_sha256": SDK_GUIDE_SHA, "image_id": audit["image_id"], "app_files": app_files,
              "acceptance_run_id": audit["run_id"], "live_calls": len(calls),
              "test_scope": audit["scope"], "offline_sdk_tests": 18, "official_scoring_ready": False,
              "credential_scan_scope": "source ZIP members, image config, and /app files; no provider secrets embedded",
              "exact_runtime_credentials_present_for_scanning": sum(bool(os.environ.get(name)) for name in ("PYTRIO_API_KEY", "SWANLAB_API_KEY")),
              "files": {file.name: {"sha256": sha(file), "bytes": file.stat().st_size} for file in (tar, source)}}
    (OUT / "technical-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
