"""Regression coverage for the standalone SDK contracts."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SDK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK_ROOT))
from contest_harness.errors import HarnessError
from contest_harness.predict import serve_predictions
from contest_harness.round import Round, _declaration
from tests.test_sdk import start_frame


class ContractRegressions(unittest.TestCase):
    def command(self, directory, command, source=None, *, input="", submission="participant"):
        if source is not None:
            (Path(directory) / "participant.py").write_text(source, encoding="utf-8")
        env = os.environ.copy()
        env.update({"PYTHONPATH": os.pathsep.join((str(directory), str(SDK_ROOT))),
                    "CONTEST_SUBMISSION": submission})
        return subprocess.run([sys.executable, "-m", "contest_harness", command],
                              input=input, text=True, capture_output=True, env=env)

    def test_interrupted_rpc_never_consumes_late_reply_or_sends_again(self):
        class Interrupted(BaseException):
            pass

        class Reader:
            def __init__(self):
                self.reads = 0
            def readline(self):
                self.reads += 1
                if self.reads == 1:
                    raise Interrupted()
                return '{"type":"history_response","id":0,"records":[]}\n'

        reader, writer = Reader(), io.StringIO()
        rnd = Round(start_frame('/train', '/validation'), reader, writer)
        with self.assertRaises(Interrupted):
            rnd.history()
        sent = writer.getvalue()
        for operation in (rnd.history, lambda: rnd.finish(
                selected_artifact_id='baseline-a', hypothesis='h', change_summary='c',
                reason='r', configuration={}, seed=1, model_call_ids=['call'])):
            rnd.call_ids = ['call']
            with self.assertRaises(HarnessError) as error:
                operation()
            self.assertEqual(error.exception.code, 'RPC_OUTCOME_UNKNOWN')
        self.assertEqual(writer.getvalue(), sent)
        self.assertEqual(reader.reads, 1)

    def test_interrupted_finish_and_write_also_poison_stream(self):
        class Interrupted(BaseException):
            pass

        class Reader:
            def readline(self):
                raise Interrupted()

        class Writer(io.StringIO):
            def flush(self):
                raise Interrupted()

        for reader, writer, finishing in ((Reader(), io.StringIO(), True),
                                           (io.StringIO(), Writer(), False)):
            rnd = Round(start_frame('/train', '/validation'), reader, writer)
            rnd.call_ids = ['call']
            with self.assertRaises(Interrupted):
                if finishing:
                    rnd.finish(selected_artifact_id='baseline-a', hypothesis='h', change_summary='c',
                               reason='r', configuration={}, seed=1)
                else:
                    rnd.history()
            sent = writer.getvalue()
            with self.assertRaises(HarnessError) as error:
                rnd.history()
            self.assertEqual(error.exception.code, 'RPC_OUTCOME_UNKNOWN')
            self.assertEqual(writer.getvalue(), sent)

    def test_nested_checkpoint_uses_artifact_root(self):
        seen = []
        def predict(sample, directory):
            seen.append(directory)
            return {"value": 1}
        output = io.StringIO()
        with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": "/model/nested/deep/checkpoint.bin"}, clear=True):
            serve_predictions(predict, io.StringIO('{"task":"regression","inputs":{}}\n'), output)
        self.assertEqual(seen, [Path("/model")])
        self.assertEqual(json.loads(output.getvalue()), {"outputs": {"value": 1}})


    def test_unicode_configuration_limit_uses_ascii_json_encoding(self):
        # JSON syntax contributes eight bytes; each escaped Chinese character six.
        valid = {"x": "中" * 10921 + "aa"}
        invalid = {"x": "中" * 10921 + "aaa"}
        self.assertEqual(len(json.dumps(valid, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()), 65536)
        self.assertEqual(len(json.dumps(invalid, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()), 65537)
        _declaration("h", "c", valid, 1)
        with self.assertRaises(HarnessError) as error:
            _declaration("h", "c", invalid, 1)
        self.assertEqual(error.exception.code, "CONFIG_LIMIT")

    def test_describe_rejects_missing_module_dependency_and_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = self.command(directory, "describe", submission="missing_contest_submission_9081")
            self.assertNotEqual(missing.returncode, 0)
            self.assertEqual(missing.stdout, "")
            dependency = self.command(directory, "describe", "import missing_contest_dependency_9081\n")
            self.assertNotEqual(dependency.returncode, 0)
            self.assertEqual(dependency.stdout, "")
            for source in ("train = 1\npredict = lambda *args: None\n",
                           "train = lambda *args: None\npredict = None\n"):
                result = self.command(directory, "describe", source)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_describe_and_predict_isolate_import_and_callback_logs(self):
        source = '''import os
print("import log")
os.write(1, b"native import log\\n")
def train(rnd):
    pass
def predict(sample, model_dir):
    print("prediction log")
    os.write(1, b"native prediction log\\n")
    return {"value": 1}
'''
        with tempfile.TemporaryDirectory() as directory:
            description = self.command(directory, "describe", source)
            self.assertEqual(description.returncode, 0, description.stderr)
            self.assertEqual(json.loads(description.stdout)["protocol"], "contest.harness.workspace.v2")
            self.assertIn("native import log", description.stderr)
            prediction = self.command(directory, "predict", source,
                                      input='{"task":"regression","inputs":{}}\n')
            self.assertEqual(prediction.returncode, 0, prediction.stderr)
            self.assertEqual(json.loads(prediction.stdout), {"outputs": {"value": 1}})
            self.assertIn("import log", prediction.stderr)
            self.assertIn("prediction log", prediction.stderr)
            self.assertIn("native prediction log", prediction.stderr)
