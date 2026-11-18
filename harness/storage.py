"""Transactional experiment ledger and model registry."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if not path.stat().st_size:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"another process holds {path}") from exc
        yield
        # The operating system releases the lock when the descriptor closes.


class Ledger:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "ledger.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS models (
                task TEXT NOT NULL, digest TEXT NOT NULL, path TEXT NOT NULL,
                parent TEXT, experiment TEXT, metrics TEXT NOT NULL,
                PRIMARY KEY (task, digest), FOREIGN KEY (task,parent) REFERENCES models (task,digest));
            CREATE TABLE IF NOT EXISTS registry (
                task TEXT PRIMARY KEY, baseline TEXT NOT NULL, incumbent TEXT NOT NULL, previous TEXT,
                FOREIGN KEY (task,baseline) REFERENCES models (task,digest),
                FOREIGN KEY (task,incumbent) REFERENCES models (task,digest),
                FOREIGN KEY (task,previous) REFERENCES models (task,digest));
            CREATE TABLE IF NOT EXISTS trials (
                id TEXT PRIMARY KEY, task TEXT NOT NULL, parent TEXT NOT NULL, decision TEXT NOT NULL,
                data TEXT NOT NULL, FOREIGN KEY (task,parent) REFERENCES models (task,digest));
            CREATE INDEX IF NOT EXISTS trial_task ON trials (task,decision);
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY, subject TEXT NOT NULL, state TEXT NOT NULL,
                at TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY, task TEXT NOT NULL, input_digest TEXT NOT NULL, response TEXT NOT NULL);
        """)

    def close(self):
        self.db.close()

    def freeze(self, name, value):
        encoded = encode(value)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO settings VALUES (?,?)", (name, encoded))
            saved = self.db.execute("SELECT value FROM settings WHERE key=?", (name,)).fetchone()[0]
            if saved != encoded:
                raise ValueError(f"{name} changed; create a new run")

    def setting(self, name):
        row = self.db.execute("SELECT value FROM settings WHERE key=?", (name,)).fetchone()
        return json.loads(row[0]) if row is not None else None

    def log(self, subject, state, data):
        self.db.execute("INSERT INTO events (subject,state,at,data) VALUES (?,?,?,?)",
                        (subject, state, utc_now(), encode(data)))

    def register(self, task, digest, path, metrics):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO models VALUES (?,?,?,?,?,?)",
                            (task, digest, str(path), None, None, encode(metrics)))
            self.db.execute("INSERT OR IGNORE INTO registry VALUES (?,?,?,?)", (task, digest, digest, None))

    def current(self, task):
        result = self.db.execute("SELECT * FROM registry WHERE task=?", (task,)).fetchone()
        if result is None:
            raise ValueError(f"task {task} has no verified baseline in this run")
        return dict(result)

    def model(self, task, digest):
        row = self.db.execute("SELECT * FROM models WHERE task=? AND digest=?", (task, digest)).fetchone()
        if row is None:
            raise ValueError(f"unknown model {task}/{digest}")
        return {**dict(row), "path": Path(row["path"]), "metrics": json.loads(row["metrics"])}

    def history(self, task=None):
        if task is None:
            query = self.db.execute("SELECT data FROM trials ORDER BY rowid")
        else:
            query = self.db.execute("SELECT data FROM trials WHERE task=? ORDER BY rowid", (task,))
        return [json.loads(row[0]) for row in query]

    def start(self, trial):
        with self.db:
            if trial["parent"] != self.current(trial["task"])["incumbent"]:
                raise ValueError("incumbent changed before scheduling")
            self.db.execute("INSERT INTO trials VALUES (?,?,?,?,?)",
                            (trial["id"], trial["task"], trial["parent"], trial["decision"], encode(trial)))
            for state in ("received", "profiled", "planned", "validated", "running"):
                payload = {"parent": trial["parent"], "recipe": trial["proposal"]["recipe"],
                           "evidence": trial["proposal"]["evidence_refs"]}
                self.log(trial["id"], state, payload)

    def trainer_call(self, subject, event):
        with self.db:
            self.log(subject, "trainer_call", event)

    def calls(self):
        rows = self.db.execute("SELECT data FROM events WHERE state='trainer_call' ORDER BY sequence")
        return [json.loads(row[0]) for row in rows]

    def finish(self, trial, path=None):
        decision = trial["decision"]
        with self.db:
            changed = self.db.execute("UPDATE trials SET decision=?,data=? WHERE id=? AND decision='running'",
                                      (decision, encode(trial), trial["id"])).rowcount
            if changed != 1:
                raise ValueError("trial is no longer running")
            if path is not None:
                self.db.execute("INSERT OR IGNORE INTO models VALUES (?,?,?,?,?,?)",
                                (trial["task"], trial["child"], str(path), trial["parent"], trial["id"],
                                 encode(trial["metrics"])))
                self.log(trial["id"], "evaluated", {"checkpoint": trial["child"], "gain": trial["gain"]})
            if decision == "promoted":
                updated = self.db.execute("UPDATE registry SET previous=incumbent, incumbent=? WHERE task=? AND incumbent=?",
                                          (trial["child"], trial["task"], trial["parent"])).rowcount
                if updated != 1:
                    raise ValueError("incumbent changed before promotion")
            self.log(trial["id"], decision, {"reasons": trial.get("reasons", []), "child": trial.get("child")})

    def rollback(self, task, target, reason):
        with self.db:
            old = self.current(task)
            if target not in (old["previous"], old["baseline"]) or target == old["incumbent"]:
                raise ValueError("rollback target must be a previous verified model")
            self.db.execute("UPDATE registry SET incumbent=?, previous=NULL WHERE task=?", (target, task))
            self.log(task, "rolled_back", {"from": old["incumbent"], "to": target, "reason": reason})

    def cached_prediction(self, task, event_id, input_digest):
        row = self.db.execute("SELECT task,input_digest,response FROM predictions WHERE id=?", (event_id,)).fetchone()
        if row is None:
            return None
        if (row["task"], row["input_digest"]) != (task, input_digest):
            raise ValueError("event ID already refers to different inputs")
        return json.loads(row["response"])

    def record_prediction(self, task, event_id, input_digest, response):
        with self.db:
            self.db.execute("INSERT INTO predictions VALUES (?,?,?,?)", (event_id, task, input_digest, encode(response)))
            self.log(event_id, "predicted", {"task": task, "checkpoint": response["checkpoint"], "rows": len(response["predictions"])})

    def events(self, state=None):
        rows = self.db.execute("SELECT * FROM events ORDER BY sequence") if state is None else self.db.execute(
            "SELECT * FROM events WHERE state=? ORDER BY sequence", (state,))
        return [{**dict(row), "data": json.loads(row["data"])} for row in rows]
